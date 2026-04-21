from __future__ import annotations

from dataclasses import dataclass, field, replace
from html import escape
import re
import time
from typing import Optional

import streamlit as st
import streamlit.components.v1 as components


TAB_LABELS = {
    "now_playing": "Now Playing",
    "watchlist": "Watchlist",
    "playlist": "Playlist",
    "history": "History",
}

STATUS_LABELS = {
    "idle": "IDLE",
    "playing": "PLAYING",
    "paused": "PAUSED",
}

STATUS_COLORS = {
    "idle": "#6B7280",
    "playing": "#374151",
    "paused": "#71717A",
}


@dataclass
class MediaItem:
    title: str
    subtitle: str
    media_type: str
    duration_label: str
    duration_seconds: int
    progress_pct: int
    icon: str


@dataclass
class MediaCenterState:
    active_tab: str = "now_playing"
    status: str = "idle"
    mode: str = "none"
    current_item: MediaItem = field(default_factory=lambda: empty_media_item())
    volume: int = 70
    shuffle: bool = False
    watchlist: list[MediaItem] = field(default_factory=list)
    playlist: list[MediaItem] = field(default_factory=list)
    history: list[MediaItem] = field(default_factory=list)
    feedback_message: str = "Media controller ready."
    playback_started_at: Optional[float] = None
    playback_elapsed_seconds: int = 0


class MediaCenterController:
    """Pipeline-facing adapter around the Streamlit media-center draft."""

    def __init__(self) -> None:
        self.state = build_initial_state()

    def play_music(self, song_title: str, artist_name: Optional[str] = None, album: Optional[str] = None, duration_ms: Optional[int] = None) -> str:
        item = _music_item(song_title, artist_name, album=album, duration_ms=duration_ms)
        _start_playback(self.state, item, f"Now playing {item.title}.")
        return self.state.feedback_message

    def play_movie(
        self,
        title: str,
        year: Optional[str] = None,
        rating: Optional[str] = None,
        genre: Optional[str] = None,
        runtime_min: Optional[int] = None,
    ) -> str:
        item = _movie_item(title, year=year, rating=rating, genre=genre, runtime_min=runtime_min)
        _start_playback(self.state, item, f"Now playing {item.title}.")
        return self.state.feedback_message

    def pause(self) -> str:
        if self.state.status == "playing":
            _freeze_playback(self.state)
            self.state.status = "paused"
            self.state.feedback_message = f"Paused {self.state.current_item.title}."
        elif self.state.status == "paused":
            self.state.feedback_message = f"{self.state.current_item.title} is already paused."
        else:
            self.state.feedback_message = "Nothing is playing right now."
        return self.state.feedback_message

    def resume(self) -> str:
        if self.state.status == "paused":
            self.state.status = "playing"
            self.state.playback_started_at = time.time()
            self.state.feedback_message = f"Resumed {self.state.current_item.title}."
        else:
            self.state.feedback_message = "Nothing is paused right now."
        return self.state.feedback_message

    def stop(self) -> str:
        if self.state.status == "idle" and self.state.current_item.media_type == "none":
            self.state.feedback_message = "Nothing is playing right now."
        else:
            stopped_title = self.state.current_item.title
            self.state.current_item = empty_media_item()
            self.state.status = "idle"
            self.state.mode = "none"
            _reset_playback_clock(self.state)
            self.state.feedback_message = f"Stopped {stopped_title}."
        return self.state.feedback_message

    def next_track(self) -> str:
        if not self.state.playlist:
            self.state.feedback_message = "Playlist is empty."
            return self.state.feedback_message
        item = _playlist_neighbor(self.state, step=1)
        _start_playback(self.state, item, f"Advanced to {item.title}.")
        return self.state.feedback_message

    def change_volume(self, value: str | int, modifier: str = "set") -> str:
        volume = self._parse_volume(value)
        if modifier == "decrease":
            new_volume = max(0, self.state.volume - volume)
            self.state.volume = new_volume
            self.state.feedback_message = f"Decreased volume by {volume}%, now at {new_volume}%."
        elif modifier == "increase":
            new_volume = min(100, self.state.volume + volume)
            self.state.volume = new_volume
            self.state.feedback_message = f"Increased volume by {volume}%, now at {new_volume}%."
        else:  # set
            self.state.volume = volume
            self.state.feedback_message = f"Volume set to {volume}%."
        return self.state.feedback_message

    def add_to_watchlist(self, title: str) -> str:
        title = title or "movie"
        item = (
            clone_item(self.state.current_item)
            if title.lower() in {"movie", "current", "this"} and self.state.current_item.media_type == "movie"
            else _movie_item(title)
        )
        if _contains_item(self.state.watchlist, item):
            self.state.feedback_message = f"{item.title} is already in the watchlist."
        else:
            self.state.watchlist.append(item)
            self.state.feedback_message = f"Added {item.title} to the watchlist."
        return self.state.feedback_message

    def add_to_playlist(self, song_title: str, artist_name: Optional[str] = None) -> str:
        song_title = song_title or "song"
        item = (
            clone_item(self.state.current_item)
            if song_title.lower() in {"song", "current", "this"} and self.state.current_item.media_type == "music"
            else _music_item(song_title, artist_name)
        )
        if _contains_item(self.state.playlist, item):
            self.state.feedback_message = f"{item.title} is already in the playlist."
        else:
            self.state.playlist.append(item)
            self.state.feedback_message = f"Added {item.title} to the playlist."
        return self.state.feedback_message

    def shuffle_playlist(self) -> str:
        self.state.shuffle = not self.state.shuffle
        self.state.feedback_message = "Shuffle enabled." if self.state.shuffle else "Shuffle disabled."
        return self.state.feedback_message

    def get_status_dict(self) -> dict:
        item = self.state.current_item
        elapsed_seconds = _current_elapsed_seconds(self.state)
        progress = round(_progress_from_elapsed(item.duration_seconds, elapsed_seconds), 2)
        return {
            "mode": self.state.status,
            "current_title": item.title if item.media_type != "none" else None,
            "current_type": item.media_type if item.media_type != "none" else None,
            "current_artist": _artist_from_subtitle(item),
            "volume": self.state.volume,
            "shuffle": self.state.shuffle,
            "playlist_count": len(self.state.playlist),
            "watchlist_count": len(self.state.watchlist),
            "history_count": len(self.state.history),
            "elapsed_seconds": elapsed_seconds,
            "progress": progress,
        }

    @staticmethod
    def _parse_volume(value: str | int) -> int:
        if isinstance(value, int):
            return max(0, min(100, value))

        text = str(value).lower()
        if "max" in text:
            return 100
        if "min" in text or "mute" in text:
            return 0
        if "louder" in text or "up" in text:
            return 80
        if "quieter" in text or "down" in text:
            return 40
        match = re.search(r"(\d{1,3})", text)
        if match:
            return max(0, min(100, int(match.group(1))))
        return 70


MUSIC_SAMPLE = MediaItem(
    title="Blinding Lights",
    subtitle="The Weeknd | After Hours",
    media_type="music",
    duration_label="3:20",
    duration_seconds=200,
    progress_pct=42,
    icon="TRACK",
)

MOVIE_SAMPLE = MediaItem(
    title="Inception",
    subtitle="2010 | Sci-fi thriller | IMDb 8.8",
    media_type="movie",
    duration_label="2:28:00",
    duration_seconds=8880,
    progress_pct=18,
    icon="FILM",
)

WATCHLIST_SEED = [
    MediaItem(
        title="Interstellar",
        subtitle="2014 | Sci-fi drama",
        media_type="movie",
        duration_label="2:49:00",
        duration_seconds=10140,
        progress_pct=0,
        icon="FILM",
    ),
    MediaItem(
        title="Blade Runner 2049",
        subtitle="2017 | Neo-noir sci-fi",
        media_type="movie",
        duration_label="2:44:00",
        duration_seconds=9840,
        progress_pct=0,
        icon="FILM",
    ),
]

PLAYLIST_SEED = [
    MUSIC_SAMPLE,
    MediaItem(
        title="Starboy",
        subtitle="The Weeknd | Starboy",
        media_type="music",
        duration_label="3:50",
        duration_seconds=230,
        progress_pct=15,
        icon="TRACK",
    ),
    MediaItem(
        title="Midnight City",
        subtitle="M83 | Hurry Up, We're Dreaming",
        media_type="music",
        duration_label="4:03",
        duration_seconds=243,
        progress_pct=9,
        icon="TRACK",
    ),
]

HISTORY_SEED = [
    MediaItem(
        title="Dune: Part Two",
        subtitle="Movie preview loaded",
        media_type="movie",
        duration_label="2:46:00",
        duration_seconds=9960,
        progress_pct=6,
        icon="FILM",
    ),
    MediaItem(
        title="Get Lucky",
        subtitle="Daft Punk ft. Pharrell Williams",
        media_type="music",
        duration_label="4:08",
        duration_seconds=248,
        progress_pct=27,
        icon="TRACK",
    ),
]


def empty_media_item() -> MediaItem:
    return MediaItem(
        title="Nothing playing",
        subtitle="Choose a demo action to preview the smart media controller.",
        media_type="none",
        duration_label="0:00",
        duration_seconds=0,
        progress_pct=0,
        icon="MEDIA",
    )


def clone_item(item: MediaItem) -> MediaItem:
    return replace(item)


def _format_music_duration(duration_ms: Optional[int]) -> tuple[str, int]:
    """Convert milliseconds to (label, seconds). Falls back to 3:30."""
    if duration_ms and duration_ms > 0:
        total_s = duration_ms // 1000
        m, s = divmod(total_s, 60)
        return f"{m}:{s:02d}", total_s
    return "3:30", 210


def _format_movie_duration(runtime_min: Optional[int]) -> tuple[str, int]:
    """Convert minutes to (label, seconds). Falls back to 2:00:00."""
    if runtime_min and runtime_min > 0:
        h, m = divmod(runtime_min, 60)
        label = f"{h}:{m:02d}:00" if h else f"{m}:00"
        return label, runtime_min * 60
    return "2:00:00", 7200


def _music_item(title: str, artist_name: Optional[str] = None, album: Optional[str] = None, duration_ms: Optional[int] = None) -> MediaItem:
    clean_title = (title or "song").strip() or "song"
    dur_label, dur_secs = _format_music_duration(duration_ms)

    if clean_title.lower() == MUSIC_SAMPLE.title.lower():
        item = clone_item(MUSIC_SAMPLE)
        if artist_name or album:
            parts = [p for p in (artist_name, album) if p]
            item.subtitle = " | ".join(parts)
        if duration_ms:
            item.duration_label = dur_label
            item.duration_seconds = dur_secs
        return item

    parts = [p for p in (artist_name, album) if p]
    subtitle = " | ".join(parts) if parts else ""
    return MediaItem(
        title=clean_title,
        subtitle=subtitle,
        media_type="music",
        duration_label=dur_label,
        duration_seconds=dur_secs,
        progress_pct=0,
        icon="TRACK",
    )


def _movie_item(
    title: str,
    year: Optional[str] = None,
    rating: Optional[str] = None,
    genre: Optional[str] = None,
    runtime_min: Optional[int] = None,
) -> MediaItem:
    clean_title = (title or "movie").strip() or "movie"
    dur_label, dur_secs = _format_movie_duration(runtime_min)

    if clean_title.lower() == MOVIE_SAMPLE.title.lower():
        item = clone_item(MOVIE_SAMPLE)
    else:
        item = MediaItem(
            title=clean_title,
            subtitle="",
            media_type="movie",
            duration_label=dur_label,
            duration_seconds=dur_secs,
            progress_pct=0,
            icon="FILM",
        )

    if runtime_min:
        item.duration_label = dur_label
        item.duration_seconds = dur_secs

    parts = [part for part in (year, genre, f"IMDb {rating}" if rating and rating != "N/A" else None) if part]
    if parts:
        item.subtitle = " | ".join(parts)
    return item


def _artist_from_subtitle(item: MediaItem) -> Optional[str]:
    if item.media_type != "music" or not item.subtitle:
        return None
    return item.subtitle.split("|", 1)[0].strip() or None


def build_initial_state() -> MediaCenterState:
    return MediaCenterState(
        current_item=empty_media_item(),
        watchlist=[clone_item(item) for item in WATCHLIST_SEED],
        playlist=[clone_item(item) for item in PLAYLIST_SEED],
        history=[clone_item(item) for item in HISTORY_SEED],
        feedback_message="Media controller ready. Try the demo actions below.",
    )


def init_media_center_state(media_controller: Optional[MediaCenterController] = None) -> None:
    if media_controller is not None:
        st.session_state.media_center = media_controller.state
    elif "media_center" not in st.session_state:
        st.session_state.media_center = build_initial_state()

    state: MediaCenterState = st.session_state.media_center
    if "media_center_volume" not in st.session_state:
        st.session_state.media_center_volume = state.volume
    if "media_center_last_state_volume" not in st.session_state:
        st.session_state.media_center_last_state_volume = state.volume


def _sync_volume_widget(state: MediaCenterState) -> None:
    state_volume = max(0, min(100, int(state.volume)))
    widget_volume = max(0, min(100, int(st.session_state.get("media_center_volume", state_volume))))
    last_state_volume = max(
        0,
        min(100, int(st.session_state.get("media_center_last_state_volume", state_volume))),
    )

    if state_volume != last_state_volume:
        st.session_state.media_center_volume = state_volume
    else:
        state.volume = widget_volume


def handle_media_action(action: str, media_controller: Optional[MediaCenterController] = None) -> None:
    init_media_center_state(media_controller)
    state: MediaCenterState = media_controller.state if media_controller is not None else st.session_state.media_center
    _sync_playback_clock(state)

    if action.startswith("tab:"):
        tab = action.split(":", 1)[1]
        if tab in TAB_LABELS:
            state.active_tab = tab
        return

    if action == "play_music":
        _start_playback(state, MUSIC_SAMPLE, "Now playing Blinding Lights.")
        return

    if action == "play_movie":
        _start_playback(state, MOVIE_SAMPLE, "Now playing Inception.")
        return

    if action == "toggle_play":
        if state.status == "playing":
            _freeze_playback(state)
            state.status = "paused"
            state.feedback_message = f"Paused {state.current_item.title}."
        elif state.status == "paused":
            state.status = "playing"
            state.playback_started_at = time.time()
            state.feedback_message = f"Resumed {state.current_item.title}."
        elif state.playlist:
            _start_playback(state, state.playlist[0], f"Started {state.playlist[0].title}.")
        else:
            state.feedback_message = "Nothing is queued yet."
        return

    if action == "pause":
        if state.status == "playing":
            _freeze_playback(state)
            state.status = "paused"
            state.feedback_message = f"Paused {state.current_item.title}."
        elif state.status == "paused":
            state.feedback_message = f"{state.current_item.title} is already paused."
        else:
            state.feedback_message = "Nothing is playing right now."
        return

    if action == "stop":
        if state.status == "idle" and state.current_item.media_type == "none":
            state.feedback_message = "Nothing is playing right now."
        else:
            stopped_title = state.current_item.title
            state.current_item = empty_media_item()
            state.status = "idle"
            state.mode = "none"
            _reset_playback_clock(state)
            state.feedback_message = f"Stopped {stopped_title}."
        return

    if action == "next":
        if not state.playlist:
            state.feedback_message = "Playlist is empty."
            return
        next_item = _playlist_neighbor(state, step=1)
        _start_playback(state, next_item, f"Advanced to {next_item.title}.")
        return

    if action == "previous":
        if not state.playlist:
            state.feedback_message = "Playlist is empty."
            return
        previous_item = _playlist_neighbor(state, step=-1)
        _start_playback(state, previous_item, f"Loaded {previous_item.title}.")
        return

    if action == "shuffle":
        state.shuffle = not state.shuffle
        state.feedback_message = "Shuffle enabled." if state.shuffle else "Shuffle disabled."
        return

    if action == "add_watchlist":
        if state.current_item.media_type != "movie":
            state.feedback_message = "Play a movie before adding it to the watchlist."
            return
        if _contains_item(state.watchlist, state.current_item):
            state.feedback_message = f"{state.current_item.title} is already in the watchlist."
            return
        state.watchlist.append(clone_item(state.current_item))
        state.feedback_message = f"Added {state.current_item.title} to the watchlist."
        return

    if action == "add_playlist":
        if state.current_item.media_type != "music":
            state.feedback_message = "Play a song before adding it to the playlist."
            return
        if _contains_item(state.playlist, state.current_item):
            state.feedback_message = f"{state.current_item.title} is already in the playlist."
            return
        state.playlist.append(clone_item(state.current_item))
        state.feedback_message = f"Added {state.current_item.title} to the playlist."
        return


def render_media_center(media_controller: Optional[MediaCenterController] = None) -> None:
    init_media_center_state(media_controller)
    state: MediaCenterState = media_controller.state if media_controller is not None else st.session_state.media_center
    _sync_playback_clock(state)

    st.markdown(_media_center_styles(), unsafe_allow_html=True)

    _sync_volume_widget(state)

    header_cols = st.columns([1.2, 1.25, 1.15, 1.15, 1.0, 0.9], gap="small")
    with header_cols[0]:
        st.markdown('<div class="media-center-label">MEDIA CENTER</div>', unsafe_allow_html=True)
    _render_tab_button(header_cols[1], "now_playing", state, media_controller)
    _render_tab_button(header_cols[2], "watchlist", state, media_controller)
    _render_tab_button(header_cols[3], "playlist", state, media_controller)
    _render_tab_button(header_cols[4], "history", state, media_controller)
    with header_cols[5]:
        st.markdown(_status_badge_html(state.status), unsafe_allow_html=True)

    st.markdown(_current_item_card_html(state.current_item), unsafe_allow_html=True)
    components.html(_progress_bar_html(state), height=34, scrolling=False)

    transport_cols = st.columns([2.2, 0.9, 1.05, 0.9, 0.9, 1.05, 2.2], gap="small")
    with transport_cols[1]:
        st.button(
            "⏮",
            key="media_control_previous",
            use_container_width=True,
            on_click=handle_media_action,
            args=("previous", media_controller),
            help="Previous",
        )
    with transport_cols[2]:
        st.button(
            _play_button_label(state.status),
            key="media_control_toggle",
            use_container_width=True,
            type="primary",
            on_click=handle_media_action,
            args=("toggle_play", media_controller),
            help="Play or pause",
        )
    with transport_cols[3]:
        st.button(
            "⏭",
            key="media_control_next",
            use_container_width=True,
            on_click=handle_media_action,
            args=("next", media_controller),
            help="Next",
        )
    with transport_cols[4]:
        st.button(
            "⏹",
            key="media_control_stop",
            use_container_width=True,
            on_click=handle_media_action,
            args=("stop", media_controller),
            help="Stop",
        )
    with transport_cols[5]:
        st.button(
            "⇄",
            key="media_control_shuffle",
            use_container_width=True,
            type="primary" if state.shuffle else "secondary",
            on_click=handle_media_action,
            args=("shuffle", media_controller),
            help="Shuffle",
        )

    st.markdown(_feedback_html(state.feedback_message), unsafe_allow_html=True)

    volume_cols = st.columns([0.7, 6.2], gap="small")
    with volume_cols[0]:
        st.markdown('<div class="media-volume-label">VOL</div>', unsafe_allow_html=True)
    with volume_cols[1]:
        state.volume = int(st.slider(
            "Volume",
            min_value=0,
            max_value=100,
            key="media_center_volume",
            label_visibility="collapsed",
        ))
        st.session_state.media_center_last_state_volume = state.volume

    demo_cols = st.columns(5, gap="small")
    demo_actions = [
        ("Play Blinding Lights", "play_music"),
        ("Play Inception", "play_movie"),
        ("+ Watchlist", "add_watchlist"),
        ("+ Playlist", "add_playlist"),
        ("Shuffle", "shuffle"),
    ]
    for column, (label, action) in zip(demo_cols, demo_actions):
        with column:
            st.button(
                label,
                key=f"media_demo_{action}",
                use_container_width=True,
                on_click=handle_media_action,
                args=(action, media_controller),
            )

    if state.active_tab == "now_playing":
        st.markdown(
            _collection_html(
                title="Queue Preview",
                items=state.playlist,
                current_item=state.current_item,
                empty_message="Add a song to build the playlist.",
            ),
            unsafe_allow_html=True,
        )
    elif state.active_tab == "watchlist":
        st.markdown(
            _collection_html(
                title="Watchlist",
                items=state.watchlist,
                current_item=state.current_item,
                empty_message="No movies saved yet.",
            ),
            unsafe_allow_html=True,
        )
    elif state.active_tab == "playlist":
        st.markdown(
            _collection_html(
                title="Playlist",
                items=state.playlist,
                current_item=state.current_item,
                empty_message="No songs queued yet.",
            ),
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            _collection_html(
                title="History",
                items=state.history,
                current_item=state.current_item,
                empty_message="Playback history will appear here.",
            ),
            unsafe_allow_html=True,
        )


def _render_tab_button(
    column,
    tab: str,
    state: MediaCenterState,
    media_controller: Optional[MediaCenterController] = None,
) -> None:
    label = TAB_LABELS[tab]
    if tab == "watchlist":
        label = f"{label} {len(state.watchlist)}"
    elif tab == "playlist":
        label = f"{label} {len(state.playlist)}"

    with column:
        st.button(
            label,
            key=f"media_tab_{tab}",
            use_container_width=True,
            type="primary" if state.active_tab == tab else "secondary",
            on_click=handle_media_action,
            args=(f"tab:{tab}", media_controller),
        )


def _start_playback(state: MediaCenterState, item: MediaItem, feedback_message: str) -> None:
    incoming_item = clone_item(item)
    incoming_item.progress_pct = 0
    if not _same_item(state.current_item, incoming_item):
        _push_history(state, incoming_item)
    state.current_item = incoming_item
    state.status = "playing"
    state.mode = incoming_item.media_type
    state.active_tab = "now_playing"
    state.playback_elapsed_seconds = 0
    state.playback_started_at = time.time()
    state.feedback_message = feedback_message


def _playlist_neighbor(state: MediaCenterState, step: int) -> MediaItem:
    playlist = state.playlist
    if not playlist:
        return empty_media_item()

    current_index = 0
    for idx, item in enumerate(playlist):
        if _same_item(item, state.current_item):
            current_index = idx
            break

    if state.current_item.media_type == "none":
        return clone_item(playlist[0])

    next_index = (current_index + step) % len(playlist)
    return clone_item(playlist[next_index])


def _push_history(state: MediaCenterState, item: MediaItem) -> None:
    if item.media_type == "none":
        return
    if state.history and _same_item(state.history[0], item):
        return
    state.history.insert(0, clone_item(item))
    del state.history[8:]


def _contains_item(items: list[MediaItem], candidate: MediaItem) -> bool:
    return any(_same_item(item, candidate) for item in items)


def _same_item(left: MediaItem, right: MediaItem) -> bool:
    return left.title == right.title and left.media_type == right.media_type


def _play_button_label(status: str) -> str:
    if status == "playing":
        return "⏸"
    if status == "paused":
        return "▶"
    return "▶"


def _status_badge_html(status: str) -> str:
    label = STATUS_LABELS.get(status, status.upper())
    color = STATUS_COLORS.get(status, STATUS_COLORS["idle"])
    return (
        '<div class="media-status-badge" '
        f'style="border-color:{color}; color:{color};">{escape(label)}</div>'
    )


def _current_item_card_html(item: MediaItem) -> str:
    mode_label = item.media_type.upper() if item.media_type != "none" else "STANDBY"
    return f"""
    <div class="media-current-card">
        <div class="media-art-tile">{escape(item.icon)}</div>
        <div class="media-copy-block">
            <div class="media-copy-kicker">{escape(mode_label)}</div>
            <div class="media-copy-title">{escape(item.title)}</div>
            <div class="media-copy-subtitle">{escape(item.subtitle)}</div>
        </div>
    </div>
    """


def _current_elapsed_seconds(state: MediaCenterState, now: Optional[float] = None) -> int:
    item = state.current_item
    if item.duration_seconds <= 0 or item.media_type == "none":
        return 0

    elapsed = max(0, int(state.playback_elapsed_seconds))
    if state.status == "playing" and state.playback_started_at is not None:
        elapsed += int((now or time.time()) - state.playback_started_at)

    return min(elapsed, item.duration_seconds)


def _sync_playback_clock(state: MediaCenterState) -> None:
    if state.current_item.media_type == "none":
        _reset_playback_clock(state)
        return

    elapsed = _current_elapsed_seconds(state)
    state.playback_elapsed_seconds = elapsed
    state.current_item.progress_pct = int(
        _progress_from_elapsed(state.current_item.duration_seconds, elapsed) * 100
    )
    if state.status == "playing" and elapsed < state.current_item.duration_seconds:
        state.playback_started_at = time.time()
    elif state.status == "playing":
        state.playback_started_at = None
    else:
        state.playback_started_at = None


def _freeze_playback(state: MediaCenterState) -> None:
    elapsed = _current_elapsed_seconds(state)
    state.playback_elapsed_seconds = elapsed
    state.current_item.progress_pct = int(
        _progress_from_elapsed(state.current_item.duration_seconds, elapsed) * 100
    )
    state.playback_started_at = None


def _reset_playback_clock(state: MediaCenterState) -> None:
    state.playback_started_at = None
    state.playback_elapsed_seconds = 0


def _elapsed_from_progress(duration_seconds: int, progress_pct: int) -> int:
    if duration_seconds <= 0 or progress_pct <= 0:
        return 0
    return min(duration_seconds, int(duration_seconds * (progress_pct / 100.0)))


def _progress_from_elapsed(duration_seconds: int, elapsed_seconds: int) -> float:
    if duration_seconds <= 0:
        return 0.0
    return min(1.0, max(0.0, elapsed_seconds / duration_seconds))


def _progress_bar_html(state: MediaCenterState) -> str:
    item = state.current_item
    elapsed_seconds = _current_elapsed_seconds(state)
    progress_pct = _progress_from_elapsed(item.duration_seconds, elapsed_seconds) * 100
    is_playing = "true" if state.status == "playing" and item.media_type != "none" else "false"
    duration_seconds = max(0, item.duration_seconds)
    return f"""
    <style>
    html, body {{
        margin: 0;
        padding: 0;
        background: transparent;
        overflow: hidden;
    }}
    .media-progress-wrap {{
        display: grid;
        grid-template-columns: auto 1fr auto;
        align-items: center;
        gap: 0.55rem;
        margin: 0.45rem 0 0.2rem;
        font-family: "Source Sans Pro", sans-serif;
    }}
    .media-progress-label {{
        color: #6B7280;
        font-size: 0.76rem;
        font-weight: 600;
        min-width: 3.2rem;
    }}
    .media-progress-track {{
        position: relative;
        height: 0.36rem;
        border-radius: 999px;
        background: #E5E7EB;
        overflow: hidden;
    }}
    .media-progress-fill {{
        height: 100%;
        border-radius: 999px;
        background: #6B7280;
        transition: width 0.2s linear;
    }}
    </style>
    <div class="media-progress-wrap">
        <div id="media_elapsed" class="media-progress-label">{escape(_format_timestamp(elapsed_seconds))}</div>
        <div class="media-progress-track">
            <div id="media_progress_fill" class="media-progress-fill" style="width:{progress_pct:.2f}%;"></div>
        </div>
        <div class="media-progress-label">{escape(item.duration_label)}</div>
    </div>
    <script>
    const mediaStartElapsed = {elapsed_seconds};
    const mediaDuration = {duration_seconds};
    const mediaIsPlaying = {is_playing};
    const mediaClientStartedAt = Date.now();

    function mediaFormatTimestamp(totalSeconds) {{
        totalSeconds = Math.max(0, Math.floor(totalSeconds));
        const hours = Math.floor(totalSeconds / 3600);
        const minutes = Math.floor((totalSeconds % 3600) / 60);
        const seconds = totalSeconds % 60;
        if (hours > 0) {{
            return `${{hours}}:${{String(minutes).padStart(2, "0")}}:${{String(seconds).padStart(2, "0")}}`;
        }}
        return `${{minutes}}:${{String(seconds).padStart(2, "0")}}`;
    }}

    function mediaRenderProgress() {{
        let elapsed = mediaStartElapsed;
        if (mediaIsPlaying && mediaDuration > 0) {{
            elapsed += Math.floor((Date.now() - mediaClientStartedAt) / 1000);
        }}
        elapsed = Math.min(elapsed, mediaDuration);
        document.getElementById("media_elapsed").textContent = mediaFormatTimestamp(elapsed);
        const pct = mediaDuration > 0 ? (elapsed / mediaDuration) * 100 : 0;
        document.getElementById("media_progress_fill").style.width = `${{pct}}%`;
        if (!mediaIsPlaying || elapsed >= mediaDuration) {{
            window.clearInterval(mediaProgressTimer);
        }}
    }}

    const mediaProgressTimer = window.setInterval(mediaRenderProgress, 1000);
    mediaRenderProgress();
    </script>
    """


def _feedback_html(message: str) -> str:
    return f'<div class="media-feedback">{escape(message)}</div>'


def _collection_html(
    title: str,
    items: list[MediaItem],
    current_item: MediaItem,
    empty_message: str,
) -> str:
    if not items:
        body = f'<div class="media-empty">{escape(empty_message)}</div>'
    else:
        item_markup = "".join(
            _collection_item_html(item, current_item)
            for item in items
        )
        body = f'<div class="media-collection-grid">{item_markup}</div>'

    return f"""
    <div class="media-section-title">{escape(title)}</div>
    {body}
    """


def _collection_item_html(item: MediaItem, current_item: MediaItem) -> str:
    item_class = "media-collection-item media-collection-item-active" if _same_item(item, current_item) else "media-collection-item"
    return f"""
    <div class="{item_class}">
        <div class="media-chip-icon">{escape(item.icon)}</div>
        <div class="media-chip-copy">
            <div class="media-chip-title">{escape(item.title)}</div>
            <div class="media-chip-subtitle">{escape(item.subtitle)}</div>
        </div>
        <div class="media-chip-duration">{escape(item.duration_label)}</div>
    </div>
    """


def _elapsed_label(duration_seconds: int, progress_pct: int) -> str:
    if duration_seconds <= 0 or progress_pct <= 0:
        return "0:00"
    elapsed_seconds = int(duration_seconds * (progress_pct / 100.0))
    return _format_timestamp(elapsed_seconds)


def _format_timestamp(total_seconds: int) -> str:
    hours, remainder = divmod(max(total_seconds, 0), 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    return f"{minutes}:{seconds:02d}"


def _media_center_styles() -> str:
    return """
    <style>
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1.2rem;
    }

    div.stButton button {
        min-height: 2.35rem;
        padding: 0.28rem 0.7rem;
        font-size: 0.95rem;
    }

    div.stButton button[kind="primary"],
    div.stButton button[data-testid="baseButton-primary"],
    div.stButton button[data-testid="stBaseButton-primary"] {
        background: #F3F4F6;
        border: 1px solid #9CA3AF;
        color: #374151;
        box-shadow: none;
    }

    div.stButton button[kind="primary"]:hover,
    div.stButton button[data-testid="baseButton-primary"]:hover,
    div.stButton button[data-testid="stBaseButton-primary"]:hover {
        background: #E5E7EB;
        border-color: #6B7280;
        color: #111827;
    }

    div.stButton button[kind="primary"]:focus,
    div.stButton button[data-testid="baseButton-primary"]:focus,
    div.stButton button[data-testid="stBaseButton-primary"]:focus {
        border-color: #6B7280;
        box-shadow: 0 0 0 0.2rem rgba(107, 114, 128, 0.16);
    }

    div.stButton button[kind="secondary"],
    div.stButton button[data-testid="baseButton-secondary"],
    div.stButton button[data-testid="stBaseButton-secondary"] {
        background: #FFFFFF;
        border: 1px solid #D1D5DB;
        color: #374151;
        box-shadow: none;
    }

    div.stButton button[kind="secondary"]:hover,
    div.stButton button[data-testid="baseButton-secondary"]:hover,
    div.stButton button[data-testid="stBaseButton-secondary"]:hover {
        background: #F9FAFB;
        border-color: #9CA3AF;
        color: #111827;
    }

    div.stButton button:disabled {
        opacity: 0.55;
        border-color: #E5E7EB;
    }

    .media-center-label {
        padding-top: 0.42rem;
        color: #6B7280;
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.16rem;
        text-transform: uppercase;
    }

    .media-status-badge {
        margin-top: 0.08rem;
        border: 1px solid #D1D5DB;
        border-radius: 999px;
        padding: 0.45rem 0.65rem;
        text-align: center;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.08rem;
        background: #FFFFFF;
    }

    .media-current-card {
        display: flex;
        gap: 0.8rem;
        align-items: center;
        margin-top: 0.55rem;
        padding: 0.8rem 0.85rem;
        border: 1px solid #E5E7EB;
        border-radius: 1rem;
        background: #FFFFFF;
        box-shadow: 0 1px 2px rgba(17, 24, 39, 0.04);
    }

    .media-art-tile {
        min-width: 4.6rem;
        min-height: 4.6rem;
        display: flex;
        align-items: center;
        justify-content: center;
        border: 1px solid #D1D5DB;
        border-radius: 0.9rem;
        background: #F3F4F6;
        color: #374151;
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0.08rem;
    }

    .media-copy-block {
        min-width: 0;
    }

    .media-copy-kicker {
        color: #6B7280;
        font-size: 0.69rem;
        font-weight: 700;
        letter-spacing: 0.12rem;
        text-transform: uppercase;
    }

    .media-copy-title {
        margin-top: 0.1rem;
        color: #111827;
        font-size: 1.28rem;
        font-weight: 700;
    }

    .media-copy-subtitle {
        margin-top: 0.12rem;
        color: #6B7280;
        font-size: 0.86rem;
    }

    .media-progress-wrap {
        display: grid;
        grid-template-columns: auto 1fr auto;
        align-items: center;
        gap: 0.55rem;
        margin: 0.45rem 0 0.2rem;
    }

    .media-progress-label {
        color: #6B7280;
        font-size: 0.76rem;
        font-weight: 600;
        min-width: 3.2rem;
    }

    .media-progress-track {
        position: relative;
        height: 0.36rem;
        border-radius: 999px;
        background: #E5E7EB;
        overflow: hidden;
    }

    .media-progress-fill {
        height: 100%;
        border-radius: 999px;
        background: #6B7280;
    }

    .media-feedback {
        margin-top: 0.45rem;
        margin-bottom: 0.05rem;
        padding: 0.5rem 0.7rem;
        border: 1px solid #E5E7EB;
        border-radius: 0.75rem;
        background: #F9FAFB;
        color: #374151;
        font-size: 0.84rem;
    }

    .media-volume-label {
        padding-top: 0.25rem;
        color: #6B7280;
        font-size: 0.74rem;
        font-weight: 700;
        letter-spacing: 0.08rem;
        text-transform: uppercase;
        text-align: center;
    }

    .stSlider {
        padding-top: 0.08rem;
    }

    .stSlider [data-baseweb="slider"] > div > div {
        background: #E5E7EB !important;
    }

    .stSlider [data-baseweb="slider"] > div > div > div {
        background: #6B7280 !important;
    }

    .stSlider [data-baseweb="slider"] [role="slider"] {
        background: #374151 !important;
        border-color: #FFFFFF !important;
        box-shadow: 0 0 0 2px rgba(17, 24, 39, 0.14) !important;
    }

    .stSlider [data-baseweb="slider"] [role="slider"] + div,
    .stSlider [data-baseweb="slider"] [data-testid="stSliderThumbValue"],
    .stSlider [data-baseweb="slider"] [data-testid="stTooltipContent"],
    .stSlider [data-baseweb="slider"] div[style*="color"],
    .stSlider [data-baseweb="slider"] span {
        color: #374151 !important;
        font-weight: 700;
    }

    .media-section-title {
        margin-top: 0.6rem;
        margin-bottom: 0.3rem;
        color: #6B7280;
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.1rem;
        text-transform: uppercase;
    }

    .media-collection-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
        gap: 0.45rem;
        margin: 0.1rem 0 0.9rem;
    }

    .media-collection-item {
        display: grid;
        grid-template-columns: auto 1fr auto;
        gap: 0.65rem;
        align-items: center;
        padding: 0.65rem 0.75rem;
        border: 1px solid #E5E7EB;
        border-radius: 0.8rem;
        background: #FFFFFF;
    }

    .media-collection-item-active {
        border-color: #9CA3AF;
        background: #F9FAFB;
        box-shadow: inset 0 0 0 1px rgba(156, 163, 175, 0.35);
    }

    .media-chip-icon {
        min-width: 2.75rem;
        padding: 0.52rem 0.45rem;
        border-radius: 0.7rem;
        background: #F3F4F6;
        color: #374151;
        font-size: 0.66rem;
        font-weight: 700;
        text-align: center;
        letter-spacing: 0.06rem;
    }

    .media-chip-copy {
        min-width: 0;
    }

    .media-chip-title {
        color: #111827;
        font-size: 0.9rem;
        font-weight: 700;
    }

    .media-chip-subtitle {
        color: #6B7280;
        font-size: 0.76rem;
        margin-top: 0.08rem;
    }

    .media-chip-duration {
        color: #6B7280;
        font-size: 0.74rem;
        font-weight: 700;
    }

    .media-empty {
        padding: 0.75rem 0.85rem;
        margin-bottom: 0.9rem;
        border: 1px dashed #D1D5DB;
        border-radius: 0.8rem;
        color: #6B7280;
        background: #F9FAFB;
        font-size: 0.82rem;
    }
    </style>
    """
