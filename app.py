"""Atlas – Voice Assistant  |  Full pipeline Streamlit UI."""
from __future__ import annotations

import json
import os
import re
from textwrap import dedent
import time as _time
from pathlib import Path

import requests
import streamlit as st
import streamlit.components.v1 as _components

from nlu.inference import JointNLUPredictor
from nlu.utils import INTENTS
from src.answer_generation import generate_answer
from src.audio import audio_input_to_numpy, capture_microphone, is_hf_mode

_HF_MODE = is_hf_mode()

if _HF_MODE:
    from src.model_download import download_models
    download_models()

from src.asr import WhisperASR
from src.config import (
    ASR_MODEL_NAME,
    ASR_MULTILINGUAL_ENABLED,
    ASR_RECORD_SECONDS,
    ASR_TRANSLATE_TO_ENGLISH,
    BYPASS_PIN_ENV_KEY,
    DEFAULT_RECORD_SECONDS,
    WAKEWORD_CONFIG_PATH,
    WAKEWORD_DETECTION_THRESHOLD,
    WAKEWORD_MODEL_PATH,
)
from src.control_system import MediaCenterController, render_media_center
from src.fulfillment import CONTROL_INTENTS, fulfill
from src.state_machine import AssistantController, AssistantState
from src.tts import speak, tts_engines_available
from src.verifier import UserVerifier
from src.wakeword import WakeWordDetector, start_wakeword_listener

# ── Cached model loaders (survive Streamlit reruns within same process) ───────

@st.cache_resource(show_spinner="Loading speaker verification model…")
def _load_verifier():
    return UserVerifier()

@st.cache_resource(show_spinner="Loading wake word detector…")
def _load_wakeword_detector():
    if WAKEWORD_MODEL_PATH.exists() and WAKEWORD_CONFIG_PATH.exists():
        return WakeWordDetector(WAKEWORD_MODEL_PATH, WAKEWORD_CONFIG_PATH)
    return None

@st.cache_resource(show_spinner="Loading Whisper ASR model…")
def _load_asr(model_name: str):
    return WhisperASR(model_name=model_name)

@st.cache_resource(show_spinner="Loading NLU model…")
def _load_nlu(model_dir: str):
    return JointNLUPredictor(model_dir=model_dir)

@st.cache_resource(show_spinner="Checking TTS engines…")
def _load_tts_engines():
    return tts_engines_available()


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Atlas – Voice Assistant",
    page_icon="🎙️",
    layout="wide",
)

st.markdown(
    """
    <style>
    .status-badge {
        display:inline-block; padding:4px 12px; border-radius:12px;
        font-size:0.85rem; font-weight:600; margin:2px 0;
    }
    .badge-locked   { background:#fee2e2; color:#b91c1c; }
    .badge-unlocked { background:#fef3c7; color:#92400e; }
    .badge-ready    { background:#d1fae5; color:#065f46; }
    .badge-idle     { background:#e0e7ff; color:#3730a3; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ══════════════════════════════════════════════════════════════════════════════
# Weather helpers
# ══════════════════════════════════════════════════════════════════════════════

_WMO_EMOJI: dict[int, tuple[str, str]] = {
    0:  ("☀️",  "Clear sky"),
    1:  ("🌤️", "Mainly clear"),
    2:  ("⛅",  "Partly cloudy"),
    3:  ("☁️",  "Overcast"),
    45: ("🌫️", "Fog"),
    48: ("🌫️", "Fog"),
    51: ("🌦️", "Light drizzle"),
    53: ("🌦️", "Drizzle"),
    55: ("🌦️", "Heavy drizzle"),
    61: ("🌧️", "Light rain"),
    63: ("🌧️", "Rain"),
    65: ("🌧️", "Heavy rain"),
    71: ("🌨️", "Light snow"),
    73: ("🌨️", "Snow"),
    75: ("🌨️", "Heavy snow"),
    77: ("🌨️", "Snow grains"),
    80: ("🌦️", "Rain showers"),
    81: ("🌦️", "Rain showers"),
    82: ("🌦️", "Heavy showers"),
    85: ("🌨️", "Snow showers"),
    86: ("🌨️", "Heavy snow showers"),
    95: ("⛈️",  "Thunderstorm"),
    96: ("⛈️",  "Thunderstorm w/ hail"),
    99: ("⛈️",  "Severe thunderstorm"),
}

def _wmo_icon(code: int) -> tuple[str, str]:
    return _WMO_EMOJI.get(code, _WMO_EMOJI.get((code // 10) * 10, ("🌡️", "Unknown")))


def _fetch_weather() -> dict:
    """Fetch current weather for the user's approximate location."""
    try:
        if _HF_MODE:
            # On HF Spaces the server IP is in Ashburn VA — use Ottawa as default
            city, region = "Ottawa", "Ontario"
            lat, lng = 45.4215, -75.6972
        else:
            loc = requests.get("https://ipinfo.io/json", timeout=4).json()
            city = loc.get("city", "Unknown")
            region = loc.get("region", "")
            lat_lng = loc.get("loc", "45.4215,-75.6972").split(",")
            lat, lng = float(lat_lng[0]), float(lat_lng[1])
        w = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={"latitude": lat, "longitude": lng,
                    "current_weather": True, "temperature_unit": "celsius"},
            timeout=8,
        ).json().get("current_weather", {})
        icon, desc = _wmo_icon(int(w.get("weathercode", 0)))
        return {
            "city": city, "region": region,
            "temp": w.get("temperature", "?"),
            "wind": w.get("windspeed", "?"),
            "icon": icon, "desc": desc,
            "ok": True,
        }
    except Exception:
        return {"city": "—", "region": "", "temp": "?", "wind": "?",
                "icon": "🌡️", "desc": "Unavailable", "ok": False}


_WORD_NUMS_APP = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
    "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "a": 1, "an": 1,
}


def _parse_duration_secs(text: str) -> int:
    """Parse a human duration string like '10 minutes' or 'five minutes' to seconds."""
    text = text.lower()
    # Replace word numbers with digits
    _pat = r"\b(" + "|".join(re.escape(w) for w in _WORD_NUMS_APP) + r")\b"
    text = re.sub(_pat, lambda m: str(_WORD_NUMS_APP[m.group(0)]), text)
    m = re.search(r"(\d+)\s*(hours?|hrs?|minutes?|mins?|seconds?|secs?|h|m|s)(?:\b|$)", text)
    if not m:
        return 0
    val = int(m.group(1))
    unit = m.group(2)
    if unit.startswith("h"):
        return val * 3600
    if unit.startswith("min") or unit == "m":
        return val * 60
    return val


# ══════════════════════════════════════════════════════════════════════════════
# Session-state initialisation
# ══════════════════════════════════════════════════════════════════════════════

def _init_state() -> None:
    defaults = {
        "messages": ["System initialized. Status: Locked."],
        "recorded_wav": None,
        "wakeword_wav": None,
        "asr_recorded_wav": None,
        "asr_text": "",
        "asr_language": "",
        "asr_translated_text": "",
        "asr_edit_text": "",
        "nlu_result": None,
        "nlu_source_text": "",
        "awaiting_nlu_confirmation": False,
        "nlu_manual_slots_text": "{}",
        "nlu_manual_intent": INTENTS[0],
        # Post-NLU
        "confirmed_intent": None,
        "confirmed_slots": {},
        "confirmed_question": "",
        "fulfillment_result": None,
        "generated_answer": "",
        "tts_audio": None,
        # Answer generation
        "use_llm": False,
        "pipeline_view_step": "wakeword",
        "pipeline_manual_view": False,
        # Clock / timer
        "timer_end_epoch": 0,
        "timer_total_secs": 0,
        "timer_label": "",
        # Weather
        "weather": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # Weather (fetch once per session)
    if st.session_state.weather is None:
        st.session_state.weather = _fetch_weather()

    # Heavyweight objects (load once)
    if "controller" not in st.session_state:
        st.session_state.controller = AssistantController()

    if (
        "media_controller" not in st.session_state
        or not isinstance(st.session_state.media_controller, MediaCenterController)
    ):
        st.session_state.media_controller = MediaCenterController()

    if "verifier_ready" not in st.session_state:
        try:
            st.session_state.verifier = _load_verifier()
            st.session_state.verifier_ready = True
            st.session_state.verifier_error = ""
        except FileNotFoundError as exc:
            st.session_state.verifier = None
            st.session_state.verifier_ready = False
            st.session_state.verifier_error = str(exc)
        except Exception as exc:
            st.session_state.verifier = None
            st.session_state.verifier_ready = False
            st.session_state.verifier_error = str(exc)

    if "wakeword_detector" not in st.session_state:
        try:
            det = _load_wakeword_detector()
            st.session_state.wakeword_detector = det
            st.session_state.wakeword_ready = det is not None
        except Exception:
            st.session_state.wakeword_detector = None
            st.session_state.wakeword_ready = False

    if "asr_engine" not in st.session_state:
        try:
            st.session_state.asr_engine = _load_asr(ASR_MODEL_NAME)
            st.session_state.asr_ready = True
            st.session_state.asr_error = ""
        except Exception as exc:
            st.session_state.asr_engine = None
            st.session_state.asr_ready = False
            st.session_state.asr_error = str(exc)

    if "nlu_predictor" not in st.session_state:
        nlu_dir = Path("models/joint_nlu")
        if (nlu_dir / "model_state.pt").exists() and (nlu_dir / "label_mappings.json").exists():
            try:
                st.session_state.nlu_predictor = _load_nlu(str(nlu_dir))
                st.session_state.nlu_ready = True
                st.session_state.nlu_error = ""
            except Exception as exc:
                st.session_state.nlu_predictor = None
                st.session_state.nlu_ready = False
                st.session_state.nlu_error = str(exc)
        else:
            st.session_state.nlu_predictor = None
            st.session_state.nlu_ready = False
            st.session_state.nlu_error = (
                "NLU model not trained. Run:\n"
                "  python scripts/generate_nlu_dataset.py\n"
                "  python -m nlu.train --data_path data/nlu/train.full.json "
                "--output_dir models/joint_nlu"
            )

    if "tts_engines" not in st.session_state:
        st.session_state.tts_engines = _load_tts_engines()


_init_state()

controller: AssistantController         = st.session_state.controller
media_ctrl: MediaCenterController       = st.session_state.media_controller
messages:   list[str]                   = st.session_state.messages


def _msg(text: str) -> None:
    messages.append(text)


PIPELINE_STEPS = [
    ("wakeword", "1", "Wake Word"),
    ("asr", "2", "ASR"),
    ("intent", "3", "Intent"),
    ("fulfillment", "4", "Fulfillment"),
    ("tts", "5", "TTS"),
]

PIPELINE_STEP_TITLES = {
    "wakeword": "1 · Wake Word Detection",
    "asr": "2 · Automatic Speech Recognition",
    "intent": "3 · Intent Detection",
    "fulfillment": "4 · Fulfillment & Answer Generation",
    "tts": "5 · Text-to-Speech",
}

PIPELINE_STEP_IDS = [step_id for step_id, _, _ in PIPELINE_STEPS]


def _get_active_pipeline_step(controller: AssistantController) -> str:
    if st.session_state.awaiting_nlu_confirmation:
        return "intent"
    if (
        controller.state == AssistantState.WAKE_WORD_DETECTED
        or st.session_state.asr_text
        or st.session_state.asr_recorded_wav is not None
    ):
        return "asr"
    if st.session_state.generated_answer:
        return "tts"
    if st.session_state.confirmed_intent is not None:
        return "fulfillment"
    return "wakeword"


def _pipeline_step_index(step_id: str) -> int:
    return PIPELINE_STEP_IDS.index(step_id)


def _pipeline_step_id_at(index: int) -> str:
    return PIPELINE_STEP_IDS[max(0, min(index, len(PIPELINE_STEP_IDS) - 1))]


def _get_unlocked_pipeline_step_index(controller: AssistantController) -> int:
    unlocked_index = 0

    if (
        controller.state == AssistantState.WAKE_WORD_DETECTED
        or st.session_state.asr_text
        or st.session_state.asr_recorded_wav is not None
        or st.session_state.awaiting_nlu_confirmation
        or st.session_state.confirmed_intent is not None
        or st.session_state.generated_answer
    ):
        unlocked_index = 1

    if (
        st.session_state.awaiting_nlu_confirmation
        or st.session_state.confirmed_intent is not None
        or st.session_state.generated_answer
    ):
        unlocked_index = 2

    if st.session_state.confirmed_intent is not None:
        unlocked_index = 3

    if st.session_state.generated_answer:
        unlocked_index = 4

    return unlocked_index


def _get_completed_pipeline_step_index(controller: AssistantController) -> int:
    completed_index = _get_unlocked_pipeline_step_index(controller) - 1
    if st.session_state.tts_audio:
        completed_index = len(PIPELINE_STEPS) - 1
    return max(-1, min(completed_index, len(PIPELINE_STEPS) - 1))


def _follow_live_pipeline() -> None:
    st.session_state.pipeline_manual_view = False
    st.session_state.pipeline_view_step = "wakeword"


def _sync_pipeline_view(controller: AssistantController) -> None:
    active_step = _get_active_pipeline_step(controller)
    active_index = _pipeline_step_index(active_step)
    unlocked_index = _get_unlocked_pipeline_step_index(controller)
    completed_index = _get_completed_pipeline_step_index(controller)

    current_view = st.session_state.get("pipeline_view_step", active_step)
    if current_view not in PIPELINE_STEP_IDS:
        current_view = active_step

    if st.session_state.get("pipeline_manual_view", False):
        current_view = _pipeline_step_id_at(
            min(_pipeline_step_index(current_view), unlocked_index)
        )
    else:
        current_view = active_step

    st.session_state.pipeline_view_step = current_view
    st.session_state.pipeline_active_step = active_step
    st.session_state.pipeline_active_step_index = active_index
    st.session_state.pipeline_unlocked_step_index = unlocked_index
    st.session_state.pipeline_completed_step_index = completed_index


def _reset_pipeline(controller: AssistantController) -> None:
    controller.return_to_locked()
    controller.last_detected_wakeword = None

    reset_values = {
        "recorded_wav": None,
        "wakeword_wav": None,
        "asr_recorded_wav": None,
        "asr_text": "",
        "asr_language": "",
        "asr_translated_text": "",
        "asr_edit_text": "",
        "nlu_result": None,
        "nlu_source_text": "",
        "awaiting_nlu_confirmation": False,
        "nlu_manual_slots_text": "{}",
        "nlu_manual_intent": INTENTS[0],
        "confirmed_intent": None,
        "confirmed_slots": {},
        "confirmed_question": "",
        "fulfillment_result": None,
        "generated_answer": "",
        "tts_audio": None,
        "pipeline_view_step": "wakeword",
        "pipeline_manual_view": False,
    }
    for key, value in reset_values.items():
        st.session_state[key] = value

    # Widget-bound keys cannot be assigned directly — delete them so the
    # widgets revert to their default values on the next render.
    for widget_key in ("ww_bypass_text", "asr_type_bypass", "top_pin_bypass"):
        st.session_state.pop(widget_key, None)

    _msg("Pipeline reset. System locked.")


def _render_pipeline_stepper(view_step: str, completed_index: int) -> None:
    view_index = _pipeline_step_index(view_step)
    st.markdown(
        dedent(
            """
            <style>
            .pipeline-step {
                display: flex;
                align-items: center;
                gap: 0.45rem;
                min-height: 4rem;
                padding: 0.55rem 0.6rem;
                border-radius: 0.85rem;
                border: 1px solid #E5E7EB;
                background: #F9FAFB;
                margin-bottom: 0.4rem;
            }
            .pipeline-step-number {
                min-width: 1.6rem;
                min-height: 1.6rem;
                display: flex;
                align-items: center;
                justify-content: center;
                border-radius: 999px;
                font-size: 0.76rem;
                font-weight: 700;
                border: 1px solid #D1D5DB;
                background: #FFFFFF;
                color: #6B7280;
                flex-shrink: 0;
            }
            .pipeline-step-label {
                color: #6B7280;
                font-size: 0.78rem;
                font-weight: 600;
                line-height: 1.2;
                word-break: break-word;
            }
            .pipeline-step-current {
                border-color: #9CA3AF;
                background: #F3F4F6;
                box-shadow: inset 0 0 0 1px rgba(156, 163, 175, 0.28);
            }
            .pipeline-step-current .pipeline-step-number {
                border-color: #6B7280;
                background: #E5E7EB;
                color: #111827;
            }
            .pipeline-step-current .pipeline-step-label {
                color: #111827;
            }
            .pipeline-step-complete {
                background: #FFFFFF;
            }
            .pipeline-step-complete .pipeline-step-number {
                border-color: #9CA3AF;
                background: #F3F4F6;
                color: #374151;
            }
            .pipeline-step-complete .pipeline-step-label {
                color: #374151;
            }
            </style>
            """
        ).strip(),
        unsafe_allow_html=True,
    )

    step_columns = st.columns([1, 1, 1, 1.2, 0.9], gap="small")
    for column, (idx, (_, number, label)) in zip(step_columns, enumerate(PIPELINE_STEPS)):
        if idx == view_index:
            step_class = "pipeline-step pipeline-step-current"
        elif idx <= completed_index:
            step_class = "pipeline-step pipeline-step-complete"
        else:
            step_class = "pipeline-step pipeline-step-upcoming"
        with column:
            st.markdown(
                (
                    f'<div class="{step_class}">'
                    f'<div class="pipeline-step-number">{number}</div>'
                    f'<div class="pipeline-step-label">{label}</div>'
                    f"</div>"
                ),
                unsafe_allow_html=True,
            )


def _render_pipeline_navigation(controller: AssistantController) -> None:
    _sync_pipeline_view(controller)
    current_step = st.session_state.pipeline_view_step
    current_index = _pipeline_step_index(current_step)
    unlocked_index = st.session_state.pipeline_unlocked_step_index

    nav_cols = st.columns([1, 1, 1.2], gap="small")
    with nav_cols[0]:
        back_clicked = st.button(
            "← Back",
            use_container_width=True,
            disabled=current_index <= 0,
            key="pipeline_nav_back",
        )
    with nav_cols[1]:
        next_clicked = st.button(
            "Next →",
            use_container_width=True,
            disabled=current_index >= unlocked_index,
            key="pipeline_nav_next",
        )
    with nav_cols[2]:
        reset_clicked = st.button(
            "Reset Pipeline",
            use_container_width=True,
            key="pipeline_nav_reset",
        )

    if back_clicked:
        st.session_state.pipeline_manual_view = True
        st.session_state.pipeline_view_step = _pipeline_step_id_at(current_index - 1)
        st.rerun()

    if next_clicked:
        st.session_state.pipeline_manual_view = True
        st.session_state.pipeline_view_step = _pipeline_step_id_at(current_index + 1)
        st.rerun()

    if reset_clicked:
        _reset_pipeline(controller)
        st.rerun()

    st.caption("Use Back/Next to review unlocked steps. Next unlocks only after the current step is completed.")


def _render_wakeword_step(controller: AssistantController) -> None:
    ww_enabled = controller.state in {
        AssistantState.UNLOCKED,
        AssistantState.LISTENING,
        AssistantState.WAKE_WORD_DETECTED,
    }

    st.markdown(f"**{PIPELINE_STEP_TITLES['wakeword']}**")
    if not ww_enabled:
        st.caption("Unlock the assistant in System Control before running wake word detection.")

    if st.session_state.wakeword_ready:
        st.caption("Say 'Hey Atlas' or type it to bypass.")

        if _HF_MODE:
            ww_audio = st.audio_input("Record wake word", key="rec_ww_hf", disabled=not ww_enabled)
            det_ww = st.button("Detect", key="det_ww", use_container_width=True, disabled=not ww_enabled)

            if ww_audio is not None:
                try:
                    st.session_state.wakeword_wav = audio_input_to_numpy(ww_audio, preprocess=True)
                    _msg("Audio captured. Click Detect.")
                except Exception as exc:
                    _msg(f"Audio error: {exc}")

            if det_ww and ww_enabled:
                wav = st.session_state.wakeword_wav
                if wav is None:
                    _msg("No recording. Record first.")
                else:
                    try:
                        result = st.session_state.wakeword_detector.detect_waveform(
                            wav,
                            threshold=WAKEWORD_DETECTION_THRESHOLD,
                        )
                        if result.detected:
                            _msg(f"✓ Wake word detected (conf={result.confidence:.3f})")
                            controller.on_wakeword_detected("Hey Atlas")
                            _follow_live_pipeline()
                            st.rerun()
                        else:
                            _msg(f"✗ Not detected (conf={result.confidence:.3f})")
                        st.session_state.wakeword_wav = None
                    except Exception as exc:
                        _msg(f"Detection error: {exc}")
        else:
            ww_c1, ww_c2 = st.columns(2, gap="small")
            with ww_c1:
                rec_ww = st.button(
                    "Record",
                    key="rec_ww",
                    use_container_width=True,
                    disabled=not ww_enabled,
                )
            with ww_c2:
                det_ww = st.button(
                    "Detect",
                    key="det_ww",
                    use_container_width=True,
                    disabled=not ww_enabled,
                )

            if rec_ww and ww_enabled:
                try:
                    _msg("Recording wake word (3 s)…")
                    st.session_state.wakeword_wav = capture_microphone(DEFAULT_RECORD_SECONDS)
                    _msg("Done. Click Detect.")
                except Exception as exc:
                    _msg(f"Record error: {exc}")

            if det_ww and ww_enabled:
                wav = st.session_state.wakeword_wav
                if wav is None:
                    _msg("No recording. Click Record first.")
                else:
                    try:
                        result = st.session_state.wakeword_detector.detect_waveform(
                            wav,
                            threshold=WAKEWORD_DETECTION_THRESHOLD,
                        )
                        if result.detected:
                            _msg(f"✓ Wake word detected (conf={result.confidence:.3f})")
                            controller.on_wakeword_detected("Hey Atlas")
                            _follow_live_pipeline()
                            st.rerun()
                        else:
                            _msg(f"✗ Not detected (conf={result.confidence:.3f})")
                        st.session_state.wakeword_wav = None
                    except Exception as exc:
                        _msg(f"Detection error: {exc}")

        ww_bypass = st.text_input(
            "Type wake phrase to bypass",
            key="ww_bypass_text",
            placeholder="hey atlas",
            disabled=not ww_enabled,
        )
        if ww_bypass and ww_bypass.strip().lower() in {"hey atlas", "atlas"}:
            controller.on_wakeword_detected("Hey Atlas")
            _msg("Wake word bypassed via text.")
            _follow_live_pipeline()
            st.rerun()
    else:
        st.caption("Model not trained. Run `train_wakeword.py`.")


def _render_asr_step(controller: AssistantController) -> None:
    asr_enabled = controller.state == AssistantState.WAKE_WORD_DETECTED

    st.markdown(f"**{PIPELINE_STEP_TITLES['asr']}**")
    if not asr_enabled:
        st.caption("Wake word must be detected before ASR becomes active.")

    if not st.session_state.asr_ready:
        st.caption(f"ASR unavailable: {st.session_state.asr_error}")
        return

    st.caption("Record your command, then transcribe.")
    if not _HF_MODE:
        asr_c1, asr_c2 = st.columns(2, gap="small")
        with asr_c1:
            rec_asr = st.button(
                "Record",
                key="rec_asr",
                use_container_width=True,
                disabled=not asr_enabled,
            )
        with asr_c2:
            trans_asr = st.button(
                "Transcribe",
                key="trans_asr",
                use_container_width=True,
                disabled=not asr_enabled,
            )
    else:
        rec_asr = False
        asr_audio_hf = st.audio_input("Record command", key="rec_asr_hf", disabled=not asr_enabled)
        trans_asr = st.button(
            "Transcribe",
            key="trans_asr",
            use_container_width=True,
            disabled=not asr_enabled,
        )
        if asr_audio_hf is not None:
            try:
                st.session_state.asr_recorded_wav = audio_input_to_numpy(asr_audio_hf, preprocess=False)
                _msg("Audio captured. Click Transcribe.")
            except Exception as exc:
                _msg(f"Audio error: {exc}")

    st.markdown("**Type command (bypass recording):**")
    bypass_cols = st.columns([3, 1], gap="small")
    with bypass_cols[0]:
        asr_type = st.text_input(
            "bypass_text",
            key="asr_type_bypass",
            placeholder="e.g. play blinding lights by the weeknd",
            label_visibility="collapsed",
            disabled=not asr_enabled,
        )
    with bypass_cols[1]:
        submit_type = st.button(
            "Submit",
            key="asr_type_submit",
            use_container_width=True,
            disabled=not asr_enabled,
        )

    if submit_type and asr_type.strip() and asr_enabled:
        text = asr_type.strip()
        st.session_state.asr_text = text
        st.session_state.asr_edit_text = text
        st.session_state.nlu_source_text = text
        st.session_state.awaiting_nlu_confirmation = True
        if st.session_state.nlu_ready:
            st.session_state.nlu_result = st.session_state.nlu_predictor.predict(text)
            _msg(f"Text input: '{text}' → {st.session_state.nlu_result['intent']}")
        else:
            st.session_state.nlu_result = None
            _msg(f"Text input: '{text}' (NLU unavailable, use override)")
        _follow_live_pipeline()
        st.rerun()

    if not _HF_MODE:
        st.markdown("**— or record audio —**")

    if rec_asr and asr_enabled:
        try:
            _msg(f"Recording command ({ASR_RECORD_SECONDS} s)…")
            st.session_state.asr_recorded_wav = capture_microphone(
                ASR_RECORD_SECONDS,
                preprocess=False,
            )
            _msg("Done. Click Transcribe.")
        except Exception as exc:
            _msg(f"ASR record error: {exc}")

    if trans_asr and asr_enabled:
        if st.session_state.asr_recorded_wav is None:
            _msg("No recording. Record first.")
        else:
            try:
                res = st.session_state.asr_engine.transcribe_waveform(
                    st.session_state.asr_recorded_wav,
                    multilingual=ASR_MULTILINGUAL_ENABLED,
                    include_english_translation=ASR_TRANSLATE_TO_ENGLISH,
                )
                st.session_state.asr_text = res.text
                st.session_state.asr_language = res.language
                st.session_state.asr_translated_text = res.translated_text or ""
                st.session_state.asr_edit_text = res.translated_text or res.text
                _msg("Transcription ready. Review and confirm.")
            except Exception as exc:
                _msg(f"ASR error: {exc}")

    if st.session_state.asr_text:
        st.text_area(
            "Original transcript",
            value=st.session_state.asr_text,
            height=70,
            disabled=True,
            key="asr_orig_disp",
        )
        if ASR_MULTILINGUAL_ENABLED:
            st.caption(f"Language: {st.session_state.asr_language or 'unknown'}")
        if ASR_MULTILINGUAL_ENABLED and st.session_state.asr_translated_text:
            st.text_area(
                "English translation",
                value=st.session_state.asr_translated_text,
                height=68,
                disabled=True,
                key="asr_trans_disp",
            )

        edited = st.text_area(
            "Editable transcript (correct if needed)",
            value=st.session_state.asr_edit_text,
            height=90,
            key="asr_edit_in",
        )
        st.session_state.asr_edit_text = edited

        conf_c, clr_c = st.columns(2, gap="small")
        with conf_c:
            confirm_asr = st.button(
                "Confirm Transcript",
                key="confirm_asr",
                use_container_width=True,
            )
        with clr_c:
            clear_asr = st.button("Clear", key="clear_asr", use_container_width=True)

        if confirm_asr:
            final = st.session_state.asr_edit_text.strip()
            if not final:
                _msg("Empty transcript. Edit or re-record.")
            else:
                _msg(f"Transcript confirmed: {final}")
                st.session_state.nlu_source_text = final
                st.session_state.awaiting_nlu_confirmation = True
                if st.session_state.nlu_ready:
                    st.session_state.nlu_result = (
                        st.session_state.nlu_predictor.predict(final)
                    )
                    _msg(
                        f"Intent predicted: "
                        f"{st.session_state.nlu_result['intent']}"
                    )
                else:
                    st.session_state.nlu_result = None
                    _msg("NLU unavailable – use manual override.")
                _follow_live_pipeline()
                st.rerun()

        if clear_asr:
            for key in (
                "asr_recorded_wav",
                "asr_text",
                "asr_language",
                "asr_translated_text",
                "asr_edit_text",
                "nlu_result",
                "nlu_source_text",
            ):
                st.session_state[key] = "" if key.endswith("text") else None
            st.session_state.asr_edit_text = ""
            st.session_state.awaiting_nlu_confirmation = False
            _msg("ASR cleared.")
            _follow_live_pipeline()
            st.rerun()


def _render_intent_step(controller: AssistantController, media_ctrl: MediaCenterController) -> None:
    st.markdown(f"**{PIPELINE_STEP_TITLES['intent']}**")
    if not st.session_state.awaiting_nlu_confirmation:
        st.caption("Waiting for ASR confirmation.")
        return

    st.caption("Review the predicted intent and slots, then confirm.")
    st.text_area(
        "Confirmed transcript",
        value=st.session_state.nlu_source_text,
        height=68,
        disabled=True,
        key="nlu_src_disp",
    )

    if st.session_state.nlu_ready and st.session_state.nlu_result:
        st.text_input(
            "Predicted intent",
            value=st.session_state.nlu_result.get("intent", "OOS"),
            disabled=True,
            key="nlu_pred_int",
        )
        conf = st.session_state.nlu_result.get("intent_confidence")
        if conf is not None:
            st.caption(f"Confidence: {conf:.2%}")
        st.text_area(
            "Predicted slots (JSON)",
            value=json.dumps(
                st.session_state.nlu_result.get("slots", {}),
                indent=2,
            ),
            height=90,
            disabled=True,
            key="nlu_pred_sl",
        )
    else:
        st.caption("Automatic NLU unavailable. Use manual override below.")
        if st.session_state.nlu_error:
            with st.expander("NLU error details"):
                st.code(st.session_state.nlu_error)

    st.markdown("**Manual Override (bypass)**")
    sel_intent = st.selectbox(
        "Intent",
        options=INTENTS,
        index=INTENTS.index(st.session_state.nlu_manual_intent)
        if st.session_state.nlu_manual_intent in INTENTS
        else 0,
        key="nlu_man_int_sel",
    )
    st.session_state.nlu_manual_intent = sel_intent
    man_slots_txt = st.text_area(
        "Slots JSON",
        value=st.session_state.nlu_manual_slots_text,
        height=80,
        key="nlu_man_sl_in",
    )
    st.session_state.nlu_manual_slots_text = man_slots_txt

    ov_c, conf_c = st.columns(2, gap="small")
    with ov_c:
        apply_bypass = st.button(
            "Apply Override",
            key="apply_nlu_bypass",
            use_container_width=True,
        )
    with conf_c:
        confirm_nlu = st.button(
            "Confirm & Fulfill",
            key="confirm_nlu",
            use_container_width=True,
        )

    if apply_bypass:
        try:
            man_slots = json.loads(st.session_state.nlu_manual_slots_text or "{}")
            if not isinstance(man_slots, dict):
                raise ValueError("Must be a JSON object")
            man_slots = {str(k): str(v) for k, v in man_slots.items()}
            st.session_state.nlu_result = {
                "intent": sel_intent,
                "slots": man_slots,
                "bypass_used": True,
            }
            _msg("Manual override applied.")
            st.rerun()
        except Exception as exc:
            _msg(f"Invalid slots JSON: {exc}")

    if confirm_nlu:
        if st.session_state.nlu_result is None:
            _msg("No NLU result. Run prediction or apply override.")
        else:
            intent = st.session_state.nlu_result.get("intent", "OOS")
            slots = st.session_state.nlu_result.get("slots", {})
            question = st.session_state.get("nlu_source_text", "")

            # "Lock the system" and variants → dedicated LockSystem intent.
            # Locking is only done through the System Control panel.
            if re.search(r"\block\b", question, re.IGNORECASE):
                intent = "LockSystem"
                slots = {}

            _msg(f"Intent confirmed: {intent}  slots={json.dumps(slots)}")
            st.session_state.confirmed_intent = intent
            st.session_state.confirmed_slots = slots
            st.session_state.confirmed_question = question

            with st.spinner("Calling APIs…"):
                try:
                    fr = fulfill(intent, slots, media_controller=media_ctrl)
                except Exception as exc:
                    fr = {"error": str(exc)}
            st.session_state.fulfillment_result = fr

            if intent == "SetTimer":
                dur_str = slots.get("duration", "")
                secs = _parse_duration_secs(dur_str)
                if secs > 0:
                    st.session_state.timer_end_epoch = _time.time() + secs
                    st.session_state.timer_total_secs = secs
                    src_text = st.session_state.get("nlu_source_text", "")
                    label_match = re.search(
                        r"(?:set\s+(?:a\s+)?|start\s+(?:a\s+)?|create\s+(?:a\s+)?)"
                        r"(\w+)\s+timer",
                        src_text,
                        re.IGNORECASE,
                    )
                    if label_match and label_match.group(1).lower() not in {
                        "a", "an", "the", "me", "my", "new",
                    }:
                        st.session_state.timer_label = label_match.group(1).capitalize() + " Timer"
                    else:
                        mins, secs_remainder = divmod(secs, 60)
                        st.session_state.timer_label = (
                            f"{mins}m Timer" if mins else f"{secs_remainder}s Timer"
                        )
                    _msg(f"Timer started: {st.session_state.timer_label} ({secs}s)")

            ans = generate_answer(
                intent, fr, slots,
                use_llm=st.session_state.use_llm,
                question=question,
            )
            st.session_state.generated_answer = ans
            st.session_state.tts_audio = None
            _msg(f"Answer: {ans}")

            if intent == "LockSystem":
                _reset_pipeline(controller)
                st.rerun()
                return

            controller.on_asr_confirmed()
            for key in (
                "asr_recorded_wav",
                "asr_text",
                "asr_language",
                "asr_translated_text",
                "asr_edit_text",
            ):
                st.session_state[key] = "" if key.endswith("text") else None
            st.session_state.asr_edit_text = ""
            st.session_state.nlu_result = None
            st.session_state.nlu_source_text = ""
            st.session_state.awaiting_nlu_confirmation = False
            _follow_live_pipeline()
            st.rerun()


def _render_fulfillment_step() -> None:
    st.markdown(f"**{PIPELINE_STEP_TITLES['fulfillment']}**")
    if st.session_state.confirmed_intent is None:
        st.caption("Waiting for intent confirmation.")
        return

    intent = st.session_state.confirmed_intent
    slots = st.session_state.confirmed_slots
    fr = st.session_state.fulfillment_result or {}

    st.text_input("Intent", value=intent, disabled=True, key="res_intent")
    st.text_area(
        "Slots",
        value=json.dumps(slots, indent=2),
        height=70,
        disabled=True,
        key="res_slots",
    )

    question = st.session_state.get("confirmed_question", "")
    with st.expander("Raw fulfillment data", expanded=bool(fr)):
        if question:
            st.caption(f"User question: {question}")
        if fr:
            st.json(fr)
        else:
            st.caption("No API response.")

    ans = st.session_state.generated_answer
    if ans:
        st.markdown("**Answer**")
        st.info(ans)

    regen_col, llm_col = st.columns([1, 1], gap="small")
    with regen_col:
        if st.button("Regenerate", key="regen_ans", use_container_width=True):
            new_ans = generate_answer(
                intent, fr, slots,
                use_llm=st.session_state.use_llm,
                question=st.session_state.get("confirmed_question", ""),
            )
            st.session_state.generated_answer = new_ans
            st.session_state.tts_audio = None
            st.rerun()
    with llm_col:
        import os as _os_fulfillment
        _has_key = bool(_os_fulfillment.environ.get("GROQ_API_KEY", "").strip())
        _llm_toggle = st.toggle(
            "Llama 3.3 (Groq)",
            value=st.session_state.use_llm,
            help=(
                "Use Llama 3.3 70B via Groq for natural spoken responses. "
                "Control commands always use templates."
                if _has_key else
                "Set GROQ_API_KEY in .env to enable. Falls back to templates."
            ),
        )
        if _llm_toggle != st.session_state.use_llm:
            st.session_state.use_llm = _llm_toggle
            st.rerun()


def _render_tts_step() -> None:
    st.markdown(f"**{PIPELINE_STEP_TITLES['tts']}**")
    ans = st.session_state.generated_answer
    if not ans:
        st.caption("No answer yet. Complete fulfillment first.")
        return

    if st.session_state.confirmed_intent is not None:
        st.caption("Final answer ready. Speak it or review the latest fulfillment output below.")
        with st.expander("Latest fulfillment data"):
            fr = st.session_state.fulfillment_result or {}
            question = st.session_state.get("confirmed_question", "")
            if question:
                st.caption(f"User question: {question}")
            if fr:
                st.json(fr)
            else:
                st.caption("No API response.")

    st.info(ans)
    if st.button("Regenerate", key="regen_ans_tts", use_container_width=True):
        intent = st.session_state.confirmed_intent
        slots = st.session_state.confirmed_slots
        fr = st.session_state.fulfillment_result or {}
        st.session_state.generated_answer = generate_answer(
            intent, fr, slots,
            use_llm=st.session_state.use_llm,
            question=st.session_state.get("confirmed_question", ""),
        )
        st.session_state.tts_audio = None
        st.rerun()

    speak_btn = st.button("🔊 Speak", key="speak_btn", use_container_width=True)
    if speak_btn:
        with st.spinner("Synthesising speech…"):
            try:
                audio = speak(ans)
                st.session_state.tts_audio = audio
                _msg("TTS audio generated.")
            except Exception as exc:
                _msg(f"TTS error: {exc}")

    if st.session_state.tts_audio:
        st.audio(st.session_state.tts_audio, format="audio/mpeg")


def _render_user_verification_panel(controller: AssistantController) -> None:
    with st.container(border=True):
        st.markdown("**User Verification**")
        if not st.session_state.verifier_ready:
            err = st.session_state.get("verifier_error", "")
            if err:
                st.warning(
                    f"Voice verifier unavailable: {err}\n\n"
                    "Run `git lfs pull`, or regenerate voiceprints with "
                    "`python scripts/enroll.py` and `python scripts/tune_threshold.py`."
                )
            else:
                st.warning("Voiceprint store missing. Run `enroll.py` and `tune_threshold.py`.")

        if _HF_MODE:
            voice_audio_hf = st.audio_input("Record voice for verification", key="rec_voice_hf")
            verify_clicked = st.button("Verify User", use_container_width=True)
            record_clicked = False
            if voice_audio_hf is not None:
                try:
                    st.session_state.recorded_wav = audio_input_to_numpy(voice_audio_hf, preprocess=True)
                    _msg("Audio captured. Click Verify User.")
                except Exception as exc:
                    st.session_state.recorded_wav = None
                    _msg(f"Audio error: {exc}")
        else:
            rec_col, ver_col = st.columns(2, gap="small")
            with rec_col:
                record_clicked = st.button("Record Voice", use_container_width=True)
            with ver_col:
                verify_clicked = st.button("Verify User", use_container_width=True)

            if record_clicked:
                try:
                    _msg("Recording voice (3 s)…")
                    st.session_state.recorded_wav = capture_microphone(DEFAULT_RECORD_SECONDS)
                    _msg("Recording done. Click Verify User.")
                except Exception as exc:
                    st.session_state.recorded_wav = None
                    _msg(f"Recording error: {exc}")

        if verify_clicked and st.session_state.verifier_ready:
            if st.session_state.recorded_wav is None:
                _msg("No recording. Click Record Voice first.")
            else:
                try:
                    result = st.session_state.verifier.verify_waveform(
                        st.session_state.recorded_wav
                    )
                    if result.verified:
                        _msg(
                            f"✓ Verified: {result.matched_user} "
                            f"(score={result.score:.3f}, threshold={result.threshold:.3f})"
                        )
                        controller.on_verified()
                        _msg("System Unlocked. Listening for wake word…")
                        start_wakeword_listener()
                        _follow_live_pipeline()
                        st.rerun()
                    else:
                        controller.on_failed_verification()
                        _msg(
                            f"✗ Not verified (score={result.score:.3f} "
                            f"< threshold={result.threshold:.3f}). Try again."
                        )
                    st.session_state.recorded_wav = None
                except Exception as exc:
                    controller.on_failed_verification()
                    _msg(f"Verification error: {exc}")

        st.divider()
        pin_col, btn_col = st.columns([1.2, 1], gap="small")
        with pin_col:
            pin = st.text_input(
                "PIN bypass",
                type="password",
                label_visibility="collapsed",
                placeholder="PIN bypass",
                key="top_pin_bypass",
            )
        with btn_col:
            bypass_clicked = st.button(
                "Unlock with PIN",
                use_container_width=True,
                key="top_pin_unlock",
            )

        if bypass_clicked:
            expected = os.getenv(BYPASS_PIN_ENV_KEY, "1234")
            if pin == expected:
                bypass_result = controller.apply_bypass()
                _msg(f"PIN accepted. {bypass_result.message}")
                if bypass_result.should_start_wakeword_listener:
                    start_wakeword_listener()
                _follow_live_pipeline()
                st.rerun()
            else:
                controller.on_failed_verification()
                _msg("Invalid PIN.")


def _render_system_state_panel(controller: AssistantController) -> None:
    with st.container(border=True):
        state = controller.state
        state_map = {
            AssistantState.LOCKED:              ("🔒 LOCKED",          "status-badge badge-locked"),
            AssistantState.UNLOCKED:            ("🔓 UNLOCKED",        "status-badge badge-unlocked"),
            AssistantState.SLEEP:               ("💤 SLEEP",           "status-badge badge-idle"),
            AssistantState.LISTENING:           ("👂 LISTENING",       "status-badge badge-ready"),
            AssistantState.WAKE_WORD_DETECTED:  ("✅ READY",           "status-badge badge-ready"),
        }
        label, css = state_map.get(state, (state.value, "status-badge badge-idle"))
        st.markdown(f'<span class="{css}">{label}</span>', unsafe_allow_html=True)
        if controller.failed_attempts > 0:
            st.caption(f"Failed attempts: {controller.failed_attempts}")


def _render_activity_log_panel(messages: list[str], height: int = 220) -> None:
    with st.container(border=True):
        st.markdown("**Activity Log**")
        st.text_area(
            "log",
            value="\n".join(messages[-12:]),
            height=height,
            disabled=True,
            label_visibility="collapsed",
        )


# ══════════════════════════════════════════════════════════════════════════════
# Header
# ══════════════════════════════════════════════════════════════════════════════

st.title("🎙️ Atlas – Voice Assistant")
st.caption(
    "Pipeline: User Verification → Wake Word → ASR → Intent Detection → "
    "Fulfillment → Answer Generation → TTS"
)

# ── Top row: clock + timer + weather + verification + state/log ────────────
_clock_col, _timer_col, _weather_col, _verify_col, _log_col = st.columns(
    [1, 1, 1.55, 1.45, 1.4],
    gap="large",
)

# ── Analog clock ──────────────────────────────────────────────────────────────
with _clock_col:
    _clock_html = """
<canvas id="atlasClock" width="130" height="130"
        style="display:block; margin:0 auto;"></canvas>
<div id="atlasDigital" style="text-align:center; font-family:monospace;
     font-size:0.9rem; color:#475569; margin-top:4px;"></div>
<script>
(function(){
  var canvas = document.getElementById('atlasClock');
  if(!canvas) return;
  var ctx = canvas.getContext('2d');
  function drawClock(){
    var W=canvas.width, H=canvas.height;
    var cx=W/2, cy=H/2, r=cx-6;
    var now=new Date();
    ctx.clearRect(0,0,W,H);

    // Face
    ctx.beginPath(); ctx.arc(cx,cy,r,0,2*Math.PI);
    var grad=ctx.createRadialGradient(cx,cy-r*0.3,r*0.1,cx,cy,r);
    grad.addColorStop(0,'#1e3a5f'); grad.addColorStop(1,'#0f2340');
    ctx.fillStyle=grad; ctx.fill();
    ctx.strokeStyle='#90cdf4'; ctx.lineWidth=2; ctx.stroke();

    // Hour ticks
    for(var i=0;i<12;i++){
      var a=i/12*2*Math.PI;
      var big=(i%3===0);
      var x1=cx+(r-(big?12:6))*Math.cos(a), y1=cy+(r-(big?12:6))*Math.sin(a);
      var x2=cx+r*Math.cos(a), y2=cy+r*Math.sin(a);
      ctx.beginPath(); ctx.moveTo(x1,y1); ctx.lineTo(x2,y2);
      ctx.strokeStyle=big?'#fff':'#90cdf4';
      ctx.lineWidth=big?2:1; ctx.stroke();
    }

    // Hour numbers
    ctx.font='bold 10px sans-serif'; ctx.fillStyle='#cbd5e1';
    ctx.textAlign='center'; ctx.textBaseline='middle';
    [12,3,6,9].forEach(function(n){
      var a=(n/12*2*Math.PI)-Math.PI/2;
      ctx.fillText(n, cx+(r-20)*Math.cos(a), cy+(r-20)*Math.sin(a));
    });

    var h=now.getHours()%12, m=now.getMinutes(), s=now.getSeconds();

    // Hour hand
    var ha=(h+m/60)/12*2*Math.PI-Math.PI/2;
    ctx.beginPath(); ctx.moveTo(cx,cy);
    ctx.lineTo(cx+r*0.5*Math.cos(ha), cy+r*0.5*Math.sin(ha));
    ctx.strokeStyle='#fff'; ctx.lineWidth=4; ctx.lineCap='round'; ctx.stroke();

    // Minute hand
    var ma=(m+s/60)/60*2*Math.PI-Math.PI/2;
    ctx.beginPath(); ctx.moveTo(cx,cy);
    ctx.lineTo(cx+r*0.72*Math.cos(ma), cy+r*0.72*Math.sin(ma));
    ctx.strokeStyle='#90cdf4'; ctx.lineWidth=3; ctx.lineCap='round'; ctx.stroke();

    // Second hand
    var sa=s/60*2*Math.PI-Math.PI/2;
    ctx.beginPath(); ctx.moveTo(cx,cy);
    ctx.lineTo(cx+r*0.82*Math.cos(sa), cy+r*0.82*Math.sin(sa));
    ctx.strokeStyle='#f87171'; ctx.lineWidth=1.5; ctx.lineCap='round'; ctx.stroke();

    // Centre dot
    ctx.beginPath(); ctx.arc(cx,cy,4,0,2*Math.PI);
    ctx.fillStyle='#f87171'; ctx.fill();

    // Digital time below
    var dEl=document.getElementById('atlasDigital');
    if(dEl){
      var hh=String(now.getHours()).padStart(2,'0');
      var mm=String(m).padStart(2,'0');
      var ampm=now.getHours()<12?'AM':'PM';
      dEl.textContent=hh+':'+mm+' '+ampm;
    }
  }
  setInterval(drawClock,1000); drawClock();
})();
</script>
"""
    _components.html(_clock_html, height=155)

# ── Timer widget ───────────────────────────────────────────────────────────────
with _timer_col:
    _t_end   = float(st.session_state.timer_end_epoch or 0)
    _t_total = float(st.session_state.timer_total_secs or 0)
    _t_label = str(st.session_state.timer_label or "")
    _timer_html = f"""
<canvas id="atlasTimer" width="130" height="130"
        style="display:block; margin:0 auto;"></canvas>
<div id="atlasTimerStatus" style="text-align:center; font-family:sans-serif;
     font-size:0.78rem; color:#64748b; margin-top:4px;">No timer set</div>
<script>
(function(){{
  var canvas = document.getElementById('atlasTimer');
  if(!canvas) return;
  var ctx = canvas.getContext('2d');
  var TIMER_END = {_t_end};
  var TIMER_TOTAL = {_t_total};
  var LABEL = {json.dumps(_t_label)};

  function drawTimer(){{
    var W=canvas.width, H=canvas.height;
    var cx=W/2, cy=H/2, r=cx-12;
    var now=Date.now()/1000;
    ctx.clearRect(0,0,W,H);

    if(TIMER_END<=0){{
      // Idle face
      ctx.beginPath(); ctx.arc(cx,cy,r,0,2*Math.PI);
      ctx.strokeStyle='#e2e8f0'; ctx.lineWidth=8; ctx.stroke();
      ctx.font='28px serif'; ctx.textAlign='center'; ctx.textBaseline='middle';
      ctx.fillText('⏱',cx,cy);
      document.getElementById('atlasTimerStatus').textContent='No timer set';
      return;
    }}

    var remaining = TIMER_END - now;
    var progress = TIMER_TOTAL>0 ? Math.max(0, remaining/TIMER_TOTAL) : 0;

    // Background ring
    ctx.beginPath(); ctx.arc(cx,cy,r,0,2*Math.PI);
    ctx.strokeStyle='#e2e8f0'; ctx.lineWidth=10; ctx.stroke();

    if(remaining>0){{
      // Colour: green→yellow→red as it counts down
      var pct=progress;
      var colour = pct>0.5 ? '#22c55e' : pct>0.25 ? '#f59e0b' : '#ef4444';
      var start=-Math.PI/2;
      var end=start+2*Math.PI*progress;
      ctx.beginPath(); ctx.arc(cx,cy,r,start,end);
      ctx.strokeStyle=colour; ctx.lineWidth=10; ctx.lineCap='round'; ctx.stroke();

      var mins=Math.floor(remaining/60), secs=Math.floor(remaining%60);
      ctx.font='bold 18px monospace'; ctx.fillStyle='#1e3a5f';
      ctx.textAlign='center'; ctx.textBaseline='middle';
      ctx.fillText(String(mins).padStart(2,'0')+':'+String(secs).padStart(2,'0'), cx, LABEL?cy-8:cy);
      if(LABEL){{
        ctx.font='10px sans-serif'; ctx.fillStyle='#64748b';
        ctx.fillText(LABEL,cx,cy+10);
      }}
      document.getElementById('atlasTimerStatus').textContent='Running…';
    }} else {{
      // Done
      ctx.beginPath(); ctx.arc(cx,cy,r,0,2*Math.PI);
      ctx.strokeStyle='#ef4444'; ctx.lineWidth=10; ctx.stroke();
      ctx.font='22px serif'; ctx.textAlign='center'; ctx.textBaseline='middle';
      ctx.fillText('✅',cx,cy-6);
      if(LABEL){{
        ctx.font='10px sans-serif'; ctx.fillStyle='#ef4444';
        ctx.fillText(LABEL,cx,cy+14);
      }}
      document.getElementById('atlasTimerStatus').textContent='Done!';
    }}
  }}
  setInterval(drawTimer,1000); drawTimer();
}})();
</script>
"""
    _components.html(_timer_html, height=155)

# ── Weather ────────────────────────────────────────────────────────────────────
with _weather_col:
    _w = st.session_state.weather or {}
    _city    = _w.get("city", "—")
    _region  = _w.get("region", "")
    _temp    = _w.get("temp", "?")
    _wind    = _w.get("wind", "?")
    _icon    = _w.get("icon", "🌡️")
    _desc    = _w.get("desc", "—")
    _loc_str = f"{_city}, {_region}" if _region else _city
    st.markdown(
        f"""
<div style="border:1px solid #e2e8f0; border-radius:12px; padding:12px 18px;
     display:inline-flex; align-items:center; gap:16px; background:#f8fafc;">
  <div style="font-size:3rem; line-height:1;">{_icon}</div>
  <div>
    <div style="font-size:1.6rem; font-weight:700; color:#1e3a5f;">{_temp}°C</div>
    <div style="font-size:0.9rem; color:#475569;">{_desc}</div>
    <div style="font-size:0.78rem; color:#94a3b8;">📍 {_loc_str} · 💨 {_wind} km/h</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )
    if st.button("↻ Refresh weather", key="refresh_weather"):
        st.session_state.weather = _fetch_weather()
        st.rerun()

with _verify_col:
    _render_user_verification_panel(controller)

with _log_col:
    _render_system_state_panel(controller)
    _render_activity_log_panel(messages, height=120)

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# Layout: pipeline column  |  sidebar column
# ══════════════════════════════════════════════════════════════════════════════

pipeline_col, sidebar_col = st.columns([1.35, 1.3], gap="large")

# ─────────────────────────────────────────────────────────────────────────────
# RIGHT SIDEBAR  – Verification + system status + messages
# ─────────────────────────────────────────────────────────────────────────────

with sidebar_col:
    if controller.state != AssistantState.LOCKED:
        with st.container(border=True):
            st.subheader("Smart Media Center")
            render_media_center(media_ctrl)
    else:
        with st.container(border=True):
            st.markdown("**Smart Media Center**")
            st.caption("Unlock the assistant to access the media center.")

# ─────────────────────────────────────────────────────────────────────────────
# LEFT PIPELINE  – sequential steps
# ─────────────────────────────────────────────────────────────────────────────

with pipeline_col:
    st.subheader("Pipeline")
    with st.container(border=True):
        _sync_pipeline_view(controller)
        viewed_pipeline_step = st.session_state.pipeline_view_step
        active_pipeline_step = st.session_state.pipeline_active_step
        completed_pipeline_step_index = st.session_state.pipeline_completed_step_index

        _render_pipeline_stepper(viewed_pipeline_step, completed_pipeline_step_index)
        _render_pipeline_navigation(controller)
        st.caption(
            f"Viewing: {PIPELINE_STEP_TITLES[viewed_pipeline_step]} | "
            f"Live step: {PIPELINE_STEP_TITLES[active_pipeline_step]}"
        )
        st.divider()

        if viewed_pipeline_step == "wakeword":
            _render_wakeword_step(controller)
        elif viewed_pipeline_step == "asr":
            _render_asr_step(controller)
        elif viewed_pipeline_step == "intent":
            _render_intent_step(controller, media_ctrl)
        elif viewed_pipeline_step == "fulfillment":
            _render_fulfillment_step()
        else:
            _render_tts_step()
