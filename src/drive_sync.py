"""Google Drive sync for large private assets.

Uploads/downloads raw_recordings/, wakeword_dataset/, and voiceprints/
using a service account. The service account has access only to the
shared Atlas folder — no other Drive content is accessible.

Required env vars:
    GOOGLE_SERVICE_ACCOUNT_JSON  — path to credentials.json, or the JSON
                                   content itself as a single-line string
    DRIVE_RAW_RECORDINGS_FOLDER_ID
    DRIVE_WAKEWORD_DATASET_FOLDER_ID
    DRIVE_VOICEPRINTS_FOLDER_ID

Usage:
    from src.drive_sync import DriveSync
    ds = DriveSync()
    ds.upload_folder(local_path, folder_id)   # push local → Drive
    ds.download_folder(folder_id, local_path) # pull Drive → local
"""
from __future__ import annotations

import io
import json
import os
from pathlib import Path


def _build_service():
    """Build an authenticated Google Drive service from env credentials."""
    try:
        from google.oauth2 import service_account
        from googleapiclient.discovery import build
    except ImportError as exc:
        raise ImportError(
            "google-api-python-client is not installed. "
            "Run: pip install google-api-python-client google-auth"
        ) from exc

    raw = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON", "").strip()
    if not raw:
        raise EnvironmentError(
            "GOOGLE_SERVICE_ACCOUNT_JSON is not set. "
            "Set it to the path of your credentials.json or paste the JSON directly."
        )

    # Accept either a file path or inline JSON string
    if raw.startswith("{"):
        info = json.loads(raw)
    else:
        info = json.loads(Path(raw).read_text())

    creds = service_account.Credentials.from_service_account_info(
        info,
        scopes=["https://www.googleapis.com/auth/drive"],
    )
    return build("drive", "v3", credentials=creds, cache_discovery=False)


class DriveSync:
    def __init__(self) -> None:
        self._svc = _build_service()

    # ── upload ────────────────────────────────────────────────────────────────

    def upload_file(self, local_path: Path, parent_folder_id: str) -> str:
        """Upload a single file; returns the Drive file ID."""
        from googleapiclient.http import MediaFileUpload

        file_metadata = {"name": local_path.name, "parents": [parent_folder_id]}
        media = MediaFileUpload(str(local_path), resumable=True)

        # Check if file already exists in Drive (update instead of duplicate)
        existing_id = self._find_file(local_path.name, parent_folder_id)
        if existing_id:
            result = (
                self._svc.files()
                .update(fileId=existing_id, media_body=media)
                .execute()
            )
            return result["id"]

        result = (
            self._svc.files()
            .create(body=file_metadata, media_body=media, fields="id")
            .execute()
        )
        return result["id"]

    def upload_folder(self, local_dir: Path, drive_folder_id: str) -> None:
        """Recursively upload a local directory to a Drive folder."""
        local_dir = Path(local_dir)
        if not local_dir.exists():
            print(f"[DriveSync] Skipping upload — folder not found: {local_dir}")
            return

        # Build a map of existing Drive subfolders to avoid duplicates
        drive_subfolder_ids: dict[str, str] = {}

        for item in sorted(local_dir.rglob("*")):
            if item.is_file():
                rel = item.relative_to(local_dir)
                parts = rel.parts

                # Ensure parent folders exist in Drive
                current_parent = drive_folder_id
                for part in parts[:-1]:
                    key = f"{current_parent}/{part}"
                    if key not in drive_subfolder_ids:
                        drive_subfolder_ids[key] = self._ensure_folder(
                            part, current_parent
                        )
                    current_parent = drive_subfolder_ids[key]

                print(f"[DriveSync] Uploading {rel}…")
                self.upload_file(item, current_parent)

    # ── download ──────────────────────────────────────────────────────────────

    def download_file(self, file_id: str, dest_path: Path) -> None:
        """Download a single Drive file to dest_path."""
        from googleapiclient.http import MediaIoBaseDownload

        dest_path.parent.mkdir(parents=True, exist_ok=True)
        request = self._svc.files().get_media(fileId=file_id)
        buf = io.BytesIO()
        downloader = MediaIoBaseDownload(buf, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
        dest_path.write_bytes(buf.getvalue())

    def download_folder(self, drive_folder_id: str, local_dir: Path) -> None:
        """Recursively download a Drive folder to a local directory."""
        local_dir = Path(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)
        self._download_folder_recursive(drive_folder_id, local_dir)

    def _download_folder_recursive(self, folder_id: str, dest: Path) -> None:
        results = (
            self._svc.files()
            .list(
                q=f"'{folder_id}' in parents and trashed=false",
                fields="files(id, name, mimeType)",
                pageSize=1000,
            )
            .execute()
        )
        for f in results.get("files", []):
            target = dest / f["name"]
            if f["mimeType"] == "application/vnd.google-apps.folder":
                target.mkdir(exist_ok=True)
                self._download_folder_recursive(f["id"], target)
            else:
                print(f"[DriveSync] Downloading {target}…")
                self.download_file(f["id"], target)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _find_file(self, name: str, parent_id: str) -> str | None:
        name_escaped = name.replace("'", "\\'")
        results = (
            self._svc.files()
            .list(
                q=f"name='{name_escaped}' and '{parent_id}' in parents and trashed=false",
                fields="files(id)",
                pageSize=1,
            )
            .execute()
        )
        files = results.get("files", [])
        return files[0]["id"] if files else None

    def _ensure_folder(self, name: str, parent_id: str) -> str:
        existing = self._find_file(name, parent_id)
        if existing:
            return existing
        meta = {
            "name": name,
            "mimeType": "application/vnd.google-apps.folder",
            "parents": [parent_id],
        }
        result = self._svc.files().create(body=meta, fields="id").execute()
        return result["id"]


# ── convenience functions ─────────────────────────────────────────────────────

def sync_upload_all() -> None:
    """Push all private asset folders to Drive. Run after recording new data."""
    from .config import RAW_RECORDINGS_DIR, WAKEWORD_DATASET_DIR, VOICEPRINT_STORE_PATH

    ds = DriveSync()
    ds.upload_folder(
        RAW_RECORDINGS_DIR,
        os.environ["DRIVE_RAW_RECORDINGS_FOLDER_ID"],
    )
    ds.upload_folder(
        WAKEWORD_DATASET_DIR,
        os.environ["DRIVE_WAKEWORD_DATASET_FOLDER_ID"],
    )
    voiceprint_file = VOICEPRINT_STORE_PATH
    if voiceprint_file.exists():
        ds.upload_file(voiceprint_file, os.environ["DRIVE_VOICEPRINTS_FOLDER_ID"])


def sync_download_all() -> None:
    """Pull all private asset folders from Drive. Run on a fresh clone."""
    from .config import RAW_RECORDINGS_DIR, WAKEWORD_DATASET_DIR, VOICEPRINT_STORE_PATH

    ds = DriveSync()
    ds.download_folder(
        os.environ["DRIVE_RAW_RECORDINGS_FOLDER_ID"],
        RAW_RECORDINGS_DIR,
    )
    ds.download_folder(
        os.environ["DRIVE_WAKEWORD_DATASET_FOLDER_ID"],
        WAKEWORD_DATASET_DIR,
    )
    ds.download_folder(
        os.environ["DRIVE_VOICEPRINTS_FOLDER_ID"],
        VOICEPRINT_STORE_PATH.parent,
    )
