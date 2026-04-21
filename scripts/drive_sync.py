"""CLI script to push/pull private assets to/from Google Drive.

Usage:
    python scripts/drive_sync.py upload   # local → Drive
    python scripts/drive_sync.py download # Drive → local
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from src.drive_sync import sync_download_all, sync_upload_all

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "upload":
        print("Uploading private assets to Google Drive…")
        sync_upload_all()
        print("Done.")
    elif cmd == "download":
        print("Downloading private assets from Google Drive…")
        sync_download_all()
        print("Done.")
    else:
        print("Usage: python scripts/drive_sync.py [upload|download]")
        sys.exit(1)
