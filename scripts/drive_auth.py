"""One-time OAuth authorisation for Google Drive uploads.

Run this once:
    python scripts/drive_auth.py

A browser window will open. Sign in with your Google account and allow
access. The token is saved to drive_token.json (gitignored) and reused
automatically on all future sync runs — no browser needed after this.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCOPES = ["https://www.googleapis.com/auth/drive.file"]
OAUTH_CREDS = ROOT / "oauth_credentials.json"
TOKEN_FILE  = ROOT / "drive_token.json"


def main() -> None:
    try:
        from google_auth_oauthlib.flow import InstalledAppFlow
    except ImportError:
        print("Installing google-auth-oauthlib…")
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install",
                               "google-auth-oauthlib", "-q"])
        from google_auth_oauthlib.flow import InstalledAppFlow

    flow = InstalledAppFlow.from_client_secrets_file(str(OAUTH_CREDS), SCOPES)
    creds = flow.run_local_server(port=0)
    TOKEN_FILE.write_text(creds.to_json())
    print(f"✓ Token saved to {TOKEN_FILE}")
    print("You can now run: python scripts/drive_sync.py upload")


if __name__ == "__main__":
    main()
