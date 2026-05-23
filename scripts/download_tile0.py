import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ee, json, requests
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from src.utils import initialize_gee, logger
from src.config import config

initialize_gee(project="tribal-dispatch-494405-u4")

cred_path = ee.oauth.get_credentials_path()
with open(cred_path) as f:
    stored = json.load(f)

new_creds = Credentials(
    token=None,
    refresh_token=stored["refresh_token"],
    token_uri="https://oauth2.googleapis.com/token",
    client_id=ee.oauth.CLIENT_ID,
    client_secret=ee.oauth.CLIENT_SECRET,
    scopes=["https://www.googleapis.com/auth/drive"],
)
new_creds.refresh(Request())
token = new_creds.token
print(f"Token obtained: {token[:40]}...")

fid = "1Qv6TfBjEJw_X8fGmxZykaB_0AltlfTT-"
headers = {"Authorization": f"Bearer {token}"}

r = requests.get(f"https://www.googleapis.com/drive/v3/files/{fid}", headers=headers)
meta = r.json()
name = meta.get("name", "unknown")
size_gb = int(meta.get("size", 0)) / 1e9
print(f"File: {name}, size: {size_gb:.2f} GB")

out = str(config.output_dir / "data" / "feature_stack_30m_tile0_redo.tif")
dl_url = f"https://www.googleapis.com/drive/v3/files/{fid}?alt=media"
r2 = requests.get(dl_url, headers=headers, stream=True, timeout=7200)
r2.raise_for_status()
with open(out, "wb") as f:
    for chunk in r2.iter_content(chunk_size=8*1024*1024):
        if chunk:
            f.write(chunk)
print(f"Downloaded: {out} ({os.path.getsize(out)/1e9:.2f} GB)")
