import ee, os, json, requests
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
print(f"Access token obtained: {token[:40]}...")

# Search Drive
headers = {"Authorization": f"Bearer {token}"}
params = {
    "q": "name contains 'aral_saxaul_fs_30m_mosaic' and mimeType contains 'tiff'",
    "spaces": "drive",
    "fields": "files(id, name, size)",
}
r = requests.get("https://www.googleapis.com/drive/v3/files", headers=headers, params=params)
data = r.json()
files = data.get("files", [])
print(f"Found {len(files)} files:")
for f in files:
    print(f"  {f['name']} ({int(f['size'])/1e6:.0f} MB) id={f['id']}")

for i, f in enumerate(files):
    fid = f["id"]
    fname = f["name"]
    out_path = os.path.join(str(config.output_dir / "data"), f"feature_stack_30m_tile{i}.tif")

    # Share file publicly
    body = {"role": "reader", "type": "anyone"}
    requests.post(f"https://www.googleapis.com/drive/v3/files/{fid}/permissions",
                  headers=headers, json=body)

    # Create share link
    link = f"https://drive.google.com/file/d/{fid}/view"
    print(f"\nDownload via geemap:")
    print(f"import geemap")
    print(f"geemap.download_from_gdrive(gfile_url='{link}', file_name='feature_stack_30m.tif', out_dir='{str(config.output_dir / 'data')}')")

    # Direct download
    dl_url = f"https://www.googleapis.com/drive/v3/files/{fid}?alt=media"
    size_mb = int(files[0]["size"]) / 1e6
    print(f"\nDirect download ({size_mb:.0f} MB)...")
    r2 = requests.get(dl_url, headers=headers, stream=True, timeout=3600)
    r2.raise_for_status()
    with open(out_path, "wb") as f:
        for chunk in r2.iter_content(chunk_size=8*1024*1024):
            f.write(chunk)
    dl_size = os.path.getsize(out_path) / 1e6
    print(f"Downloaded: {out_path} ({dl_size:.0f} MB)")
else:
    print("No files found. Check Drive folder 'ee_exports/'")
