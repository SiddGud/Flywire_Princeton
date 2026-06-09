"""
Fetch BANC metadata — the KEY to this challenge.
BANC has pre-computed fafb_783_match_id fields that link
BANC neurons to their FAFB counterparts.

Trying multiple methods in order:
  1. CAVEclient (official API)
  2. Direct Codex download URL
  3. Harvard Dataverse
"""

import urllib.request
import json
import os
import time

PY311 = r"C:\Users\sahaj\AppData\Local\Programs\Python\Python311\python.exe"

# ─────────────────────────────────────
# Method 1: Try CAVEclient
# ─────────────────────────────────────
print("="*55)
print("  METHOD 1: CAVEclient")
print("="*55)

try:
    from caveclient import CAVEclient
    print("CAVEclient already installed!")
except ImportError:
    print("Installing caveclient...")
    os.system(f'{PY311} -m pip install caveclient -q')
    try:
        from caveclient import CAVEclient
        print("Installed successfully!")
    except ImportError:
        print("Failed to install caveclient")

try:
    # BANC datastack
    client = CAVEclient('brain_and_nerve_cord')
    print(f"\nConnected to BANC!")
    print(f"Available tables:")
    tables = client.annotation.get_tables()
    for t in tables:
        print(f"  - {t}")
except Exception as e:
    print(f"CAVEclient connection failed: {e}")
    print("Trying without auth...")
    try:
        client = CAVEclient('brain_and_nerve_cord', auth_token=None)
        tables = client.annotation.get_tables()
        print(f"Tables (no auth): {tables}")
    except Exception as e2:
        print(f"Also failed: {e2}")

# ─────────────────────────────────────
# Method 2: Try known Codex download URLs
# ─────────────────────────────────────
print("\n" + "="*55)
print("  METHOD 2: Codex Static Download URLs")
print("="*55)

# These are typical Codex download patterns
urls_to_try = [
    "https://codex.flywire.ai/api/download?data_type=cell_info&dataset=banc",
    "https://codex.flywire.ai/api/download?data_type=annotations&dataset=banc",
    "https://codex.flywire.ai/api/cell_info?dataset=banc",
    # FlyWire main FAFB info (this one is often public)
    "https://codex.flywire.ai/api/download?data_type=cell_info&dataset=fafb",
]

for url in urls_to_try:
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as r:
            content = r.read(500)
            print(f"✅ {url[:60]}...")
            print(f"   Status: {r.status}, Content preview: {content[:100]}")
    except Exception as e:
        print(f"❌ {url[:60]}... → {type(e).__name__}: {str(e)[:60]}")

# ─────────────────────────────────────
# Method 3: Harvard Dataverse search
# ─────────────────────────────────────
print("\n" + "="*55)
print("  METHOD 3: Harvard Dataverse Search")
print("="*55)

dataverse_search = "https://dataverse.harvard.edu/api/search?q=BANC+connectome+flywire&type=dataset&per_page=5"
try:
    req = urllib.request.Request(dataverse_search, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=15) as r:
        data = json.loads(r.read())
        items = data.get('data', {}).get('items', [])
        print(f"Found {len(items)} datasets on Harvard Dataverse:")
        for item in items:
            print(f"\n  Title: {item.get('name', 'N/A')}")
            print(f"  URL:   {item.get('url', 'N/A')}")
            print(f"  DOI:   {item.get('global_id', 'N/A')}")
except Exception as e:
    print(f"Dataverse search failed: {e}")

# Also try FlyWire Zenodo
print("\n" + "="*55)
print("  METHOD 4: Zenodo Search")
print("="*55)

zenodo_search = "https://zenodo.org/api/records?q=banc+connectome+flywire&sort=mostrecent&size=5"
try:
    req = urllib.request.Request(zenodo_search, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=15) as r:
        data = json.loads(r.read())
        hits = data.get('hits', {}).get('hits', [])
        print(f"Found {len(hits)} records on Zenodo:")
        for h in hits:
            meta = h.get('metadata', {})
            print(f"\n  Title: {meta.get('title', 'N/A')}")
            print(f"  DOI:   {h.get('doi', 'N/A')}")
            print(f"  URL:   https://zenodo.org/record/{h.get('id', '')}")
            # Check files
            files = h.get('files', [])
            if files:
                print(f"  Files: {[f.get('key','') for f in files[:3]]}")
except Exception as e:
    print(f"Zenodo search failed: {e}")

print("\n" + "="*55)
print("  DONE — check results above for download links")
print("="*55)
