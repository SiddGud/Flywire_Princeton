"""
Try all methods to get BANC metadata with cross-dataset match IDs.
"""
import sys
PY = sys.executable

# ─── Method 1: banc Python package ───────────────────────────
print("=" * 55)
print("METHOD 1: banc Python package")
print("=" * 55)
try:
    import banc
    print(f"banc version: {banc.__version__}")
    print(f"Available functions: {[x for x in dir(banc) if not x.startswith('_')]}")

    # Try fetching metadata directly
    for fn_name in ['get_meta', 'fetch_meta', 'get_annotations', 'meta',
                    'get_neurons', 'neuron_info', 'cell_info']:
        if hasattr(banc, fn_name):
            print(f"\nFound function: banc.{fn_name}")
            try:
                result = getattr(banc, fn_name)()
                print(f"  Result type: {type(result)}")
                if hasattr(result, 'columns'):
                    print(f"  Columns: {list(result.columns)[:20]}")
                    print(f"  Shape: {result.shape}")
            except Exception as e:
                print(f"  Error: {e}")
except Exception as e:
    print(f"banc import failed: {e}")

# ─── Method 2: Try banc.codex / banc.cave ────────────────────
print("\n" + "=" * 55)
print("METHOD 2: banc submodules")
print("=" * 55)
try:
    import banc
    # Look for submodules
    import importlib, pkgutil
    banc_path = banc.__path__
    submods = [m.name for m in pkgutil.walk_packages(banc_path)]
    print(f"Submodules: {submods}")

    # Try common ones
    for mod in ['banc.cave', 'banc.codex', 'banc.annotations', 'banc.meta']:
        try:
            m = importlib.import_module(mod)
            print(f"\n  {mod} loaded OK")
            print(f"  Functions: {[x for x in dir(m) if not x.startswith('_')][:15]}")
        except Exception as e:
            print(f"  {mod}: {e}")
except Exception as e:
    print(f"Error: {e}")

# ─── Method 3: CAVEclient with banc token helper ─────────────
print("\n" + "=" * 55)
print("METHOD 3: CAVEclient via banc")
print("=" * 55)
try:
    from caveclient import CAVEclient
    # Try with banc's built-in token infrastructure
    try:
        import banc
        if hasattr(banc, 'get_cave_client') or hasattr(banc, 'cave_client'):
            fn = getattr(banc, 'get_cave_client', None) or getattr(banc, 'cave_client')
            client = fn()
            print(f"Got client via banc: {type(client)}")
        else:
            # Try direct CAVEclient with banc server
            client = CAVEclient(server_address="https://global.daf-apis.com")
            print("Connected without datastack")
            # Try to get token instruction
            print(f"Auth URL: {client.auth.get_new_token()}")
    except Exception as e:
        print(f"banc client error: {e}")
except Exception as e:
    print(f"CAVEclient error: {e}")

# ─── Method 4: Direct Dataverse download with session cookie ──
print("\n" + "=" * 55)
print("METHOD 4: Dataverse with API token (if we have one)")
print("=" * 55)
print("File ID: 13923343")
print("File: banc_888_meta.feather (57MB, has FAFB+MCNS match IDs)")
print("URL: https://dataverse.harvard.edu/api/access/datafile/13923343")
print()
print("To get API token:")
print("  1. Go to: https://dataverse.harvard.edu/")
print("  2. Log in with your institution email")
print("  3. Click your name (top right) -> API Token -> Create Token")
print("  4. Paste the token here and rerun with:")
print("     headers = {'X-Dataverse-key': 'YOUR_TOKEN'}")
print()

# Try without token first (might work if file is truly unrestricted)
import urllib.request
try:
    req = urllib.request.Request(
        "https://dataverse.harvard.edu/api/access/datafile/13923343",
        headers={"User-Agent": "Mozilla/5.0", "Accept": "*/*"}
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        first_bytes = r.read(200)
        print(f"Status: {r.status} - File IS accessible without token!")
        print(f"Content-Type: {r.headers.get('Content-Type')}")
except Exception as e:
    print(f"Without token: {e}")

# ─── Method 5: Check if banc_888_meta downloadable via codex ──
print("\n" + "=" * 55)
print("METHOD 5: Try FlyWire Codex API for BANC metadata")
print("=" * 55)
import urllib.request, json
codex_urls = [
    "https://codex.flywire.ai/api/banc/meta",
    "https://codex.flywire.ai/api/banc/cell_info",
    "https://codex.flywire.ai/api/data/banc",
]
for url in codex_urls:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=8) as r:
            data = r.read(500)
            if b'<!doctype' not in data.lower():
                print(f"SUCCESS: {url}")
                print(f"  Data: {data[:200]}")
            else:
                print(f"HTML (login needed): {url}")
    except Exception as e:
        print(f"Failed: {url} -> {e}")
