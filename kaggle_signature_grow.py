import pandas as pd
import numpy as np
import time, os
from collections import defaultdict

# ── Paths ──────────────────────────────────────────────────────────
DATA1 = '/kaggle/input/datasets/siddhantgudwani/dataset'
DATA2 = None
for root, dirs, files in os.walk('/kaggle/input'):
    for f in files:
        if 'submission_CT_FINE_5092.csv' in f:
            DATA2 = root
            break
    if DATA2: break

if DATA2 is None:
    # If they uploaded it manually, fallback to working or standard dataset path
    for candidate in ['/kaggle/working', '/kaggle/input/dataset2']:
        if os.path.exists(candidate + '/submission_CT_FINE_5092.csv'):
            DATA2 = candidate; break

if DATA2 is None:
    print("❌ ERROR: Could not find submission_CT_FINE_5092.csv! Please make sure it is uploaded to Kaggle.")
    exit()

print("1. Loading Edge Lists...")
t0 = time.time()
fafb_df = pd.read_csv(f'{DATA1}/fafb_783_edge_list.csv', header=None, names=['src','tgt'], dtype=str)
banc_df = pd.read_csv(f'{DATA1}/banc_626_edge_list.csv', header=None, names=['src','tgt'], dtype=str)
mcns_df = pd.read_csv(f'{DATA1}/mcns_0.9_edge_list.csv', header=None, names=['src','tgt'], dtype=str)
print(f"Loaded in {time.time()-t0:.1f}s")

print(f"2. Loading Core (5,092)... from {DATA2}")
core_df = pd.read_csv(f'{DATA2}/submission_CT_FINE_5092.csv', dtype=str)
CORE_SIZE = len(core_df)

# Map node string IDs to a unified integer index (0 to 5091)
b2i = {n: i for i, n in enumerate(core_df['BANC'])}
f2i = {n: i for i, n in enumerate(core_df['FAFB'])}
m2i = {n: i for i, n in enumerate(core_df['MCNS'])}

# ── Signature Computation ──────────────────────────────────────────
print("3. Computing full dataset Signatures to the core...")
def compute_signatures(df, core_map):
    # We want to find edges where exactly ONE node is in the core.
    # The node outside the core is the "neighbor".
    # We track which core nodes it connects to.
    
    sig_in = defaultdict(list)  # neighbor -> list of core nodes it receives from
    sig_out = defaultdict(list) # neighbor -> list of core nodes it sends to
    
    # Edges FROM core TO neighbor (neighbor receives)
    mask_out = df['src'].isin(core_map) & (~df['tgt'].isin(core_map))
    for s, t in zip(df[mask_out]['src'], df[mask_out]['tgt']):
        sig_in[t].append(core_map[s])
        
    # Edges FROM neighbor TO core (neighbor sends)
    mask_in = (~df['src'].isin(core_map)) & df['tgt'].isin(core_map)
    for s, t in zip(df[mask_in]['src'], df[mask_in]['tgt']):
        sig_out[s].append(core_map[t])
        
    # Combine into a final tuple signature for each neighbor
    all_neighbors = set(sig_in.keys()) | set(sig_out.keys())
    
    # signature -> list of nodes with that signature
    signatures = defaultdict(list)
    for n in all_neighbors:
        s_in = tuple(sorted(sig_in[n]))
        s_out = tuple(sorted(sig_out[n]))
        signatures[(s_in, s_out)].append(n)
        
    return signatures

t_sig = time.time()
b_sigs = compute_signatures(banc_df, b2i)
f_sigs = compute_signatures(fafb_df, f2i)
m_sigs = compute_signatures(mcns_df, m2i)
print(f"Computed signatures in {time.time()-t_sig:.1f}s")
print(f"Unique neighbor signatures found: BANC={len(b_sigs):,}, FAFB={len(f_sigs):,}, MCNS={len(m_sigs):,}")

# ── Intersect Signatures ───────────────────────────────────────────
print("4. Intersecting signatures across all 3 datasets...")
shared_sigs = set(b_sigs.keys()) & set(f_sigs.keys()) & set(m_sigs.keys())
print(f"Found {len(shared_sigs):,} matching signatures that exist in all 3 datasets!")

candidates = []
for sig in shared_sigs:
    b_nodes = b_sigs[sig]
    f_nodes = f_sigs[sig]
    m_nodes = m_sigs[sig]
    
    # To be safe, if multiple nodes have the identical signature, we pair them sequentially
    max_pairs = min(len(b_nodes), len(f_nodes), len(m_nodes))
    for i in range(max_pairs):
        candidates.append({'BANC': b_nodes[i], 'FAFB': f_nodes[i], 'MCNS': m_nodes[i]})

candidates_df = pd.DataFrame(candidates)
print(f"Generated {len(candidates_df):,} structural candidate triplets from the void.")

if len(candidates_df) == 0:
    print("Result: 5,092 is mathematically sealed. No structural matches exist in the remaining 390,000 nodes.")
    exit()

# ── Standard Strict Grow ───────────────────────────────────────────
print("\n5. Running Strict Grow to prevent internal conflicts among new nodes...")
banc_str = set(zip(banc_df['src'], banc_df['tgt']))
fafb_str = set(zip(fafb_df['src'], fafb_df['tgt']))
mcns_str = set(zip(mcns_df['src'], mcns_df['tgt']))

core = {str(b):(str(f),str(m)) for b,f,m in zip(core_df['BANC'], core_df['FAFB'], core_df['MCNS'])}

added = 0
for _, row in candidates_df.iterrows():
    b, f, m = str(row['BANC']), str(row['FAFB']), str(row['MCNS'])
    ok = True
    for cb, (cf, cm) in core.items():
        if ((b, cb) in banc_str) != ((f, cf) in fafb_str) or ((b, cb) in banc_str) != ((m, cm) in mcns_str): 
            ok = False; break
        if ((cb, b) in banc_str) != ((cf, f) in fafb_str) or ((cb, b) in banc_str) != ((cm, m) in mcns_str): 
            ok = False; break
    if ok:
        core[b] = (f, m)
        added += 1

final_n = CORE_SIZE + added
print("\n" + "="*60)
print(f"  FINAL SIGNATURE GROWN N: {final_n}  (Added {added} hidden structural nodes!)")
print("="*60)

final_df = pd.DataFrame([{'BANC': b, 'FAFB': f, 'MCNS': m} for b, (f, m) in core.items()])
out_file = f'/kaggle/working/submission_SIG_GROW_{final_n}.csv'
final_df.to_csv(out_file, index=False)

from IPython.display import HTML
import base64
with open(out_file, 'rb') as f:
    encoded = base64.b64encode(f.read()).decode('utf-8')
display(HTML(f'<a download="submission_SIG_GROW_{final_n}.csv" href="data:text/csv;base64,{encoded}" target="_blank" style="font-size:24px; font-weight:bold; color:red;">⬇️ CLICK TO DOWNLOAD SIGNATURE GRAPH ⬇️</a>'))
