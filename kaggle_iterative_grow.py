import pandas as pd
import numpy as np
import time, os
from collections import defaultdict

# ── Paths ──────────────────────────────────────────────────────────
DATA1 = '/kaggle/input/datasets/siddhantgudwani/dataset'

# Auto-find the 7822 submission CSV
CORE_FILE = None
for root, dirs, files in os.walk('/kaggle/input'):
    for f in files:
        if 'submission_SIG_GROW_7822.csv' in f or 'SIG_GROW_7822' in f:
            CORE_FILE = os.path.join(root, f)
            break
    if CORE_FILE: break

if CORE_FILE is None:
    print("❌ ERROR: Could not find submission_SIG_GROW_7822.csv!")
    print("   Please upload it to Kaggle as a dataset input.")
    exit()

print("="*60)
print("  ITERATIVE FIXED-POINT SIGNATURE GROW")
print("="*60)

print("\n1. Loading Edge Lists...")
t0 = time.time()
fafb_df = pd.read_csv(f'{DATA1}/fafb_783_edge_list.csv', header=None, names=['src','tgt'], dtype=str)
banc_df = pd.read_csv(f'{DATA1}/banc_626_edge_list.csv', header=None, names=['src','tgt'], dtype=str)
mcns_df = pd.read_csv(f'{DATA1}/mcns_0.9_edge_list.csv', header=None, names=['src','tgt'], dtype=str)
print(f"Loaded in {time.time()-t0:.1f}s")

# Pre-build edge sets for fast lookup (used in strict grow)
print("2. Pre-building edge sets for O(1) lookup...")
t0 = time.time()
banc_set = set(zip(banc_df['src'], banc_df['tgt']))
fafb_set = set(zip(fafb_df['src'], fafb_df['tgt']))
mcns_set = set(zip(mcns_df['src'], mcns_df['tgt']))
print(f"Done in {time.time()-t0:.1f}s")

# ── Signature function ─────────────────────────────────────────────
def compute_signatures(df, core_map):
    """For every neuron OUTSIDE the core, compute its connection signature to the core."""
    sig_in  = defaultdict(list)   # neighbor <- core (neighbor receives from core)
    sig_out = defaultdict(list)   # neighbor -> core (neighbor sends to core)

    mask_out = df['src'].isin(core_map) & (~df['tgt'].isin(core_map))
    for s, t in zip(df[mask_out]['src'], df[mask_out]['tgt']):
        sig_in[t].append(core_map[s])

    mask_in = (~df['src'].isin(core_map)) & df['tgt'].isin(core_map)
    for s, t in zip(df[mask_in]['src'], df[mask_in]['tgt']):
        sig_out[s].append(core_map[t])

    all_neighbors = set(sig_in.keys()) | set(sig_out.keys())
    signatures = defaultdict(list)
    for n in all_neighbors:
        key = (tuple(sorted(sig_in[n])), tuple(sorted(sig_out[n])))
        signatures[key].append(n)
    return signatures

# ── Strict Grow ────────────────────────────────────────────────────
def strict_grow(candidates_df, core):
    """Add only candidates that don't conflict with ANY existing core node."""
    added = 0
    for _, row in candidates_df.iterrows():
        b, f, m = str(row['BANC']), str(row['FAFB']), str(row['MCNS'])
        if b in core: continue  # already in core
        ok = True
        for cb, (cf, cm) in list(core.items()):
            b_has_bc = (b, cb) in banc_set;  f_has_bc = (f, cf) in fafb_set;  m_has_bc = (m, cm) in mcns_set
            b_has_cb = (cb, b) in banc_set;  f_has_cb = (cf, f) in fafb_set;  m_has_cb = (cm, m) in mcns_set
            if b_has_bc != f_has_bc or b_has_bc != m_has_bc:  ok = False; break
            if b_has_cb != f_has_cb or b_has_cb != m_has_cb:  ok = False; break
        if ok:
            core[b] = (f, m)
            added += 1
    return added

# ── Load starting core ─────────────────────────────────────────────
print(f"\n3. Loading starting core from: {CORE_FILE}")
core_df = pd.read_csv(CORE_FILE, dtype=str)
print(f"   Starting N = {len(core_df):,}")

# Build core dict: BANC_id -> (FAFB_id, MCNS_id)
core = {str(b): (str(f), str(m)) for b, f, m in zip(core_df['BANC'], core_df['FAFB'], core_df['MCNS'])}

# ── Iterative Fixed-Point Loop ─────────────────────────────────────
print("\n4. Starting Iterative Fixed-Point Expansion...")
round_num = 0
while True:
    round_num += 1
    N_before = len(core)
    t_round = time.time()

    # Build current core maps
    b2i = {b: i for i, b in enumerate(core.keys())}
    f2i = {f: i for i, (f, m) in enumerate(core.values())}
    m2i = {m: i for i, (f, m) in enumerate(core.values())}

    # Compute signatures for all 3 datasets against current core
    b_sigs = compute_signatures(banc_df, b2i)
    f_sigs = compute_signatures(fafb_df, f2i)
    m_sigs = compute_signatures(mcns_df, m2i)

    # Intersect: only keep signatures that exist in ALL 3 datasets
    shared_sigs = set(b_sigs.keys()) & set(f_sigs.keys()) & set(m_sigs.keys())

    # Build candidate triplets
    candidates = []
    for sig in shared_sigs:
        b_nodes = b_sigs[sig]
        f_nodes = f_sigs[sig]
        m_nodes = m_sigs[sig]
        max_pairs = min(len(b_nodes), len(f_nodes), len(m_nodes))
        for i in range(max_pairs):
            b, f, m = b_nodes[i], f_nodes[i], m_nodes[i]
            if b not in core:  # skip if already in core
                candidates.append({'BANC': b, 'FAFB': f, 'MCNS': m})

    candidates_df = pd.DataFrame(candidates)

    if len(candidates_df) == 0:
        print(f"\n  Round {round_num}: No new candidate signatures found. CONVERGED! ✅")
        break

    print(f"  Round {round_num}: Found {len(candidates_df):,} signature candidates...", end=' ')

    # Strict grow
    added = strict_grow(candidates_df, core)
    N_after = len(core)

    elapsed = time.time() - t_round
    print(f"Added {added:,} | N = {N_after:,}  ({elapsed:.1f}s)")

    if added == 0:
        print(f"\n  Round {round_num}: Signature candidates exist but none passed strict grow. CONVERGED! ✅")
        break

# ── Save Final Result ──────────────────────────────────────────────
final_n = len(core)
print("\n" + "="*60)
print(f"  FINAL CONVERGED N: {final_n:,}")
print("="*60)

final_df = pd.DataFrame([{'BANC': b, 'FAFB': f, 'MCNS': m} for b, (f, m) in core.items()])
out_file = f'/kaggle/working/submission_ITER_GROW_{final_n}.csv'
final_df.to_csv(out_file, index=False)
print(f"Saved to: {out_file}")

from IPython.display import HTML
import base64
with open(out_file, 'rb') as fh:
    encoded = base64.b64encode(fh.read()).decode('utf-8')
display(HTML(f'<a download="submission_ITER_GROW_{final_n}.csv" href="data:text/csv;base64,{encoded}" target="_blank" style="font-size:24px; font-weight:bold; color:green;">⬇️ CLICK TO DOWNLOAD ITER GROW N={final_n} ⬇️</a>'))
