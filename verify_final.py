"""
verify_final.py
===============
Rigorously verifies the final submission CSV against the exact
challenge constraints:
  1. Strict 1-to-1 bijection (no repeated neuron IDs in any column)
  2. DIRECTED edge consistency: for every pair (i, j) of matched neurons,
     the directed edge i->j must be present in ALL three datasets, OR
     absent in ALL three datasets. Direction is preserved.
  3. Reports total violations (must be 0)
"""

import pandas as pd
import glob, time

# Find the best submission
files = sorted(glob.glob('submission_FINAL_*.csv') +
               glob.glob('submission_OVERNIGHT_*.csv') +
               glob.glob('submission_GROWN_*.csv'))
print("Available submission files:")
for f in files:
    df = pd.read_csv(f)
    print(f"  {f}: {len(df):,} rows")

if not files:
    print("No submission files found!")
    exit()

def get_n(f):
    try:
        return int(f.split('_')[-1].replace('.csv',''))
    except:
        return 0

best_file = max(files, key=get_n)
print(f"\nVerifying: {best_file}")
sub = pd.read_csv(best_file, dtype=str)
print(f"Rows: {len(sub):,}")

# ── Check 1: Bijection ──────────────────────────────────────────
print("\n[CHECK 1] Bijection uniqueness...")
ok = True
for col in ['BANC', 'FAFB', 'MCNS']:
    n_unique = sub[col].nunique()
    if n_unique != len(sub):
        print(f"  FAIL: {col} has {n_unique} unique values but {len(sub)} rows!")
        ok = False
    else:
        print(f"  OK: {col} all {n_unique} values unique")

# ── Check 2: DIRECTED edge isomorphism ─────────────────────────
print("\n[CHECK 2] Directed edge isomorphism...")
print("Loading edge lists...")
t0 = time.time()
fafb_df = pd.read_csv('fafb_783_edge_list.csv', header=None, names=['src','tgt'], dtype=str)
banc_df = pd.read_csv('banc_626_edge_list.csv', header=None, names=['src','tgt'], dtype=str)
mcns_df = pd.read_csv('mcns_0.9_edge_list.csv', header=None, names=['src','tgt'], dtype=str)
print(f"Loaded in {time.time()-t0:.1f}s")

# Build DIRECTED edge sets (tuples preserve direction: (src, tgt) != (tgt, src))
banc_nodes = set(sub['BANC'])
fafb_nodes = set(sub['FAFB'])
mcns_nodes = set(sub['MCNS'])

banc_edges = set(zip(banc_df['src'], banc_df['tgt']))
fafb_edges = set(zip(fafb_df['src'], fafb_df['tgt']))
mcns_edges = set(zip(mcns_df['src'], mcns_df['tgt']))

# Build index mapping: BANC_id -> (FAFB_id, MCNS_id)
banc_to_fafb = dict(zip(sub['BANC'], sub['FAFB']))
banc_to_mcns = dict(zip(sub['BANC'], sub['MCNS']))

node_list = sub['BANC'].tolist()
N = len(node_list)
print(f"Checking {N*(N-1):,} directed pairs (N={N}, N*(N-1) ordered pairs)...")

violations = 0
example_violations = []
t0 = time.time()

for i in range(N):
    bi = node_list[i]
    fi = banc_to_fafb[bi]
    mi = banc_to_mcns[bi]
    for j in range(N):
        if i == j:
            continue
        bj = node_list[j]
        fj = banc_to_fafb[bj]
        mj = banc_to_mcns[bj]

        # Check directed edge i->j in each dataset
        e_b = (bi, bj) in banc_edges
        e_f = (fi, fj) in fafb_edges
        e_m = (mi, mj) in mcns_edges

        if not (e_b == e_f == e_m):
            violations += 1
            if len(example_violations) < 5:
                example_violations.append({
                    'BANC': f"{bi}->{bj}",
                    'FAFB': f"{fi}->{fj}",
                    'MCNS': f"{mi}->{mj}",
                    'in_BANC': e_b, 'in_FAFB': e_f, 'in_MCNS': e_m
                })

elapsed = time.time() - t0
print(f"Check completed in {elapsed:.1f}s")
print(f"\nTotal directed edge violations: {violations:,}")

if violations == 0:
    print("\n✅ PERFECT: Submission is a valid directed induced isomorphic subgraph!")
    print(f"✅ N = {len(sub):,} — ready to submit!")
else:
    print(f"\n❌ FAIL: {violations:,} violations found!")
    print("Example violations:")
    for v in example_violations:
        print(f"  {v}")
