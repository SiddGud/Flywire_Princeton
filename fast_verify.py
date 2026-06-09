"""
fast_verify.py
==============
Fast numpy-based directed edge isomorphism verifier.
Uses boolean matrix operations instead of per-pair Python loops.
Verifies submission_FINAL_4833.csv in ~30 seconds.
"""

import pandas as pd
import numpy as np
import glob, time

# Find best submission
files = [f for f in glob.glob('submission_FINAL_*.csv') + glob.glob('submission_OVERNIGHT_*.csv')
         if f.replace('.csv','').split('_')[-1].isdigit()]
if not files:
    print("No submission files found!")
    exit()

best_file = max(files, key=lambda f: int(f.split('_')[-1].replace('.csv','')))
print(f"Verifying: {best_file}")
sub = pd.read_csv(best_file, dtype=str)
N = len(sub)
print(f"N = {N:,}")

# Check 1: Bijection
print("\n[CHECK 1] Bijection...")
for col in ['BANC','FAFB','MCNS']:
    assert sub[col].nunique() == N, f"{col} not unique!"
    print(f"  {col}: OK")

# Check 2: Directed edge isomorphism via numpy boolean matrices
print("\n[CHECK 2] Directed edge isomorphism (numpy matrices)...")
print("Loading edge lists...")
t0 = time.time()
fafb_df = pd.read_csv('fafb_783_edge_list.csv', header=None, names=['src','tgt'], dtype=str)
banc_df = pd.read_csv('banc_626_edge_list.csv', header=None, names=['src','tgt'], dtype=str)
mcns_df = pd.read_csv('mcns_0.9_edge_list.csv', header=None, names=['src','tgt'], dtype=str)
print(f"Loaded in {time.time()-t0:.1f}s")

# Build index maps
banc_nodes = sub['BANC'].tolist()
fafb_nodes = sub['FAFB'].tolist()
mcns_nodes = sub['MCNS'].tolist()

banc_idx = {n: i for i, n in enumerate(banc_nodes)}
fafb_idx = {n: i for i, n in enumerate(fafb_nodes)}
mcns_idx = {n: i for i, n in enumerate(mcns_nodes)}

# Build NxN directed adjacency matrices
print(f"Building {N}x{N} directed adjacency matrices...")
t0 = time.time()

mat_b = np.zeros((N, N), dtype=bool)
mat_f = np.zeros((N, N), dtype=bool)
mat_m = np.zeros((N, N), dtype=bool)

for df, mat, idx_src, idx_tgt in [
    (banc_df, mat_b, banc_idx, banc_idx),
    (fafb_df, mat_f, fafb_idx, fafb_idx),
    (mcns_df, mat_m, mcns_idx, mcns_idx)
]:
    mask = df['src'].isin(idx_src) & df['tgt'].isin(idx_tgt)
    df_int = df[mask]
    rows = df_int['src'].map(idx_src).values
    cols = df_int['tgt'].map(idx_tgt).values
    mat[rows, cols] = True

print(f"Matrices built in {time.time()-t0:.1f}s")
print(f"  BANC internal edges: {mat_b.sum():,}")
print(f"  FAFB internal edges: {mat_f.sum():,}")
print(f"  MCNS internal edges: {mat_m.sum():,}")

# Compute violations: positions where the three matrices disagree
print("\nComputing violations (matrix XOR)...")
t0 = time.time()
violations = ((mat_b ^ mat_f) | (mat_b ^ mat_m))
# Exclude diagonal (self-loops don't count)
np.fill_diagonal(violations, False)
total_violations = violations.sum()
print(f"Completed in {time.time()-t0:.1f}s")

print(f"\nTotal directed edge violations: {total_violations:,}")
if total_violations == 0:
    print("\n✅ PERFECT ISOMORPHISM VERIFIED!")
    print(f"✅ N = {N:,} — 100% valid, ready to submit!")
else:
    print(f"\n❌ {total_violations:,} violations found!")
    # Show examples
    viol_rows, viol_cols = np.where(violations)
    print("Example violations:")
    for k in range(min(5, len(viol_rows))):
        i, j = viol_rows[k], viol_cols[k]
        print(f"  [{i},{j}]: BANC={mat_b[i,j]} FAFB={mat_f[i,j]} MCNS={mat_m[i,j]}")
        print(f"         BANC: {banc_nodes[i]}->{banc_nodes[j]}")
        print(f"         FAFB: {fafb_nodes[i]}->{fafb_nodes[j]}")
        print(f"         MCNS: {mcns_nodes[i]}->{mcns_nodes[j]}")
