"""
CONTINUATION PRUNER — Smarter approach
=======================================
Issues found:
  - 32,698 conflicts remain after 2000 iterations
  - Main causes: BANC 3-4 syn edges, sexual dimorphism (FAFB=female, MCNS=male)
  - Greedy pruning converging slowly

Strategy:
  1. Pre-filter: remove sexually dimorphic neurons FIRST (they'll never agree)
  2. Pre-filter: keep only well-proofread neurons
  3. Continue greedy pruning on clean subset (much faster convergence)
"""
import pandas as pd
import pyarrow.feather as feather
import time
from collections import defaultdict

print("=" * 60)
print("  SMARTER PRUNER — Pre-filter + Continue")
print("=" * 60)

# ─── Load metadata for filtering ──────────────────────────────
print("\nLoading BANC metadata for pre-filtering...")
meta = feather.read_feather('banc_888_meta.feather')
print(f"Metadata shape: {meta.shape}")
print(f"sexually_dimorphic values: {meta['sexually_dimorphic'].value_counts().head(10).to_dict()}")
print(f"roughly_proofread values:  {meta['roughly_proofread'].value_counts().to_dict()}")

# ─── Load the pre-pruned triplets (output from last run) ──────
print("\nLoading submission_raw.csv (11,072 triplets)...")
submission = pd.read_csv('submission_raw.csv')

# Merge metadata back in
meta_sub = meta[['root_626', 'fafb_match', 'malecns_match',
                  'sexually_dimorphic', 'roughly_proofread',
                  'proofread', 'super_class', 'cell_class']].copy()
meta_sub['BANC'] = pd.to_numeric(meta_sub['root_626'], errors='coerce')
meta_sub['FAFB'] = meta_sub['fafb_match'].astype(str)
meta_sub['MCNS'] = meta_sub['malecns_match'].astype(str)

# Make submission BANC column numeric too for merge
submission['BANC_num'] = pd.to_numeric(submission['BANC'], errors='coerce')
meta_sub_merge = meta_sub[['BANC', 'sexually_dimorphic', 'roughly_proofread', 'super_class']].rename(columns={'BANC': 'BANC_num'})

submission_meta = submission.merge(meta_sub_merge, on='BANC_num', how='left')

print(f"\nSexual dimorphism in our triplets:")
print(submission_meta['sexually_dimorphic'].value_counts().to_string())
print(f"\nProofread status:")
print(submission_meta['roughly_proofread'].value_counts().to_string())

# ─── Pre-filter: remove known problem neurons ─────────────────
print("\n=== PRE-FILTERING ===")

# KEY: 'isomorphic' = same across sexes (GOOD), 'dimorphic'/'female-specific'/'male-specific' = BAD
print(f"\nsexually_dimorphic values in our triplets:")
print(submission_meta['sexually_dimorphic'].value_counts().to_string())

isomorphic_mask = submission_meta['sexually_dimorphic'] == 'isomorphic'
print(f"\nIsomorphic (sex-invariant) neurons: {isomorphic_mask.sum():,} / {len(submission):,}")

# Use isomorphic filter — removes neurons known to differ between male and female flies
if isomorphic_mask.sum() > 1000:
    filtered = submission[isomorphic_mask].reset_index(drop=True)
    print(f"Using isomorphic filter: {len(filtered):,} triplets (removed {(~isomorphic_mask).sum():,} sex-specific)")
else:
    filtered = submission.copy().reset_index(drop=True)
    print(f"Filter too aggressive, using all: {len(filtered):,} triplets")

# ─── Load edge lists ──────────────────────────────────────────
print("\nLoading edge lists...")
t0 = time.time()
fafb_df = pd.read_csv('fafb_783_edge_list.csv'); fafb_df.columns = ['src','tgt']
banc_df = pd.read_csv('banc_626_edge_list.csv'); banc_df.columns = ['src','tgt']
mcns_df = pd.read_csv('mcns_0.9_edge_list.csv'); mcns_df.columns = ['src','tgt']
print(f"Loaded in {time.time()-t0:.1f}s")

# ─── Build index-based edge sets ──────────────────────────────
print("Building edge sets...")
banc_list = filtered['BANC'].tolist()
fafb_list = filtered['FAFB'].tolist()
mcns_list = filtered['MCNS'].tolist()

banc_to_idx = {b: i for i, b in enumerate(banc_list)}
fafb_to_idx = {f: i for i, f in enumerate(fafb_list)}
mcns_to_idx = {m: i for i, m in enumerate(mcns_list)}

banc_df_s = banc_df.astype(str)
fafb_df_s = fafb_df.astype(str)
mcns_df_s = mcns_df.astype(str)

banc_int = banc_df_s[banc_df_s['src'].isin(banc_to_idx) & banc_df_s['tgt'].isin(banc_to_idx)]
fafb_int = fafb_df_s[fafb_df_s['src'].isin(fafb_to_idx) & fafb_df_s['tgt'].isin(fafb_to_idx)]
mcns_int = mcns_df_s[mcns_df_s['src'].isin(mcns_to_idx) & mcns_df_s['tgt'].isin(mcns_to_idx)]

banc_idx_edges = set((banc_to_idx[s], banc_to_idx[t])
    for s,t in zip(banc_int['src'],banc_int['tgt'])
    if s in banc_to_idx and t in banc_to_idx)
fafb_idx_edges = set((fafb_to_idx[s], fafb_to_idx[t])
    for s,t in zip(fafb_int['src'],fafb_int['tgt'])
    if s in fafb_to_idx and t in fafb_to_idx)
mcns_idx_edges = set((mcns_to_idx[s], mcns_to_idx[t])
    for s,t in zip(mcns_int['src'],mcns_int['tgt'])
    if s in mcns_to_idx and t in mcns_to_idx)

print(f"Internal edges — BANC:{len(banc_idx_edges):,} FAFB:{len(fafb_idx_edges):,} MCNS:{len(mcns_idx_edges):,}")

# ─── Count initial conflicts ──────────────────────────────────
all_edges = banc_idx_edges | fafb_idx_edges | mcns_idx_edges
init_conflicts = sum(
    1 for (i,j) in all_edges
    if not (((i,j) in banc_idx_edges) == ((i,j) in fafb_idx_edges) == ((i,j) in mcns_idx_edges))
)
print(f"Initial conflicts after pre-filter: {init_conflicts:,}")
print(f"Reduction vs original 11,072 set:   {184573 - init_conflicts:,} fewer conflicts")

# ─── Greedy pruning (no iteration limit) ─────────────────────
print(f"\n=== GREEDY PRUNING (no limit, will converge) ===")
t = time.time()
active = set(range(len(filtered)))

iteration = 0
while True:
    all_active = banc_idx_edges | fafb_idx_edges | mcns_idx_edges
    conflicts = defaultdict(int)
    total = 0

    for (i,j) in all_active:
        if i not in active or j not in active: continue
        in_b = (i,j) in banc_idx_edges
        in_f = (i,j) in fafb_idx_edges
        in_m = (i,j) in mcns_idx_edges
        if not (in_b == in_f == in_m):
            total += 1
            conflicts[i] += 1
            conflicts[j] += 1

    if total == 0:
        print(f"\n✅ CONVERGED! 0 conflicts after {iteration} removals!")
        break

    worst = max(conflicts, key=conflicts.get)
    active.discard(worst)
    banc_idx_edges = {(i,j) for i,j in banc_idx_edges if i in active and j in active}
    fafb_idx_edges = {(i,j) for i,j in fafb_idx_edges if i in active and j in active}
    mcns_idx_edges = {(i,j) for i,j in mcns_idx_edges if i in active and j in active}

    iteration += 1
    if iteration % 100 == 0:
        elapsed = time.time() - t
        print(f"  iter {iteration:4d}: {len(active):,} neurons, {total:,} conflicts ({elapsed:.0f}s)")

# ─── Save final perfectly isomorphic result ───────────────────
final = filtered.iloc[sorted(active)].copy()
print(f"\n=== FINAL PERFECTLY ISOMORPHIC RESULT ===")
print(f"N = {len(final):,} neurons")
print(f"Removed: {len(filtered) - len(final):,} neurons")
print(f"Time: {time.time()-t:.1f}s")

final[['BANC','FAFB','MCNS']].to_csv('submission_FINAL_VALID.csv', index=False)
print(f"\nSaved perfectly isomorphic submission to: submission_FINAL_VALID.csv")
print(final[['BANC','FAFB','MCNS']].head(10).to_string())
