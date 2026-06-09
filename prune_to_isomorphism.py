"""
SMART PRUNER: Find the largest perfectly isomorphic subset
===========================================================
We have 11,072 triplets but 6.07% violation rate.
Violations are likely BANC 3-4 synapse edges that don't appear in FAFB/MCNS.

Algorithm:
  1. Build edge sets for all 3 datasets (only between matched neurons)
  2. Find all "conflict pairs" where edge status disagrees
  3. For each neuron, count how many conflicts it's involved in
  4. Iteratively remove the most-conflicted neuron
  5. Repeat until 0 conflicts → perfectly isomorphic subgraph!

This gives us the LARGEST clean submittable set.
"""
import pandas as pd
import numpy as np
import time
from collections import defaultdict

print("=" * 60)
print("  SMART PRUNER FOR ISOMORPHIC SUBGRAPH")
print("=" * 60)

# ─── Load triplets ────────────────────────────────────────────
print("\nLoading triplets...")
submission = pd.read_csv('submission_raw.csv')
print(f"Input triplets: {len(submission):,}")

banc_list = submission['BANC'].astype(str).tolist()
fafb_list = submission['FAFB'].astype(str).tolist()
mcns_list = submission['MCNS'].astype(str).tolist()

# Create mapping: dataset_id → index in our triplet list
banc_to_idx = {b: i for i, b in enumerate(banc_list)}
fafb_to_idx = {f: i for i, f in enumerate(fafb_list)}
mcns_to_idx = {m: i for i, m in enumerate(mcns_list)}

# ─── Load challenge edge lists ────────────────────────────────
print("Loading edge lists...")
t = time.time()
fafb_df = pd.read_csv('fafb_783_edge_list.csv'); fafb_df.columns = ['src','tgt']
banc_df = pd.read_csv('banc_626_edge_list.csv'); banc_df.columns = ['src','tgt']
mcns_df = pd.read_csv('mcns_0.9_edge_list.csv'); mcns_df.columns = ['src','tgt']
print(f"Loaded in {time.time()-t:.1f}s")

# ─── Build internal edges (only between our matched neurons) ──
print("\nBuilding internal edge sets (edges between our matched neurons)...")
t = time.time()

banc_set = set(banc_to_idx.keys())
fafb_set = set(fafb_to_idx.keys())
mcns_set = set(mcns_to_idx.keys())

fafb_df_s = fafb_df.astype(str)
banc_df_s = banc_df.astype(str)
mcns_df_s = mcns_df.astype(str)

# Internal = both src AND tgt are in our matched set
banc_int = banc_df_s[banc_df_s['src'].isin(banc_set) & banc_df_s['tgt'].isin(banc_set)]
fafb_int = fafb_df_s[fafb_df_s['src'].isin(fafb_set) & fafb_df_s['tgt'].isin(fafb_set)]
mcns_int = mcns_df_s[mcns_df_s['src'].isin(mcns_set) & mcns_df_s['tgt'].isin(mcns_set)]

# Convert to index-based edges (use triplet index, not neuron ID)
banc_idx_edges = set(
    (banc_to_idx[s], banc_to_idx[t])
    for s, t in zip(banc_int['src'], banc_int['tgt'])
    if s in banc_to_idx and t in banc_to_idx
)
fafb_idx_edges = set(
    (fafb_to_idx[s], fafb_to_idx[t])
    for s, t in zip(fafb_int['src'], fafb_int['tgt'])
    if s in fafb_to_idx and t in fafb_to_idx
)
mcns_idx_edges = set(
    (mcns_to_idx[s], mcns_to_idx[t])
    for s, t in zip(mcns_int['src'], mcns_int['tgt'])
    if s in mcns_to_idx and t in mcns_to_idx
)

print(f"Internal edges — BANC: {len(banc_idx_edges):,} | FAFB: {len(fafb_idx_edges):,} | MCNS: {len(mcns_idx_edges):,}")
print(f"Built in {time.time()-t:.1f}s")

# ─── Diagnose violations ─────────────────────────────────────
print("\n=== VIOLATION DIAGNOSIS ===")
all_edges = banc_idx_edges | fafb_idx_edges | mcns_idx_edges

total_conflicts = 0
banc_extra = 0   # BANC has edge, FAFB/MCNS don't (3-4 synapse)
fafb_extra = 0   # FAFB has edge, BANC/MCNS don't
mcns_extra = 0   # MCNS has edge, BANC/FAFB don't

conflict_nodes = defaultdict(int)  # node_idx → number of conflicts

for (i, j) in all_edges:
    in_banc = (i, j) in banc_idx_edges
    in_fafb = (i, j) in fafb_idx_edges
    in_mcns = (i, j) in mcns_idx_edges
    if in_banc == in_fafb == in_mcns:
        continue  # all agree, no conflict
    total_conflicts += 1
    conflict_nodes[i] += 1
    conflict_nodes[j] += 1
    if in_banc and not in_fafb and not in_mcns:
        banc_extra += 1
    elif in_fafb and not in_banc and not in_mcns:
        fafb_extra += 1
    elif in_mcns and not in_banc and not in_fafb:
        mcns_extra += 1

print(f"Total conflicting edges:  {total_conflicts:,}")
print(f"  BANC-only (3-4 syn):    {banc_extra:,}  <- BANC threshold issue")
print(f"  FAFB-only:              {fafb_extra:,}")
print(f"  MCNS-only:              {mcns_extra:,}")
print(f"  Other (2-way disagree): {total_conflicts - banc_extra - fafb_extra - mcns_extra:,}")
print(f"Neurons involved in conflicts: {len(conflict_nodes):,} / {len(submission):,}")

# ─── Iterative greedy pruning ─────────────────────────────────
print("\n=== ITERATIVE PRUNING ===")
print("Removing most-conflicted neurons one at a time...")
t = time.time()

active = set(range(len(submission)))  # all neuron indices active
active_banc_edges = set(banc_idx_edges)
active_fafb_edges = set(fafb_idx_edges)
active_mcns_edges = set(mcns_idx_edges)

iteration = 0
removed = []

while True:
    # Find all currently conflicting edges
    all_active = active_banc_edges | active_fafb_edges | active_mcns_edges
    conflicts = defaultdict(int)
    total = 0
    for (i, j) in all_active:
        if i not in active or j not in active:
            continue
        in_b = (i,j) in active_banc_edges
        in_f = (i,j) in active_fafb_edges
        in_m = (i,j) in active_mcns_edges
        if not (in_b == in_f == in_m):
            total += 1
            conflicts[i] += 1
            conflicts[j] += 1

    if total == 0:
        print(f"\nConverged! 0 conflicts after removing {len(removed)} neurons")
        break

    # Remove neuron with most conflicts
    worst = max(conflicts, key=conflicts.get)
    active.discard(worst)

    # Remove all edges involving this neuron
    active_banc_edges = {(i,j) for i,j in active_banc_edges if i in active and j in active}
    active_fafb_edges = {(i,j) for i,j in active_fafb_edges if i in active and j in active}
    active_mcns_edges = {(i,j) for i,j in active_mcns_edges if i in active and j in active}

    removed.append(worst)
    iteration += 1

    if iteration % 50 == 0:
        print(f"  iter {iteration:4d}: {len(active):,} neurons remaining, {total:,} conflicts")
    if iteration > 2000:
        print("Stopped at 2000 iterations")
        break

elapsed = time.time() - t
print(f"Pruning took {elapsed:.1f}s")

# ─── Final result ─────────────────────────────────────────────
final = submission.iloc[sorted(active)].copy()
print(f"\n=== FINAL RESULT ===")
print(f"Final clean triplets:  {len(final):,}")
print(f"Neurons removed:       {len(removed):,}")
print(f"Retention rate:        {100*len(final)/len(submission):.1f}%")

# Save final submission
final.to_csv('submission_final.csv', index=False)
print(f"\nSaved to submission_final.csv")
print(final.head(10).to_string())
