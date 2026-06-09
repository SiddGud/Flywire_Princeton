import pandas as pd
import time
from collections import defaultdict

print("Loading candidate triplets...")
filtered = pd.read_csv('expanded_triplets.csv', dtype=str)
print(f"Candidates: {len(filtered)}")

print("\nLoading edge lists...")
t0 = time.time()
fafb_df = pd.read_csv('fafb_783_edge_list.csv'); fafb_df.columns = ['src','tgt']
banc_df = pd.read_csv('banc_626_edge_list.csv'); banc_df.columns = ['src','tgt']
mcns_df = pd.read_csv('mcns_0.9_edge_list.csv'); mcns_df.columns = ['src','tgt']
print(f"Loaded in {time.time()-t0:.1f}s")

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

banc_idx_edges = set((banc_to_idx[s], banc_to_idx[t]) for s,t in zip(banc_int['src'],banc_int['tgt']) if s in banc_to_idx and t in banc_to_idx)
fafb_idx_edges = set((fafb_to_idx[s], fafb_to_idx[t]) for s,t in zip(fafb_int['src'],fafb_int['tgt']) if s in fafb_to_idx and t in fafb_to_idx)
mcns_idx_edges = set((mcns_to_idx[s], mcns_to_idx[t]) for s,t in zip(mcns_int['src'],mcns_int['tgt']) if s in mcns_to_idx and t in mcns_to_idx)

print(f"Internal edges: FAFB {len(fafb_idx_edges)}, BANC {len(banc_idx_edges)}, MCNS {len(mcns_idx_edges)}")

print("\nStarting Greedy Pruning (O(E) update)...")
t = time.time()
active = set(range(len(filtered)))

iteration = 0
while True:
    all_active = banc_idx_edges | fafb_idx_edges | mcns_idx_edges
    conflicts = defaultdict(int)
    total = 0

    for (i,j) in all_active:
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
    banc_idx_edges = {(i,j) for i,j in banc_idx_edges if i != worst and j != worst}
    fafb_idx_edges = {(i,j) for i,j in fafb_idx_edges if i != worst and j != worst}
    mcns_idx_edges = {(i,j) for i,j in mcns_idx_edges if i != worst and j != worst}

    iteration += 1
    if iteration % 100 == 0:
        elapsed = time.time() - t
        print(f"  iter {iteration:4d}: {len(active):,} nodes remaining, {total:,} conflicts ({elapsed:.0f}s)")

final = filtered.iloc[sorted(active)].copy()
print(f"Final N = {len(final)}")
final.to_csv('submission_NBLAST_FINAL.csv', index=False)
