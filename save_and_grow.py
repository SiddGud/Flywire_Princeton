"""
save_and_grow.py
================
1. Re-run the best SA config (α=8, seed=9) on hungarian_triplets.csv
   to regenerate the 4,780-node core CSV.
2. Run fast Grow-from-Core on that result.
"""

import pandas as pd
import numpy as np
import time
from collections import defaultdict
import glob

# ── Step 1: Check if we already have the 4780 CSV ──────────────
sa_files = sorted(glob.glob('submission_HUNGARIAN_SA_*.csv'))
print(f"Existing SA files: {sa_files}")

if sa_files:
    best_file = max(sa_files, key=lambda f: int(f.split('_')[-1].replace('.csv','')))
    sa_df = pd.read_csv(best_file, dtype=str)
    print(f"Loaded existing SA result: {len(sa_df):,} nodes from {best_file}")
else:
    # Re-run best SA config
    print("No SA file found. Re-running best SA (α=8, seed=9) on hungarian_triplets.csv...")
    np.random.seed(9)

    filtered = pd.read_csv('hungarian_triplets.csv', dtype=str)
    fafb_df = pd.read_csv('fafb_783_edge_list.csv'); fafb_df.columns = ['src','tgt']
    banc_df = pd.read_csv('banc_626_edge_list.csv'); banc_df.columns = ['src','tgt']
    mcns_df = pd.read_csv('mcns_0.9_edge_list.csv'); mcns_df.columns = ['src','tgt']

    banc_to_idx = {b: i for i,b in enumerate(filtered['BANC'].tolist())}
    fafb_to_idx = {f: i for i,f in enumerate(filtered['FAFB'].tolist())}
    mcns_to_idx = {m: i for i,m in enumerate(filtered['MCNS'].tolist())}

    def build_edges(df, a, b):
        df_s = df.astype(str)
        mask = df_s['src'].isin(a) & df_s['tgt'].isin(b)
        return set((a[s], b[t]) for s,t in zip(df_s[mask]['src'], df_s[mask]['tgt']) if s in a and t in b)

    banc_edges = build_edges(banc_df, banc_to_idx, banc_to_idx)
    fafb_edges = build_edges(fafb_df, fafb_to_idx, fafb_to_idx)
    mcns_edges = build_edges(mcns_df, mcns_to_idx, mcns_to_idx)
    all_active = banc_edges | fafb_edges | mcns_edges

    conflicts = defaultdict(int)
    adj = defaultdict(list)
    total = 0
    for (i,j) in all_active:
        adj[i].append((i,j)); adj[j].append((i,j))
        ib, if_, im = (i,j) in banc_edges, (i,j) in fafb_edges, (i,j) in mcns_edges
        if not (ib == if_ == im):
            total += 1; conflicts[i] += 1; conflicts[j] += 1

    active = set(range(len(filtered)))
    t0 = time.time()
    itr = 0
    while total > 0:
        # SA with α=8
        top_k = min(50, len(conflicts))
        items = sorted(conflicts.items(), key=lambda x: x[1], reverse=True)[:top_k]
        nodes = [x[0] for x in items]
        counts = np.array([x[1] for x in items], dtype=np.float64)
        weights = counts ** 8
        probs = weights / weights.sum()
        worst = np.random.choice(nodes, p=probs)

        active.discard(worst)
        for e in adj[worst]:
            if e in all_active:
                ib, if_, im = e in banc_edges, e in fafb_edges, e in mcns_edges
                if not (ib == if_ == im):
                    total -= 1; conflicts[e[0]] -= 1; conflicts[e[1]] -= 1
                all_active.discard(e); banc_edges.discard(e); fafb_edges.discard(e); mcns_edges.discard(e)
        if worst in conflicts: del conflicts[worst]
        itr += 1
        if itr % 1000 == 0:
            print(f"  iter {itr}: {len(active):,} nodes, {total:,} conflicts ({time.time()-t0:.1f}s)")

    sa_df = filtered.iloc[sorted(active)].copy().reset_index(drop=True)
    print(f"\n✅ SA done: N = {len(sa_df):,}")
    sa_df.to_csv(f'submission_HUNGARIAN_SA_{len(sa_df)}.csv', index=False)

# ── Step 2: Fast Grow-from-Core ────────────────────────────────
print(f"\n{'='*60}")
print(f"  FAST GROW-FROM-CORE")
print(f"{'='*60}")
print(f"SA core: {len(sa_df):,} nodes")

all_df = pd.read_csv('hungarian_triplets.csv', dtype=str)
in_core = set(sa_df['BANC'].astype(str))
candidates = all_df[~all_df['BANC'].astype(str).isin(in_core)].copy()
print(f"Candidates to try: {len(candidates):,}")

# Load edge lists and build fast adjacency sets
print("\nBuilding adjacency sets...")
t0 = time.time()

fafb_df = pd.read_csv('fafb_783_edge_list.csv'); fafb_df.columns = ['src','tgt']
banc_df = pd.read_csv('banc_626_edge_list.csv'); banc_df.columns = ['src','tgt']
mcns_df = pd.read_csv('mcns_0.9_edge_list.csv'); mcns_df.columns = ['src','tgt']
fafb_df = fafb_df.astype(str)
banc_df = banc_df.astype(str)
mcns_df = mcns_df.astype(str)

# Build directed adjacency sets (both directions for undirected check)
def build_adj(df):
    adj = defaultdict(set)
    for s, t in zip(df['src'], df['tgt']):
        adj[s].add(t)
        adj[t].add(s)
    return adj

banc_adj = build_adj(banc_df)
fafb_adj = build_adj(fafb_df)
mcns_adj = build_adj(mcns_df)
print(f"Built adjacency sets in {time.time()-t0:.1f}s")

# Build core lookup: BANC -> (FAFB, MCNS)
banc_to_triplet = {str(b): (str(f), str(m))
                   for b, f, m in zip(sa_df['BANC'], sa_df['FAFB'], sa_df['MCNS'])}

print(f"\nGrowing...")
t0 = time.time()
added = 0

for _, row in candidates.iterrows():
    b, f, m = str(row['BANC']), str(row['FAFB']), str(row['MCNS'])

    b_nbrs = banc_adj.get(b, set())
    f_nbrs = fafb_adj.get(f, set())
    m_nbrs = mcns_adj.get(m, set())

    violation = False
    for cb, (cf, cm) in banc_to_triplet.items():
        e_b = cb in b_nbrs
        e_f = cf in f_nbrs
        e_m = cm in m_nbrs
        if e_b != e_f or e_b != e_m:
            violation = True
            break

    if not violation:
        banc_to_triplet[b] = (f, m)
        added += 1
        if added % 10 == 0:
            print(f"  Added {added}, core now {len(sa_df)+added:,} ({time.time()-t0:.1f}s)")

final_n = len(sa_df) + added
print(f"\n✅ GROW COMPLETE!")
print(f"   SA core:  {len(sa_df):,}")
print(f"   Added:    {added:,}")
print(f"   Final N:  {final_n:,}")
print(f"   Time:     {time.time()-t0:.1f}s")

rows = [{'BANC': b, 'FAFB': f, 'MCNS': m} for b, (f,m) in banc_to_triplet.items()]
final_df = pd.DataFrame(rows)
assert final_df['BANC'].nunique() == len(final_df)
assert final_df['FAFB'].nunique() == len(final_df)
assert final_df['MCNS'].nunique() == len(final_df)

out = f'submission_FINAL_{final_n}.csv'
final_df.to_csv(out, index=False)
print(f"   Saved to {out} ✅")
