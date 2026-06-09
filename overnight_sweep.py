"""
overnight_sweep.py
==================
Runs 300+ SA permutations across all CPU cores on hungarian_triplets.csv.
After finding the global best SA result, applies Grow-from-Core.

Current best to beat: N = 4,833 (SA=4,780 + Grow=53)
Target: Find if any SA config can beat N=4,780 before grow.

Alpha grid: [1.2, 1.5, 2, 3, 5, 8, 10, 15, 20, inf]
Seeds: 0-29 (30 per alpha)  = 300+ total runs
"""

import pandas as pd
import numpy as np
import time
from collections import defaultdict
import multiprocessing
import glob

TRIPLETS_FILE = 'hungarian_triplets.csv'
CURRENT_BEST = 4780  # SA best before grow

def sa_worker(params):
    alpha, seed, edges_data = params
    np.random.seed(seed)

    banc_raw, fafb_raw, mcns_raw, num_nodes = edges_data
    banc_idx_edges = set(map(tuple, banc_raw))
    fafb_idx_edges = set(map(tuple, fafb_raw))
    mcns_idx_edges = set(map(tuple, mcns_raw))
    all_active = banc_idx_edges | fafb_idx_edges | mcns_idx_edges

    conflicts = defaultdict(int)
    adj = defaultdict(list)
    total_conflicts = 0

    for (i, j) in all_active:
        adj[i].append((i, j))
        adj[j].append((i, j))
        in_b = (i, j) in banc_idx_edges
        in_f = (i, j) in fafb_idx_edges
        in_m = (i, j) in mcns_idx_edges
        if not (in_b == in_f == in_m):
            total_conflicts += 1
            conflicts[i] += 1
            conflicts[j] += 1

    active = set(range(num_nodes))

    while total_conflicts > 0:
        if alpha == float('inf') or len(conflicts) == 0:
            worst = max(conflicts, key=conflicts.get)
        else:
            top_k = min(50, len(conflicts))
            items = sorted(conflicts.items(), key=lambda x: x[1], reverse=True)[:top_k]
            nodes = [x[0] for x in items]
            counts = np.array([x[1] for x in items], dtype=np.float64)
            weights = counts ** alpha
            probs = weights / weights.sum()
            worst = int(np.random.choice(nodes, p=probs))

        active.discard(worst)
        for edge in adj[worst]:
            if edge in all_active:
                i, j = edge
                in_b = edge in banc_idx_edges
                in_f = edge in fafb_idx_edges
                in_m = edge in mcns_idx_edges
                was_conflict = not (in_b == in_f == in_m)
                if was_conflict:
                    total_conflicts -= 1
                    conflicts[i] -= 1
                    conflicts[j] -= 1
                all_active.discard(edge)
                banc_idx_edges.discard(edge)
                fafb_idx_edges.discard(edge)
                mcns_idx_edges.discard(edge)
        if worst in conflicts:
            del conflicts[worst]

    return (alpha, seed, len(active), sorted(list(active)))


def fast_grow(sa_df, all_df, banc_adj, fafb_adj, mcns_adj):
    """O(d) Grow-from-Core."""
    in_core = set(sa_df['BANC'].astype(str))
    candidates = all_df[~all_df['BANC'].astype(str).isin(in_core)].copy()
    banc_to_triplet = {str(b): (str(f), str(m))
                       for b, f, m in zip(sa_df['BANC'], sa_df['FAFB'], sa_df['MCNS'])}
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
    return added, banc_to_triplet


if __name__ == '__main__':
    print("=" * 60)
    print("  OVERNIGHT EXTENDED SA SWEEP")
    print(f"  Current best: SA={CURRENT_BEST}, After Grow=4,833")
    print("=" * 60)

    print(f"\nLoading {TRIPLETS_FILE}...")
    filtered = pd.read_csv(TRIPLETS_FILE, dtype=str)
    print(f"Pool: {len(filtered):,} triplets")

    print("Loading edge lists...")
    t0 = time.time()
    fafb_raw = pd.read_csv('fafb_783_edge_list.csv'); fafb_raw.columns = ['src', 'tgt']
    banc_raw = pd.read_csv('banc_626_edge_list.csv'); banc_raw.columns = ['src', 'tgt']
    mcns_raw = pd.read_csv('mcns_0.9_edge_list.csv'); mcns_raw.columns = ['src', 'tgt']
    print(f"Loaded in {time.time()-t0:.1f}s")

    banc_to_idx = {b: i for i, b in enumerate(filtered['BANC'].tolist())}
    fafb_to_idx = {f: i for i, f in enumerate(filtered['FAFB'].tolist())}
    mcns_to_idx = {m: i for i, m in enumerate(filtered['MCNS'].tolist())}

    def build_edges(df, a, b):
        df_s = df.astype(str)
        mask = df_s['src'].isin(a) & df_s['tgt'].isin(b)
        return [(a[s], b[t]) for s, t in zip(df_s[mask]['src'], df_s[mask]['tgt'])
                if s in a and t in b]

    print("Building index edges...")
    t0 = time.time()
    banc_edges = build_edges(banc_raw, banc_to_idx, banc_to_idx)
    fafb_edges = build_edges(fafb_raw, fafb_to_idx, fafb_to_idx)
    mcns_edges = build_edges(mcns_raw, mcns_to_idx, mcns_to_idx)
    print(f"Done in {time.time()-t0:.1f}s")
    print(f"Edges: FAFB={len(fafb_edges):,} BANC={len(banc_edges):,} MCNS={len(mcns_edges):,}")

    edges_data = (banc_edges, fafb_edges, mcns_edges, len(filtered))

    # Extended alpha/seed grid
    alphas = [1.2, 1.5, 2, 3, 5, 8, 10, 15, 20, float('inf')]
    seeds = list(range(30))  # 30 seeds per alpha

    tasks = []
    for a in alphas:
        for s in seeds:
            if a == float('inf') and s > 0:
                continue  # greedy is deterministic
            tasks.append((a, s, edges_data))

    ncores = multiprocessing.cpu_count()
    print(f"\nDispatching {len(tasks)} SA runs across {ncores} cores...")
    print("(This will take ~15-30 minutes)\n")
    t_start = time.time()

    best_n_sa = 0
    best_active = None
    best_params = None
    results_log = []

    with multiprocessing.Pool(processes=ncores) as pool:
        for res in pool.imap_unordered(sa_worker, tasks):
            a, s, final_n, active_list = res
            a_str = "Greedy" if a == float('inf') else f"α={a}"
            beat_str = f"  *** NEW BEST ***" if final_n > best_n_sa else ""
            print(f"  [{a_str:8s} | seed={s:2d}] N = {final_n:,}{beat_str}")
            results_log.append({'alpha': a_str, 'seed': s, 'N': final_n})
            if final_n > best_n_sa:
                best_n_sa = final_n
                best_active = active_list
                best_params = (a_str, s)

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"🏆 SA GLOBAL BEST: N = {best_n_sa:,}  ({best_params[0]}, seed={best_params[1]})")
    print(f"   Previous SA best: {CURRENT_BEST:,}")
    print(f"   SA improvement:  +{best_n_sa - CURRENT_BEST:,}")
    print(f"   Time: {elapsed:.0f}s for {len(tasks)} runs")
    print(f"{'='*60}")

    # Save results log
    pd.DataFrame(results_log).sort_values('N', ascending=False).to_csv('sa_sweep_results.csv', index=False)

    # Save best SA result
    sa_df = filtered.iloc[best_active].copy().reset_index(drop=True)
    sa_csv = f'submission_SWEEP_SA_{best_n_sa}.csv'
    sa_df.to_csv(sa_csv, index=False)
    print(f"SA result saved to {sa_csv}")

    # Now do Grow-from-Core
    print(f"\n{'='*60}")
    print("  GROW-FROM-CORE on sweep best")
    print(f"{'='*60}")

    print("Building adjacency sets for grow...")
    t0 = time.time()

    def build_adj(df):
        adj = defaultdict(set)
        for s, t in zip(df['src'].astype(str), df['tgt'].astype(str)):
            adj[s].add(t)
            adj[t].add(s)
        return adj

    banc_adj = build_adj(banc_raw)
    fafb_adj_d = build_adj(fafb_raw)
    mcns_adj_d = build_adj(mcns_raw)
    print(f"Adjacency sets built in {time.time()-t0:.1f}s")

    added, final_triplets = fast_grow(sa_df, filtered, banc_adj, fafb_adj_d, mcns_adj_d)
    final_n = best_n_sa + added

    print(f"\n✅ FINAL RESULT AFTER GROW:")
    print(f"   SA best:      {best_n_sa:,}")
    print(f"   Grow added:   {added:,}")
    print(f"   FINAL N:      {final_n:,}")
    print(f"   vs old best:  4,833  (improvement: +{final_n-4833:,})")

    rows = [{'BANC': b, 'FAFB': f, 'MCNS': m} for b, (f, m) in final_triplets.items()]
    final_df = pd.DataFrame(rows)
    assert final_df['BANC'].nunique() == len(final_df)
    assert final_df['FAFB'].nunique() == len(final_df)
    assert final_df['MCNS'].nunique() == len(final_df)

    out = f'submission_OVERNIGHT_{final_n}.csv'
    final_df.to_csv(out, index=False)
    print(f"   Saved to {out} ✅")

    print(f"\n{'='*60}")
    print(f"  ALL DONE! Final N = {final_n:,}")
    print(f"{'='*60}")
