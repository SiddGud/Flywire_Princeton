"""
full_pipeline.py
================
Phase 1: Run SA on Hungarian-optimized bijection (14,551 nodes)
Phase 2: Run Grow-from-Core on the SA output
"""

import pandas as pd
import time
from collections import defaultdict
import numpy as np
import random
import multiprocessing

TRIPLETS_FILE = 'hungarian_triplets.csv'

def sa_worker(params):
    alpha, seed, edges_data = params
    np.random.seed(seed)
    random.seed(seed)

    banc_raw, fafb_raw, mcns_raw, num_nodes = edges_data
    banc_idx_edges = set(map(tuple, banc_raw))
    fafb_idx_edges = set(map(tuple, fafb_raw))
    mcns_idx_edges = set(map(tuple, mcns_raw))
    all_active = banc_idx_edges | fafb_idx_edges | mcns_idx_edges

    conflicts = defaultdict(int)
    adj = defaultdict(list)
    total_conflicts = 0

    for (i,j) in all_active:
        adj[i].append((i,j))
        adj[j].append((i,j))
        in_b = (i,j) in banc_idx_edges
        in_f = (i,j) in fafb_idx_edges
        in_m = (i,j) in mcns_idx_edges
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
            worst = np.random.choice(nodes, p=probs)

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


if __name__ == '__main__':
    print("=" * 60)
    print("  FULL PIPELINE: HUNGARIAN + SA + GROW")
    print("=" * 60)

    print(f"\nLoading Hungarian triplets from {TRIPLETS_FILE}...")
    filtered = pd.read_csv(TRIPLETS_FILE, dtype=str)
    print(f"Starting pool: {len(filtered):,} triplets")

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

    def build_edges(df, to_idx_src, to_idx_tgt):
        df_s = df.astype(str)
        mask = df_s['src'].isin(to_idx_src) & df_s['tgt'].isin(to_idx_tgt)
        df_int = df_s[mask]
        return [(to_idx_src[s], to_idx_tgt[t]) for s, t in zip(df_int['src'], df_int['tgt'])
                if s in to_idx_src and t in to_idx_tgt]

    banc_edges = build_edges(banc_df, banc_to_idx, banc_to_idx)
    fafb_edges = build_edges(fafb_df, fafb_to_idx, fafb_to_idx)
    mcns_edges = build_edges(mcns_df, mcns_to_idx, mcns_to_idx)
    print(f"Internal edges: FAFB {len(fafb_edges)}, BANC {len(banc_edges)}, MCNS {len(mcns_edges)}")

    edges_data = (banc_edges, fafb_edges, mcns_edges, len(filtered))

    # Comprehensive SA grid: wider search than before
    alphas = [1.5, 2, 3, 5, 8, 10, float('inf')]
    seeds = list(range(0, 10))  # 10 seeds per alpha = 70 total runs

    tasks = []
    for a in alphas:
        for s in seeds:
            if a == float('inf') and s != 0:
                continue  # pure greedy is deterministic
            tasks.append((a, s, edges_data))

    print(f"\nDispatching {len(tasks)} SA permutations across {multiprocessing.cpu_count()} cores...")
    t_start = time.time()

    best_n = 0
    best_active = None
    best_params = None

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        for res in pool.imap_unordered(sa_worker, tasks):
            a, s, final_n, active_list = res
            a_str = "Greedy" if a == float('inf') else f"α={a}"
            print(f"  [{a_str:8s} | seed={s:2d}] N = {final_n:,}")
            if final_n > best_n:
                best_n = final_n
                best_active = active_list
                best_params = (a_str, s)

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"🏆 SA BEST: N = {best_n:,}  (params: {best_params[0]}, seed={best_params[1]})")
    print(f"   vs old baseline: 4,583")
    print(f"   Improvement: +{best_n - 4583:,}")
    print(f"   Time: {elapsed:.1f}s for {len(tasks)} runs")
    print(f"{'='*60}")

    sa_df = filtered.iloc[best_active].copy().reset_index(drop=True)

    # ── Phase 2: Grow-from-Core ──────────────────────────────────
    print(f"\n{'='*60}")
    print("  PHASE 2: GROW-FROM-CORE")
    print(f"{'='*60}")
    print(f"Starting from SA core of {len(sa_df):,} nodes...")

    # Build perfect edge sets from SA result
    core_banc = set(sa_df['BANC'].tolist())
    core_fafb = set(sa_df['FAFB'].tolist())
    core_mcns = set(sa_df['MCNS'].tolist())

    core_b2i = {b: i for i, b in enumerate(sa_df['BANC'].tolist())}
    core_f2i = {f: i for i, f in enumerate(sa_df['FAFB'].tolist())}
    core_m2i = {m: i for i, m in enumerate(sa_df['MCNS'].tolist())}

    core_banc_e = set((core_b2i[s], core_b2i[t]) for s,t in zip(banc_df['src'].astype(str), banc_df['tgt'].astype(str))
                      if s in core_b2i and t in core_b2i)
    core_fafb_e = set((core_f2i[s], core_f2i[t]) for s,t in zip(fafb_df['src'].astype(str), fafb_df['tgt'].astype(str))
                      if s in core_f2i and t in core_f2i)
    core_mcns_e = set((core_m2i[s], core_m2i[t]) for s,t in zip(mcns_df['src'].astype(str), mcns_df['tgt'].astype(str))
                      if s in core_m2i and t in core_m2i)

    # Get candidates NOT in the SA core
    in_core_mask = filtered['BANC'].isin(core_banc)
    candidates = filtered[~in_core_mask].copy()
    print(f"Candidates to try adding: {len(candidates):,}")

    added = 0
    for _, row in candidates.iterrows():
        b, f, m = str(row['BANC']), str(row['FAFB']), str(row['MCNS'])

        new_i = len(sa_df)
        # Check if adding this node creates any violations
        # Need to check all edges from this new node to existing core nodes
        violation = False

        # Get neighbors of b in banc edge list
        banc_neighbors_b = set(banc_df[banc_df['src'].astype(str) == b]['tgt'].astype(str)) | \
                           set(banc_df[banc_df['tgt'].astype(str) == b]['src'].astype(str))
        fafb_neighbors_f = set(fafb_df[fafb_df['src'].astype(str) == f]['tgt'].astype(str)) | \
                           set(fafb_df[fafb_df['tgt'].astype(str) == f]['src'].astype(str))
        mcns_neighbors_m = set(mcns_df[mcns_df['src'].astype(str) == m]['tgt'].astype(str)) | \
                           set(mcns_df[mcns_df['tgt'].astype(str) == m]['src'].astype(str))

        # For each existing node in core, check edge consistency
        for j_idx, core_row in sa_df.iterrows():
            cb, cf, cm = str(core_row['BANC']), str(core_row['FAFB']), str(core_row['MCNS'])
            edge_b = cb in banc_neighbors_b
            edge_f = cf in fafb_neighbors_f
            edge_m = cm in mcns_neighbors_m
            if not (edge_b == edge_f == edge_m):
                violation = True
                break

        if not violation:
            sa_df = pd.concat([sa_df, pd.DataFrame([row])], ignore_index=True)
            added += 1

    print(f"Grow-from-Core added: {added:,} neurons")
    print(f"Final N = {len(sa_df):,}")

    out_name = f'submission_HUNGARIAN_SA_{len(sa_df)}.csv'
    sa_df.to_csv(out_name, index=False)
    print(f"\n✅ Saved to {out_name}")
    print(f"\n{'='*60}")
    print(f"  FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"  Previous best (random bijection + SA): 4,583")
    print(f"  New SA result (Hungarian bijection):   {best_n:,}")
    print(f"  After Grow-from-Core:                  {len(sa_df):,}")
    print(f"  Total improvement:                    +{len(sa_df)-4583:,}")
