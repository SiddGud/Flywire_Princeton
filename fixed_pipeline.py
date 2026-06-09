"""
fixed_pipeline.py
=================
Fixes the directed-edge bug in Grow-from-Core.

The bug: build_adj added both adj[s].add(t) AND adj[t].add(s),
making it undirected. Directed edges A->B != B->A.

Fix: Use DIRECTED edge sets (tuples) for O(1) lookup.
     Check BOTH i->j AND j->i separately for each pair.

Runs the best SA from the overnight sweep (α=1.5, seed=24 → N=4,784)
then applies the corrected directed grow.
"""

import pandas as pd
import numpy as np
import time
from collections import defaultdict
import multiprocessing

TRIPLETS_FILE = 'hungarian_triplets.csv'

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
        adj[i].append((i, j)); adj[j].append((i, j))
        ib, iff, im = (i,j) in banc_idx_edges, (i,j) in fafb_idx_edges, (i,j) in mcns_idx_edges
        if not (ib == iff == im):
            total_conflicts += 1; conflicts[i] += 1; conflicts[j] += 1

    active = set(range(num_nodes))
    while total_conflicts > 0:
        if alpha == float('inf') or not conflicts:
            worst = max(conflicts, key=conflicts.get)
        else:
            top_k = min(50, len(conflicts))
            items = sorted(conflicts.items(), key=lambda x: x[1], reverse=True)[:top_k]
            nodes = [x[0] for x in items]
            counts = np.array([x[1] for x in items], dtype=np.float64)
            weights = counts ** alpha; probs = weights / weights.sum()
            worst = int(np.random.choice(nodes, p=probs))
        active.discard(worst)
        for edge in adj[worst]:
            if edge in all_active:
                i, j = edge
                ib, iff, im = edge in banc_idx_edges, edge in fafb_idx_edges, edge in mcns_idx_edges
                if not (ib == iff == im):
                    total_conflicts -= 1; conflicts[i] -= 1; conflicts[j] -= 1
                all_active.discard(edge); banc_idx_edges.discard(edge)
                fafb_idx_edges.discard(edge); mcns_idx_edges.discard(edge)
        if worst in conflicts: del conflicts[worst]
    return (alpha, seed, len(active), sorted(list(active)))


def directed_grow(sa_df, all_df, banc_edge_set, fafb_edge_set, mcns_edge_set):
    """
    FIXED directed grow: checks A->B and B->A separately using edge tuples.
    banc_edge_set = set of (src_str, tgt_str) directed tuples from BANC.
    """
    in_core = set(sa_df['BANC'].astype(str))
    candidates = all_df[~all_df['BANC'].astype(str).isin(in_core)].copy()

    # Core lookup: BANC_id -> (FAFB_id, MCNS_id)
    banc_to_triplet = {str(b): (str(f), str(m))
                       for b, f, m in zip(sa_df['BANC'], sa_df['FAFB'], sa_df['MCNS'])}

    added = 0
    for _, row in candidates.iterrows():
        b, f, m = str(row['BANC']), str(row['FAFB']), str(row['MCNS'])
        violation = False

        for cb, (cf, cm) in banc_to_triplet.items():
            # Check DIRECTED edge new->core: b->cb, f->cf, m->cm
            e_b_fwd = (b, cb) in banc_edge_set
            e_f_fwd = (f, cf) in fafb_edge_set
            e_m_fwd = (m, cm) in mcns_edge_set
            if not (e_b_fwd == e_f_fwd == e_m_fwd):
                violation = True; break

            # Check DIRECTED edge core->new: cb->b, cf->f, cm->m
            e_b_rev = (cb, b) in banc_edge_set
            e_f_rev = (cf, f) in fafb_edge_set
            e_m_rev = (cm, m) in mcns_edge_set
            if not (e_b_rev == e_f_rev == e_m_rev):
                violation = True; break

        if not violation:
            banc_to_triplet[b] = (f, m)
            added += 1
            if added % 10 == 0:
                print(f"  Grow: +{added}, core = {len(sa_df)+added:,}")

    return added, banc_to_triplet


if __name__ == '__main__':
    print("=" * 60)
    print("  FIXED PIPELINE (directed grow)")
    print("=" * 60)

    print(f"\nLoading {TRIPLETS_FILE}...")
    filtered = pd.read_csv(TRIPLETS_FILE, dtype=str)
    print(f"Pool: {len(filtered):,}")

    print("Loading edge lists...")
    t0 = time.time()
    fafb_raw = pd.read_csv('fafb_783_edge_list.csv'); fafb_raw.columns = ['src','tgt']
    banc_raw = pd.read_csv('banc_626_edge_list.csv'); banc_raw.columns = ['src','tgt']
    mcns_raw = pd.read_csv('mcns_0.9_edge_list.csv'); mcns_raw.columns = ['src','tgt']
    print(f"Loaded in {time.time()-t0:.1f}s")

    # Build directed string edge sets for the grow phase
    print("Building directed string edge sets for grow...")
    t0 = time.time()
    banc_str_edges = set(zip(banc_raw['src'].astype(str), banc_raw['tgt'].astype(str)))
    fafb_str_edges = set(zip(fafb_raw['src'].astype(str), fafb_raw['tgt'].astype(str)))
    mcns_str_edges = set(zip(mcns_raw['src'].astype(str), mcns_raw['tgt'].astype(str)))
    print(f"Done in {time.time()-t0:.1f}s")

    # Build index edges for SA
    banc_to_idx = {b: i for i,b in enumerate(filtered['BANC'].tolist())}
    fafb_to_idx = {f: i for i,f in enumerate(filtered['FAFB'].tolist())}
    mcns_to_idx = {m: i for i,m in enumerate(filtered['MCNS'].tolist())}

    def build_edges(df, a, b):
        df_s = df.astype(str)
        mask = df_s['src'].isin(a) & df_s['tgt'].isin(b)
        return [(a[s], b[t]) for s,t in zip(df_s[mask]['src'], df_s[mask]['tgt']) if s in a and t in b]

    banc_edges = build_edges(banc_raw, banc_to_idx, banc_to_idx)
    fafb_edges = build_edges(fafb_raw, fafb_to_idx, fafb_to_idx)
    mcns_edges = build_edges(mcns_raw, mcns_to_idx, mcns_to_idx)
    edges_data = (banc_edges, fafb_edges, mcns_edges, len(filtered))
    print(f"Edges: FAFB={len(fafb_edges):,} BANC={len(banc_edges):,} MCNS={len(mcns_edges):,}")

    # Comprehensive SA grid
    alphas = [1.2, 1.5, 2, 3, 5, 8, 10, 15, 20, float('inf')]
    seeds = list(range(30))
    tasks = []
    for a in alphas:
        for s in seeds:
            if a == float('inf') and s > 0: continue
            tasks.append((a, s, edges_data))

    ncores = multiprocessing.cpu_count()
    print(f"\nDispatching {len(tasks)} SA runs across {ncores} cores...")
    t_start = time.time()

    best_n_sa = 0; best_active = None; best_params = None; results_log = []

    with multiprocessing.Pool(processes=ncores) as pool:
        for res in pool.imap_unordered(sa_worker, tasks):
            a, s, final_n, active_list = res
            a_str = "Greedy" if a == float('inf') else f"α={a}"
            beat = "  *** NEW BEST ***" if final_n > best_n_sa else ""
            print(f"  [{a_str:8s} | seed={s:2d}] N = {final_n:,}{beat}")
            results_log.append({'alpha': a_str, 'seed': s, 'N': final_n})
            if final_n > best_n_sa:
                best_n_sa = final_n; best_active = active_list; best_params = (a_str, s)

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"🏆 SA BEST: N = {best_n_sa:,}  ({best_params[0]}, seed={best_params[1]})")
    print(f"   Time: {elapsed:.0f}s for {len(tasks)} runs")
    pd.DataFrame(results_log).sort_values('N', ascending=False).to_csv('sa_sweep_results.csv', index=False)

    sa_df = filtered.iloc[best_active].copy().reset_index(drop=True)
    sa_df.to_csv(f'submission_SA_{best_n_sa}.csv', index=False)
    print(f"SA saved to submission_SA_{best_n_sa}.csv")

    # FIXED Grow-from-Core
    print(f"\n{'='*60}")
    print("  FIXED DIRECTED GROW-FROM-CORE")
    print(f"{'='*60}")
    print(f"Starting from {best_n_sa:,} node SA core...")
    print("Building adjacency sets...")

    t0 = time.time()
    added, final_triplets = directed_grow(sa_df, filtered, banc_str_edges, fafb_str_edges, mcns_str_edges)
    final_n = best_n_sa + added

    print(f"\n{'='*60}")
    print(f"✅ FINAL RESULT")
    print(f"   SA best:    {best_n_sa:,}")
    print(f"   Grow added: {added:,}")
    print(f"   FINAL N:    {final_n:,}")
    print(f"   Grow time:  {time.time()-t0:.1f}s")
    print(f"{'='*60}")

    rows = [{'BANC': b, 'FAFB': f, 'MCNS': m} for b,(f,m) in final_triplets.items()]
    final_df = pd.DataFrame(rows)
    assert final_df['BANC'].nunique() == len(final_df)
    assert final_df['FAFB'].nunique() == len(final_df)
    assert final_df['MCNS'].nunique() == len(final_df)

    out = f'submission_FIXED_{final_n}.csv'
    final_df.to_csv(out, index=False)
    print(f"Saved to {out} ✅")
