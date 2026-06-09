import pandas as pd
import numpy as np
import time
from collections import defaultdict
import random
import multiprocessing

TRIPLETS_FILE = 'expanded_triplets.csv'

def worker(params):
    alpha, seed, edges_data = params
    np.random.seed(seed)
    random.seed(seed)
    
    banc_idx_edges_raw, fafb_idx_edges_raw, mcns_idx_edges_raw, num_filtered = edges_data
    
    # Create local copies for this process
    banc_idx_edges = set(banc_idx_edges_raw)
    fafb_idx_edges = set(fafb_idx_edges_raw)
    mcns_idx_edges = set(mcns_idx_edges_raw)
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

    active = set(range(num_filtered))
    
    while total_conflicts > 0:
        if alpha == float('inf'):
            # Pure greedy
            worst = max(conflicts, key=conflicts.get)
        else:
            # Probabilistic selection (Simulated Annealing heuristic)
            # Pick from the top 50 worst nodes to avoid calculating powers for 14,000 nodes every loop
            top_k = min(50, len(conflicts))
            # Sort is fast for small sets, but even faster: heapq.nlargest. 
            # We'll just do a quick sort
            items = sorted(conflicts.items(), key=lambda x: x[1], reverse=True)[:top_k]
            nodes = [x[0] for x in items]
            counts = np.array([x[1] for x in items], dtype=np.float64)
            
            # Probability proportional to (conflicts)^alpha
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
    print("Loading candidate triplets...")
    filtered = pd.read_csv(TRIPLETS_FILE, dtype=str)
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

    edges_data = (list(banc_idx_edges), list(fafb_idx_edges), list(mcns_idx_edges), len(filtered))

    # Grid search: Alpha (temperature equivalent) and Seed
    # alpha=inf is Greedy. alpha=1 is very random. alpha=5 is soft greedy.
    alphas = [2, 3, 5, 8, 10, float('inf')]
    seeds = [42, 100, 200, 300, 400]
    
    tasks = []
    for a in alphas:
        for s in seeds:
            if a == float('inf') and s != 42:
                continue # Pure greedy is deterministic, only run once
            tasks.append((a, s, edges_data))
            
    print(f"\nDispatching {len(tasks)} parallel Simulated Annealing permutations...")
    t_start = time.time()
    
    best_n = 0
    best_active = None
    best_params = None
    
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        for res in pool.imap_unordered(worker, tasks):
            a, s, final_n, active_list = res
            a_str = "Greedy" if a == float('inf') else f"Alpha={a}"
            print(f" -> [{a_str} | Seed={s}] Converged at N = {final_n:,}")
            if final_n > best_n:
                best_n = final_n
                best_active = active_list
                best_params = (a_str, s)

    print(f"\n==========================================")
    print(f"🏆 GLOBAL MAXIMUM FOUND: N = {best_n:,}")
    print(f"Parameters: {best_params[0]}, Seed={best_params[1]}")
    print(f"Total time for {len(tasks)} runs: {time.time()-t_start:.1f}s")
    print(f"==========================================")
    
    final_df = filtered.iloc[best_active].copy()
    final_df.to_csv('submission_ULTIMATE_SA.csv', index=False)
    print("Saved to submission_ULTIMATE_SA.csv!")
