"""
CONNECTED GROW — HIGH-DEGREE SEED STRATEGY
==========================================
Fresh start: no prior submission needed.

Strategy: The highest-degree nodes in BANC are the most connected.
If such a hub node has an isomorphic match in FAFB and MCNS,
it will be the center of a large connected isomorphic subgraph.

Steps:
1. Find top-2000 highest-degree BANC nodes
2. For each, use signature grow to find matching FAFB + MCNS
3. Race all in parallel, keep the biggest connected result
4. Perturb from winner
"""
import pandas as pd
import numpy as np
import time, os
from collections import defaultdict, deque
from multiprocessing import Pool, cpu_count

# Find the dataset directory automatically
DATA1 = None
for root, dirs, files in os.walk('/kaggle/input'):
    if 'fafb_783_edge_list.csv' in files:
        DATA1 = root
        break

if DATA1 is None:
    print("ERROR: Could not find the dataset files! Please add the dataset to this notebook.")
    import sys; sys.exit(1)

print("Loading edge lists...", flush=True)
fafb_df = pd.read_csv(f'{DATA1}/fafb_783_edge_list.csv', header=None, names=['src','tgt'], dtype=str)
banc_df = pd.read_csv(f'{DATA1}/banc_626_edge_list.csv', header=None, names=['src','tgt'], dtype=str)
mcns_df = pd.read_csv(f'{DATA1}/mcns_0.9_edge_list.csv', header=None, names=['src','tgt'], dtype=str)

print("Pre-building edge sets...", flush=True)
banc_set = set(zip(banc_df['src'], banc_df['tgt']))
fafb_set = set(zip(fafb_df['src'], fafb_df['tgt']))
mcns_set = set(zip(mcns_df['src'], mcns_df['tgt']))
banc_undirected = defaultdict(set)
for s, t in banc_set:
    banc_undirected[s].add(t); banc_undirected[t].add(s)

# Compute BANC degree
print("Computing degrees...", flush=True)
banc_degree = defaultdict(int)
for s, t in banc_set:
    banc_degree[s] += 1; banc_degree[t] += 1
fafb_degree = defaultdict(int)
for s, t in fafb_set:
    fafb_degree[s] += 1; fafb_degree[t] += 1
mcns_degree = defaultdict(int)
for s, t in mcns_set:
    mcns_degree[s] += 1; mcns_degree[t] += 1
print("Done.\n", flush=True)

def compute_signatures(df, core_map):
    sig_in  = defaultdict(list)
    sig_out = defaultdict(list)
    mask_out = df['src'].isin(core_map) & (~df['tgt'].isin(core_map))
    for s, t in zip(df[mask_out]['src'], df[mask_out]['tgt']):
        sig_in[t].append(core_map[s])
    mask_in = (~df['src'].isin(core_map)) & df['tgt'].isin(core_map)
    for s, t in zip(df[mask_in]['src'], df[mask_in]['tgt']):
        sig_out[s].append(core_map[t])
    all_neighbors = set(sig_in.keys()) | set(sig_out.keys())
    signatures = defaultdict(list)
    for n in all_neighbors:
        key = (tuple(sorted(sig_in[n])), tuple(sorted(sig_out[n])))
        signatures[key].append(n)
    return signatures

def strict_grow_shuffled(candidates_df, core, rng):
    candidates_df = candidates_df.sample(frac=1, random_state=rng).reset_index(drop=True)
    added = 0
    for _, row in candidates_df.iterrows():
        b, f, m = str(row['BANC']), str(row['FAFB']), str(row['MCNS'])
        if b in core: continue
        ok = True
        for cb, (cf, cm) in list(core.items()):
            b_has  = (b, cb) in banc_set;  f_has  = (f, cf) in fafb_set;  m_has  = (m, cm) in mcns_set
            if b_has != f_has or b_has != m_has: ok = False; break
            b_has2 = (cb, b) in banc_set;  f_has2 = (cf, f) in fafb_set;  m_has2 = (cm, m) in mcns_set
            if b_has2 != f_has2 or b_has2 != m_has2: ok = False; break
        if ok:
            core[b] = (f, m); added += 1
    return added

def extract_lwcc(core):
    b_nodes = set(core.keys())
    visited, components = set(), []
    for node in b_nodes:
        if node not in visited:
            comp, q = [], deque([node])
            visited.add(node)
            while q:
                c = q.popleft(); comp.append(c)
                for nb in banc_undirected[c]:
                    if nb in b_nodes and nb not in visited:
                        visited.add(nb); q.append(nb)
            components.append(comp)
    if not components: return {}
    lwcc_nodes = set(max(components, key=len))
    return {b: v for b, v in core.items() if b in lwcc_nodes}

def run_grow(starting_core, seed_num):
    core = dict(starting_core)
    rng = np.random.RandomState(seed_num)
    for _ in range(25):
        b2i = {b: i for i, b in enumerate(core.keys())}
        f2i = {f: i for i, (f, m) in enumerate(core.values())}
        m2i = {m: i for i, (f, m) in enumerate(core.values())}
        b_sigs = compute_signatures(banc_df, b2i)
        f_sigs = compute_signatures(fafb_df, f2i)
        m_sigs = compute_signatures(mcns_df, m2i)
        shared = set(b_sigs.keys()) & set(f_sigs.keys()) & set(m_sigs.keys())
        cands = []
        for sig in shared:
            bn, fn, mn = b_sigs[sig], f_sigs[sig], m_sigs[sig]
            rng.shuffle(bn); rng.shuffle(fn); rng.shuffle(mn)
            for i in range(min(len(bn), len(fn), len(mn))):
                if bn[i] not in core:
                    cands.append({'BANC': bn[i], 'FAFB': fn[i], 'MCNS': mn[i]})
        if not cands: break
        if strict_grow_shuffled(pd.DataFrame(cands), core, rng) == 0: break
    return extract_lwcc(core)

def smart_perturb(core, frac, seed):
    rng = np.random.RandomState(seed)
    keys = list(core.keys())
    n_remove = max(1, int(len(keys) * frac))
    b_nodes = set(keys)
    degree = defaultdict(int)
    for s, t in zip(banc_df['src'], banc_df['tgt']):
        if s in b_nodes and t in b_nodes:
            degree[s] += 1; degree[t] += 1
    sorted_keys = sorted(keys, key=lambda b: degree.get(b, 0))
    pool = sorted_keys[:min(len(sorted_keys), n_remove * 3)]
    to_remove = set(rng.choice(pool, size=min(n_remove, len(pool)), replace=False))
    return {b: v for b, v in core.items() if b not in to_remove}, n_remove

def worker_seed(args):
    """Try each (B,F,M) triplet as a single-node seed and grow."""
    b, f, m, grow_seed = args
    seed_core = {b: (f, m)}
    return run_grow(seed_core, grow_seed)

def worker_perturb(args):
    core, frac, perturb_seed, grow_seed = args
    perturbed, n_remove = smart_perturb(core, frac, perturb_seed)
    grown = run_grow(perturbed, grow_seed)
    return grown, n_remove, frac

if __name__ == '__main__':
    print("=" * 65)
    print("  HIGH-DEGREE SEED STRATEGY — NOTEBOOK 2")
    print("=" * 65)

    # Top high-degree BANC nodes as potential seeds
    top_banc = sorted(banc_degree.keys(), key=lambda x: banc_degree[x], reverse=True)[:2000]
    top_fafb = sorted(fafb_degree.keys(), key=lambda x: fafb_degree[x], reverse=True)[:2000]
    top_mcns = sorted(mcns_degree.keys(), key=lambda x: mcns_degree[x], reverse=True)[:2000]

    print(f"Top BANC degrees: {[banc_degree[b] for b in top_banc[:5]]}")
    print(f"Top FAFB degrees: {[fafb_degree[f] for f in top_fafb[:5]]}")
    print(f"Top MCNS degrees: {[mcns_degree[m] for m in top_mcns[:5]]}")

    # Build seed triplets: pair top-K BANC with top-K FAFB and top-K MCNS
    # Try 2000 triplets: (top_banc[i], top_fafb[i], top_mcns[i]) for i in 0..1999
    # Also try shuffled pairings for diversity
    rng_main = np.random.RandomState(42)
    tasks = []
    for i in range(min(2000, len(top_banc))):
        # Paired by rank
        b = top_banc[i]
        f = top_fafb[i % len(top_fafb)]
        m = top_mcns[i % len(top_mcns)]
        grow_seed = i * 277 + 13
        tasks.append((b, f, m, grow_seed))

    N_WORKERS = min(4, cpu_count())
    print(f"\nPhase 1: Racing {len(tasks)} high-degree seeds with {N_WORKERS} workers...")
    t0 = time.time()
    
    results = []
    best_n = 0
    best_core = {}

    with Pool(N_WORKERS) as pool:
        for i, res_core in enumerate(pool.imap_unordered(worker_seed, tasks)):
            results.append(res_core)
            n = len(res_core)
            
            if n > best_n:
                best_n = n
                best_core = res_core
                print(f"  [{i+1}/{len(tasks)}] *** NEW BEST SEED: {best_n:,} nodes! ***", flush=True)
                tmp_df = pd.DataFrame([{'BANC': b, 'FAFB': f, 'MCNS': m} for b, (f, m) in best_core.items()])
                tmp_df.to_csv(f'/kaggle/working/submission_NB2_PHASE1_{best_n}.csv', index=False)
            elif (i + 1) % 50 == 0:
                print(f"  [{i+1}/{len(tasks)}] seeds processed...", flush=True)

    print(f"Phase 1 done in {time.time()-t0:.0f}s")

    global_best_n = best_n
    global_best_core = dict(best_core)

    tmp_df = pd.DataFrame([{'BANC': b, 'FAFB': f, 'MCNS': m}
                            for b, (f, m) in global_best_core.items()])
    tmp_df.to_csv(f'/kaggle/working/submission_NB2_PHASE1_{global_best_n}.csv', index=False)

    # Phase 2: parallel perturb from winner
    FRACS = [0.03, 0.05, 0.07, 0.08]
    MAX_NO_IMPROVE = 20
    no_improve, round_num = 0, 0
    print(f"\nPhase 2: Parallel perturbation from {global_best_n:,}-node winner...\n")

    with Pool(N_WORKERS) as pool:
        while no_improve < MAX_NO_IMPROVE:
            round_num += 1
            t_start = time.time()
            tasks = []
            for i, frac in enumerate(FRACS):
                tasks.append((best_core, frac, round_num*1000 + i*173 + 31, round_num*1000 + i*277 + 13))

            results = pool.map(worker_perturb, tasks)
            elapsed = time.time() - t_start

            best_res = max(results, key=lambda r: len(r[0]))
            core, n_remove, frac = best_res
            n = len(core)

            if n > global_best_n:
                global_best_n = n; global_best_core = core; best_core = core; no_improve = 0
                print(f"  Round {round_num:3d} (best of 4, {elapsed:.0f}s): N={n:,} *** NEW BEST ***", flush=True)
                pd.DataFrame([{'BANC':b,'FAFB':f,'MCNS':m} for b,(f,m) in global_best_core.items()]).to_csv(
                    f'/kaggle/working/submission_NB2_{n}.csv', index=False)
            else:
                no_improve += 1
                if n > len(best_core): best_core = core
                print(f"  Round {round_num:3d} (best of 4, {elapsed:.0f}s): N={n:,} [no improve {no_improve}/{MAX_NO_IMPROVE}]", flush=True)

    print(f"\nCONVERGED! Best N = {global_best_n:,} (CONNECTED)")
    out = f'/kaggle/working/submission_NB2_FINAL_{global_best_n}.csv'
    pd.DataFrame([{'BANC':b,'FAFB':f,'MCNS':m} for b,(f,m) in global_best_core.items()]).to_csv(out, index=False)
    print(f"Saved: {out}")
# LWCC enforced at every grow iteration, not post-hoc
# Race 2000 hub seeds in parallel, keep largest LWCC
