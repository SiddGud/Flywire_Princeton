"""
CONNECTED GROW — BOUNDARY NODE STRATEGY
=========================================
Strategy: Our 19,827-node result has 2,561 disconnected components.
The BANC nodes that are ADJACENT to this result but NOT in it
were "blocked" because they created edge conflicts with our specific matching.

But with a DIFFERENT FAFB/MCNS counterpart, these boundary nodes
might unlock completely new, large, connected isomorphic subgraphs.

Steps:
1. Find all BANC nodes adjacent to our 19,827-result (the boundary)
2. For each boundary node, try growing a fresh connected subgraph
3. Race all boundary seeds in parallel
4. Perturb from winner
"""
import pandas as pd
import numpy as np
import time, os, sys
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
    sys.exit(1)

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
    b, f, m, grow_seed = args
    return run_grow({b: (f, m)}, grow_seed)

def worker_perturb(args):
    core, frac, perturb_seed, grow_seed = args
    perturbed, n_remove = smart_perturb(core, frac, perturb_seed)
    grown = run_grow(perturbed, grow_seed)
    return grown, n_remove, frac

if __name__ == '__main__':
    # Load S3 result and find boundary nodes
    SEED_FILE = None
    best_sc = 0
    for root, dirs, files in os.walk('/kaggle/input'):
        for f in files:
            f_lower = f.lower()
            if not f_lower.endswith('.csv'): continue
            if 'submission' not in f_lower: continue
            path = os.path.join(root, f)
            try:
                import re
                nums = re.findall(r'\d+', f_lower)
                n = int(nums[-1]) if nums else 0
                bonus = 4 if 'final' in f_lower else (2 if 'clean' in f_lower else 1)
                if n * bonus > best_sc: best_sc = n * bonus; SEED_FILE = path
            except: pass

    if SEED_FILE is None:
        print("\n" + "!"*65)
        print("ERROR: No seed file found!")
        print("Please make sure you uploaded submission_S3_FINAL_19827.csv")
        print("as a Kaggle Dataset and clicked 'Add Data' to attach it to this notebook.")
        print("!"*65 + "\n")
        sys.exit(1)

    print("=" * 65)
    print("  BOUNDARY NODE STRATEGY — NOTEBOOK 3")
    print("=" * 65)
    print(f"\nLoading: {SEED_FILE}")
    raw_df = pd.read_csv(SEED_FILE, dtype=str)
    existing_banc = set(str(b) for b in raw_df['BANC'])
    print(f"Existing result: {len(existing_banc):,} BANC nodes")

    # Find boundary: BANC nodes adjacent to existing but not in it
    boundary = set()
    for b in existing_banc:
        for nb in banc_undirected[b]:
            if nb not in existing_banc:
                boundary.add(nb)
    print(f"Boundary nodes found: {len(boundary):,}")

    # Sort boundary by their degree (most connected first)
    top_fafb = sorted(fafb_degree.keys(), key=lambda x: fafb_degree[x], reverse=True)
    top_mcns = sorted(mcns_degree.keys(), key=lambda x: mcns_degree[x], reverse=True)

    boundary_sorted = sorted(boundary,
        key=lambda b: sum(1 for nb in banc_undirected[b] if nb in existing_banc),
        reverse=True)  # most connected to existing core first

    # Build tasks: each boundary node paired with top FAFB/MCNS nodes
    tasks = []
    for i, b in enumerate(boundary_sorted[:2000]):
        f = top_fafb[i % len(top_fafb)]
        m = top_mcns[i % len(top_mcns)]
        tasks.append((b, f, m, i * 277 + 13))

    N_WORKERS = min(4, cpu_count())
    print(f"\nPhase 1: Racing {len(tasks)} boundary seeds with {N_WORKERS} workers...")
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
                tmp_df.to_csv(f'/kaggle/working/submission_NB3_PHASE1_{best_n}.csv', index=False)
            elif (i + 1) % 50 == 0:
                print(f"  [{i+1}/{len(tasks)}] seeds processed...", flush=True)

    print(f"Phase 1 done in {time.time()-t0:.0f}s")
    
    global_best_n = best_n
    global_best_core = dict(best_core)

    pd.DataFrame([{'BANC':b,'FAFB':f,'MCNS':m} for b,(f,m) in global_best_core.items()]).to_csv(
        f'/kaggle/working/submission_NB3_PHASE1_{global_best_n}.csv', index=False)

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
                    f'/kaggle/working/submission_NB3_{n}.csv', index=False)
            else:
                no_improve += 1
                if n > len(best_core): best_core = core
                print(f"  Round {round_num:3d} (best of 4, {elapsed:.0f}s): N={n:,} [no improve {no_improve}/{MAX_NO_IMPROVE}]", flush=True)

    print(f"\nCONVERGED! Best N = {global_best_n:,} (CONNECTED)")
    out = f'/kaggle/working/submission_NB3_FINAL_{global_best_n}.csv'
    pd.DataFrame([{'BANC':b,'FAFB':f,'MCNS':m} for b,(f,m) in global_best_core.items()]).to_csv(out, index=False)
    print(f"Saved: {out}")
