"""
CONNECTED GROW — MONTE CARLO TREE SEARCH (MCTS)
=================================================
This notebook implements an advanced search strategy based on the
Deep Research literature. Instead of Markovian "blind" simulated
annealing, this uses a Tree Search with Rollouts (an MCTS variant).

Steps:
1. Load the best result so far (e.g., from Notebook 1 or 3).
2. For the current core, identify all valid adjacent candidates (the "frontier").
3. MCTS Rollout: For each candidate, we temporarily add it to the core,
   and then perform a "fast rollout" (greedy random growth) to see its potential.
4. We permanently commit to the candidate that yielded the massive rollout size,
   effectively learning to avoid topological dead-ends and clique fragmentation.
5. If stuck, apply a small perturbation and resume tree search.
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

print("Done.\n", flush=True)

# ── Core grow functions ───────────────────────────────────────────────
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

def run_grow(starting_core, seed_num, max_iters=25):
    core = dict(starting_core)
    rng = np.random.RandomState(seed_num)
    for _ in range(max_iters):
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
    
    # Ensure connected
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

# ── MCTS Rollout Worker ──────────────────────────────────────────────
def mcts_rollout_worker(args):
    """
    Given a core and a candidate triplet to force-add,
    add the candidate and perform a greedy rollout.
    Returns the final size and the resulting core.
    """
    core, candidate, rollout_seed = args
    # Force add candidate
    test_core = dict(core)
    b, f, m = candidate
    test_core[b] = (f, m)
    
    # Fast rollout (limit iterations to save time during search tree expansion)
    grown = run_grow(test_core, rollout_seed, max_iters=15)
    return len(grown), grown

if __name__ == '__main__':
    # Find best seed file
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
        print("Please make sure you uploaded a previous submission as a dataset.")
        print("!"*65 + "\n")
        sys.exit(1)

    print("=" * 65)
    print("  MCTS SEARCH STRATEGY — NOTEBOOK 4")
    print("=" * 65)
    print(f"\nLoading base state from: {SEED_FILE}")
    raw_df = pd.read_csv(SEED_FILE, dtype=str)
    raw_core = {str(b): (str(f), str(m)) for b, f, m in zip(raw_df['BANC'], raw_df['FAFB'], raw_df['MCNS'])}

    # Ensure starting core is connected
    best_core = run_grow(raw_core, 42, max_iters=2)
    global_best_n = len(best_core)
    global_best_core = dict(best_core)
    print(f"Starting Connected State: {global_best_n:,} nodes\n")

    N_WORKERS = min(4, cpu_count())
    print(f"Phase 2: MCTS Guided Search with {N_WORKERS} workers...")
    
    MAX_ROUNDS = 500
    no_improve = 0
    
    with Pool(N_WORKERS) as pool:
        for round_num in range(1, MAX_ROUNDS + 1):
            t_start = time.time()
            
            # Step 1: Perturb slightly to create a state to search from
            # (MCTS requires some room to grow to evaluate branches)
            frac = np.random.choice([0.02, 0.04, 0.06])
            perturbed, n_rem = smart_perturb(best_core, frac, round_num * 881)
            
            # Step 2: Get immediate frontier candidates
            b2i = {b: i for i, b in enumerate(perturbed.keys())}
            f2i = {f: i for i, (f, m) in enumerate(perturbed.values())}
            m2i = {m: i for i, (f, m) in enumerate(perturbed.values())}
            b_sigs = compute_signatures(banc_df, b2i)
            f_sigs = compute_signatures(fafb_df, f2i)
            m_sigs = compute_signatures(mcns_df, m2i)
            shared = set(b_sigs.keys()) & set(f_sigs.keys()) & set(m_sigs.keys())
            
            cands = []
            rng = np.random.RandomState(round_num)
            for sig in shared:
                bn, fn, mn = b_sigs[sig], f_sigs[sig], m_sigs[sig]
                rng.shuffle(bn); rng.shuffle(fn); rng.shuffle(mn)
                for i in range(min(len(bn), len(fn), len(mn))):
                    if bn[i] not in perturbed:
                        cands.append((bn[i], fn[i], mn[i]))
            
            # Subsample candidates to avoid exploding tree
            if len(cands) > 16:
                idx = rng.choice(len(cands), 16, replace=False)
                cands = [cands[i] for i in idx]
                
            if not cands:
                print(f"  Round {round_num:3d}: Dead end. Perturbing deeper...", flush=True)
                continue
                
            # Step 3: Run Rollouts in Parallel
            tasks = [(perturbed, c, round_num * 1000 + i) for i, c in enumerate(cands)]
            results = pool.map(mcts_rollout_worker, tasks)
            
            # Step 4: Evaluate Rollout Scores
            best_rollout_n, best_rollout_core = max(results, key=lambda r: r[0])
            elapsed = time.time() - t_start
            
            if best_rollout_n > global_best_n:
                global_best_n = best_rollout_n
                global_best_core = best_rollout_core
                best_core = best_rollout_core
                no_improve = 0
                print(f"  MCTS Round {round_num:3d} | Searched {len(cands)} branches | N={best_rollout_n:,} ({elapsed:.1f}s) *** NEW BEST ***", flush=True)
                pd.DataFrame([{'BANC':b,'FAFB':f,'MCNS':m} for b,(f,m) in global_best_core.items()]).to_csv(
                    f'/kaggle/working/submission_NB4_MCTS_{global_best_n}.csv', index=False)
            else:
                no_improve += 1
                if best_rollout_n > len(best_core):
                    best_core = best_rollout_core  # Move to the best branch even if not global best
                print(f"  MCTS Round {round_num:3d} | Searched {len(cands)} branches | N={best_rollout_n:,} ({elapsed:.1f}s) [no improve {no_improve}]", flush=True)

    print(f"\nMCTS Search Complete. Best N = {global_best_n:,}")
    out = f'/kaggle/working/submission_NB4_MCTS_FINAL_{global_best_n}.csv'
    pd.DataFrame([{'BANC':b,'FAFB':f,'MCNS':m} for b,(f,m) in global_best_core.items()]).to_csv(out, index=False)
    print(f"Saved: {out}")
