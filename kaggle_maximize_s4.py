"""
CONNECTED MULTI-SEED RACE — SEASON 4 AGGRESSIVE
=================================================
Key insight: signature-based grow from a SINGLE CONNECTED SEED
is GUARANTEED to produce a weakly connected result.

Why? Every node added must have a signature edge to an existing
core node → it's adjacent to the existing component → connected.

The only reason Seasons 1-3 were disconnected: the SEED itself
was disconnected (the biological 11,892 was multi-component).

Strategy:
1. Extract all 2,561 components from our best result as seeds
2. For each seed, run connected grow (guaranteed connected)
3. Parallelise across 4 Kaggle CPUs
4. Keep the best result
5. Perturb + regrow from the winner until convergence
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
    """Grow and return LWCC — guaranteed connected."""
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

# ── Smart perturbation ────────────────────────────────────────────────
def smart_perturb(core, frac, seed):
    rng = np.random.RandomState(seed)
    keys = list(core.keys())
    n_remove = max(1, int(len(keys) * frac))
    # Boundary removal: sort by internal degree ascending
    b_nodes = set(keys)
    degree = defaultdict(int)
    for s, t in zip(banc_df['src'], banc_df['tgt']):
        if s in b_nodes and t in b_nodes:
            degree[s] += 1; degree[t] += 1
    sorted_keys = sorted(keys, key=lambda b: degree.get(b, 0))
    pool = sorted_keys[:min(len(sorted_keys), n_remove * 3)]
    to_remove = set(rng.choice(pool, size=min(n_remove, len(pool)), replace=False))
    return {b: v for b, v in core.items() if b not in to_remove}, n_remove

# ── Worker for parallel seed racing ──────────────────────────────────
def worker_grow(args):
    seed_core, grow_seed = args
    return run_grow(seed_core, grow_seed)

def worker_perturb(args):
    core, frac, perturb_seed, grow_seed = args
    perturbed, n_remove = smart_perturb(core, frac, perturb_seed)
    grown = run_grow(perturbed, grow_seed)
    return grown, n_remove, frac

if __name__ == '__main__':
    # ── Phase 1: Multi-seed race ──────────────────────────────────────
    # Find best seed file (from any previous season)
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
    print("  CONNECTED MULTI-SEED RACE — SEASON 4 AGGRESSIVE")
    print("=" * 65)
    print(f"\nLoading seed: {SEED_FILE}")
    raw_df = pd.read_csv(SEED_FILE, dtype=str)
    raw_core = {str(b): (str(f), str(m))
                for b, f, m in zip(raw_df['BANC'], raw_df['FAFB'], raw_df['MCNS'])}

    # Extract all components from the seed
    b_nodes = set(raw_core.keys())
    visited, all_comps = set(), []
    for node in b_nodes:
        if node not in visited:
            comp, q = [], deque([node])
            visited.add(node)
            while q:
                c = q.popleft(); comp.append(c)
                for nb in banc_undirected[c]:
                    if nb in b_nodes and nb not in visited:
                        visited.add(nb); q.append(nb)
            all_comps.append(comp)
    all_comps.sort(key=len, reverse=True)
    print(f"Seed has {len(raw_core):,} nodes across {len(all_comps):,} components")
    print(f"Top component sizes: {[len(c) for c in all_comps[:10]]}")

    # Race ALL components as seeds in parallel
    N_WORKERS = min(4, cpu_count())
    top_seeds = all_comps  # all components, sorted by size descending
    seed_cores = [{b: raw_core[b] for b in comp} for comp in top_seeds]

    print(f"\nPhase 1: Racing {len(seed_cores)} seeds with {N_WORKERS} workers...")
    tasks = [(sc, i * 277 + 13) for i, sc in enumerate(seed_cores)]

    t0 = time.time()
    results = []
    best_n = 0
    best_core = {}

    with Pool(N_WORKERS) as pool:
        for i, res_core in enumerate(pool.imap_unordered(worker_grow, tasks)):
            results.append(res_core)
            n = len(res_core)
            
            if n > best_n:
                best_n = n
                best_core = res_core
                print(f"  [{i+1}/{len(tasks)}] *** NEW BEST SEED: {best_n:,} nodes! ***", flush=True)
                tmp_df = pd.DataFrame([{'BANC': b, 'FAFB': f, 'MCNS': m} for b, (f, m) in best_core.items()])
                tmp_df.to_csv(f'/kaggle/working/submission_S4_PHASE1_{best_n}.csv', index=False)
            elif (i + 1) % 50 == 0:
                print(f"  [{i+1}/{len(tasks)}] seeds processed...", flush=True)

    print(f"Phase 1 done in {time.time()-t0:.0f}s")
    global_best_n = best_n
    global_best_core = dict(best_core)

    # Save phase 1 result
    tmp_df = pd.DataFrame([{'BANC': b, 'FAFB': f, 'MCNS': m}
                            for b, (f, m) in global_best_core.items()])
    tmp_df.to_csv(f'/kaggle/working/submission_S4_PHASE1_{global_best_n}.csv', index=False)
    print(f"Phase 1 saved: submission_S4_PHASE1_{global_best_n}.csv")

    # ── Phase 2: Perturbation from winner ─────────────────────────────
    FRACS = [0.03, 0.05, 0.07, 0.08] # 4 concurrent workers
    MAX_NO_IMPROVE = 20
    no_improve = 0
    round_num = 0

    print(f"\nPhase 2: Parallel perturbation from {global_best_n:,}-node winner...")
    print(f"Running {N_WORKERS} workers concurrently per round.")
    print(f"Will stop after {MAX_NO_IMPROVE} consecutive rounds with no improvement.\n")

    with Pool(N_WORKERS) as pool:
        while no_improve < MAX_NO_IMPROVE:
            round_num += 1
            t_start = time.time()

            # Build 4 parallel tasks with different seeds and fracs
            tasks = []
            for i, frac in enumerate(FRACS):
                perturb_seed = round_num * 1000 + i * 173 + 31
                grow_seed    = round_num * 1000 + i * 277 + 13
                tasks.append((best_core, frac, perturb_seed, grow_seed))

            results = pool.map(worker_perturb, tasks)
            elapsed = time.time() - t_start

            # Pick the best result from the 4 workers
            best_result = max(results, key=lambda r: len(r[0]))
            core, n_remove, frac = best_result
            n = len(core)

            if n > global_best_n:
                global_best_n = n
                global_best_core = core
                best_core = core
                best_n = n
                no_improve = 0
                print(f"  Round {round_num:3d} (best of 4, {elapsed:.0f}s): N = {n:,} *** NEW BEST ***", flush=True)
                tmp_df = pd.DataFrame([{'BANC': b, 'FAFB': f, 'MCNS': m}
                                        for b, (f, m) in global_best_core.items()])
                tmp_df.to_csv(f'/kaggle/working/submission_S4_{n}.csv', index=False)
                print(f"    Auto-saved: submission_S4_{n}.csv", flush=True)
            else:
                no_improve += 1
                if n > best_n: best_core = core; best_n = n
                print(f"  Round {round_num:3d} (best of 4, {elapsed:.0f}s): N = {n:,}  [no improve {no_improve}/{MAX_NO_IMPROVE}]", flush=True)

    # ── Final save ─────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print(f"  CONVERGED! Best N = {global_best_n:,} (WEAKLY CONNECTED)")
    print("=" * 65)
    final_df = pd.DataFrame([{'BANC': b, 'FAFB': f, 'MCNS': m}
                              for b, (f, m) in global_best_core.items()])
    out = f'/kaggle/working/submission_S4_FINAL_{global_best_n}.csv'
    final_df.to_csv(out, index=False)
    print(f"Final file: {out}")
