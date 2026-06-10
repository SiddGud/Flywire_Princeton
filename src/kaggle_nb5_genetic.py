"""
CONNECTED GROW — GENETIC ALGORITHM (GA) WITH CROSSOVER
======================================================
This notebook implements Strategy 5: A Genetic Algorithm.
Instead of constructively growing from a single core, it maintains a 
population of multiple massive valid subgraphs (e.g. from NB1, NB2, NB4).

1. Crossover: Combines two parent subgraphs. Resolves conflicts, 
   enforces strict isomorphism, and runs extract_lwcc to ensure 
   mathematical connectivity.
2. Mutation: Randomly perturbs the offspring and regrows it.
3. Selection: Keeps the top N largest subgraphs to breed the next generation.
"""
import pandas as pd
import numpy as np
import time, os, sys, random
from collections import defaultdict, deque
from multiprocessing import Pool, cpu_count

# ── 1. Load Data ────────────────────────────────────────────────────────
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

# ── 2. Core Functions ───────────────────────────────────────────────────
def extract_lwcc(core):
    b_nodes = set(core.keys())
    if not b_nodes: return {}
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

def compute_signatures(df, core_map):
    sig_in  = defaultdict(list); sig_out = defaultdict(list)
    mask_out = df['src'].isin(core_map) & (~df['tgt'].isin(core_map))
    for s, t in zip(df[mask_out]['src'], df[mask_out]['tgt']): sig_in[t].append(core_map[s])
    mask_in = (~df['src'].isin(core_map)) & df['tgt'].isin(core_map)
    for s, t in zip(df[mask_in]['src'], df[mask_in]['tgt']): sig_out[s].append(core_map[t])
    signatures = defaultdict(list)
    for n in set(sig_in.keys()) | set(sig_out.keys()):
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
            if ((b, cb) in banc_set) != ((f, cf) in fafb_set) or ((b, cb) in banc_set) != ((m, cm) in mcns_set): ok = False; break
            if ((cb, b) in banc_set) != ((cf, f) in fafb_set) or ((cb, b) in banc_set) != ((cm, m) in mcns_set): ok = False; break
        if ok:
            core[b] = (f, m); added += 1
    return added

def run_grow(starting_core, seed_num, max_iters=15):
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
                if bn[i] not in core: cands.append({'BANC': bn[i], 'FAFB': fn[i], 'MCNS': mn[i]})
        if not cands or strict_grow_shuffled(pd.DataFrame(cands), core, rng) == 0: break
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
    return {b: v for b, v in core.items() if b not in to_remove}

# ── 3. GA Operators ─────────────────────────────────────────────────────
def crossover(core1, core2, seed):
    rng = np.random.RandomState(seed)
    merged = {}
    conflicts = set()
    # Find conflicts
    for b, (f, m) in core1.items():
        if b in core2 and core2[b] != (f, m): conflicts.add(b)
    # Merge non-conflicting
    for b, fm in core1.items():
        if b not in conflicts: merged[b] = fm
    for b, fm in core2.items():
        if b not in conflicts: merged[b] = fm
        
    # We now have a merged dictionary. But it might not be strictly isomorphic!
    # To fix this, we start with an empty core and try to add all merged nodes.
    valid_core = {}
    cands = [{'BANC': b, 'FAFB': f, 'MCNS': m} for b, (f, m) in merged.items()]
    # To prevent bias, shuffle candidates
    rng.shuffle(cands)
    # Just use strict_grow_shuffled to build the guaranteed valid subset
    cands_df = pd.DataFrame(cands)
    if not cands_df.empty:
        strict_grow_shuffled(cands_df, valid_core, rng)
        
    # Ensure it's connected
    return extract_lwcc(valid_core)

# ── 4. Multiprocessing Worker ───────────────────────────────────────────
def ga_worker(args):
    parent1, parent2, seed = args
    # 1. Crossover
    offspring = crossover(parent1, parent2, seed)
    # 2. Mutate (Delete 2-6% of weak nodes)
    frac = np.random.choice([0.02, 0.04, 0.06])
    mutated = smart_perturb(offspring, frac, seed)
    # 3. Grow
    grown = run_grow(mutated, seed, max_iters=20)
    return len(grown), grown

if __name__ == '__main__':
    print("=" * 65)
    print("  GENETIC ALGORITHM WITH CROSSOVER — NOTEBOOK 5")
    print("=" * 65)
    
    # Init Population
    population = []
    print("Loading initial population from Kaggle datasets...")
    for root, dirs, files in os.walk('/kaggle/input'):
        for f in files:
            if f.lower().endswith('.csv') and 'submission' in f.lower():
                try:
                    df = pd.read_csv(os.path.join(root, f), dtype=str)
                    core = {str(b): (str(fa), str(m)) for b, fa, m in zip(df['BANC'], df['FAFB'], df['MCNS'])}
                    conn_core = extract_lwcc(core)
                    if len(conn_core) > 500: # Only keep decently sized ones
                        population.append(conn_core)
                        print(f"  Loaded {f}: {len(conn_core):,} connected nodes")
                except: pass
                
    if not population:
        print("ERROR: No valid submissions found to start population!")
        sys.exit(1)
        
    # Sort population by size
    population = sorted(population, key=len, reverse=True)
    POP_SIZE = 10
    population = population[:POP_SIZE] # Keep top 10
    
    global_best_n = len(population[0])
    print(f"\nPopulation initialized with {len(population)} cores.")
    print(f"Current Global Best: {global_best_n:,}")
    
    N_WORKERS = min(4, cpu_count())
    print(f"Starting Evolution with {N_WORKERS} workers...\n")
    
    generation = 1
    with Pool(N_WORKERS) as pool:
        while True:
            t_start = time.time()
            
            # Generate tasks: pick 4 pairs of parents
            tasks = []
            for i in range(N_WORKERS):
                # Tournament selection (bias towards larger cores)
                p1 = population[int(abs(np.random.normal(0, POP_SIZE/3))) % len(population)]
                p2 = population[int(abs(np.random.normal(0, POP_SIZE/3))) % len(population)]
                tasks.append((p1, p2, generation * 1000 + i))
                
            results = pool.map(ga_worker, tasks)
            
            # Integrate offspring
            new_best_found = False
            for n, core in results:
                if n > 500:
                    population.append(core)
                if n > global_best_n:
                    global_best_n = n
                    global_best_core = core
                    new_best_found = True
                    
            # Sort and cull population
            population = sorted(population, key=len, reverse=True)
            # Remove duplicates by size to ensure diversity
            unique_pop = []
            seen_sizes = set()
            for p in population:
                if len(p) not in seen_sizes:
                    unique_pop.append(p)
                    seen_sizes.add(len(p))
            population = unique_pop[:POP_SIZE]
            
            elapsed = time.time() - t_start
            
            best_in_gen = max(r[0] for r in results)
            avg_in_gen = int(np.mean([r[0] for r in results]))
            
            if new_best_found:
                print(f"  Gen {generation:4d} | Best Offspring: {best_in_gen:,} (Avg: {avg_in_gen:,}) | {elapsed:.1f}s | *** NEW GLOBAL BEST: {global_best_n:,} ***", flush=True)
                out = f'/kaggle/working/submission_NB5_GENETIC_{global_best_n}.csv'
                pd.DataFrame([{'BANC':b,'FAFB':f,'MCNS':m} for b,(f,m) in global_best_core.items()]).to_csv(out, index=False)
            else:
                print(f"  Gen {generation:4d} | Best Offspring: {best_in_gen:,} (Avg: {avg_in_gen:,}) | {elapsed:.1f}s | Global Best: {global_best_n:,}", flush=True)
                
            generation += 1
# Crossover: merge non-conflicting nodes from two parent cores
# Tournament selection: keep top 10 largest unique genomes
