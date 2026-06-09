"""
KAGGLE ULTIMATE SMART GROWTH PIPELINE
=====================================
This script does three things automatically:
1. Loads your best submission (e.g. 17k) and the biological metadata.
2. Optimally assigns left-over unmatched nodes using biological metrics (Manual/NBLAST).
3. Uses Simulated Annealing (SA) to prune topological conflicts in parallel across CPU cores.
"""
import pandas as pd
import numpy as np
import pyarrow.feather as feather
import time
import os
from collections import defaultdict
import random
import multiprocessing

DATA1 = '/kaggle/input/datasets/siddhantgudwani/dataset'

# --- 1. Find Files ---
print("=" * 65)
print("  KAGGLE ULTIMATE SMART GROWTH PIPELINE")
print("=" * 65)

CORE_FILE = None
best_n = 0
for root, dirs, files in os.walk('/kaggle/input'):
    for f in files:
        if f.startswith('submission') and f.endswith('.csv'):
            try:
                base = f.replace('_FINAL', '').replace('_CLEAN', '').replace('.csv', '')
                n = int(base.split('_')[-1])
                if n > best_n:
                    best_n = n
                    CORE_FILE = os.path.join(root, f)
            except: pass

META_FILE = None
for root, dirs, files in os.walk('/kaggle/input'):
    if 'banc_888_meta.feather' in files:
        META_FILE = os.path.join(root, 'banc_888_meta.feather')
        break

if not META_FILE:
    print("WARNING: banc_888_meta.feather not found in /kaggle/input! Please upload it.")
    META_FILE = 'banc_888_meta.feather'  # Fallback to local working dir

print(f"\n1. Found Core File: {CORE_FILE} (N={best_n})")
print(f"   Found Meta File: {META_FILE}")

# --- 2. Build Candidate Pool ---
core_df = pd.read_csv(CORE_FILE, dtype=str)
core_banc = set(core_df['BANC'])
core_fafb = set(core_df['FAFB'])
core_mcns = set(core_df['MCNS'])

meta = feather.read_feather(META_FILE)

def score_row(row):
    s = 0
    if row.get('proofread') == True or str(row.get('proofread')).lower() == 'true': s += 2
    if str(row.get('roughly_proofread')).lower() == 'true': s += 1
    if str(row.get('sexually_dimorphic')).lower() == 'isomorphic': s += 3
    return s

meta['_base_score'] = meta.apply(score_row, axis=1)
unused_meta = meta[~meta['root_626'].astype(str).isin(core_banc)].copy()

# FAFB candidates
f_manual = unused_meta[['root_626','fafb_match','_base_score']].dropna(subset=['fafb_match']).copy()
f_manual.rename(columns={'fafb_match': 'FAFB'}, inplace=True)
f_manual['match_score'] = f_manual['_base_score'] + 10

f_nblast = unused_meta[['root_626','fafb_nblast_match','_base_score']].dropna(subset=['fafb_nblast_match']).copy()
f_nblast.rename(columns={'fafb_nblast_match': 'FAFB'}, inplace=True)
f_nblast['match_score'] = f_nblast['_base_score'] + 7

f_all = pd.concat([f_manual, f_nblast])
f_all['BANC'] = f_all['root_626'].astype(str)
f_all['FAFB'] = f_all['FAFB'].astype(str)
f_all = f_all[~f_all['FAFB'].isin(core_fafb)]
f_all = f_all.groupby(['BANC','FAFB'])['match_score'].max().reset_index()

# MCNS candidates
m_manual = unused_meta[['root_626','malecns_match','_base_score']].dropna(subset=['malecns_match']).copy()
m_manual.rename(columns={'malecns_match': 'MCNS'}, inplace=True)
m_manual['match_score'] = m_manual['_base_score'] + 10

m_nblast = unused_meta[['root_626','malecns_nblast_match','_base_score']].dropna(subset=['malecns_nblast_match']).copy()
m_nblast.rename(columns={'malecns_nblast_match': 'MCNS'}, inplace=True)
m_nblast['match_score'] = m_nblast['_base_score'] + 7

m_all = pd.concat([m_manual, m_nblast])
m_all['BANC'] = m_all['root_626'].astype(str)
m_all['MCNS'] = m_all['MCNS'].astype(str)
m_all = m_all[~m_all['MCNS'].isin(core_mcns)]
m_all = m_all.groupby(['BANC','MCNS'])['match_score'].max().reset_index()

# Resolve optimal bijections
f_sorted = f_all.sort_values('match_score', ascending=False)
assigned_banc = set(); assigned_fafb = set(); f_resolved = []
for _, row in f_sorted.iterrows():
    if row['BANC'] in assigned_banc or row['FAFB'] in assigned_fafb: continue
    assigned_banc.add(row['BANC']); assigned_fafb.add(row['FAFB'])
    f_resolved.append(row)
f_resolved = pd.DataFrame(f_resolved)

if len(f_resolved) > 0:
    surviving_banc = set(f_resolved['BANC'])
    m_filtered = m_all[m_all['BANC'].isin(surviving_banc)].copy()
    m_sorted = m_filtered.sort_values('match_score', ascending=False)
    assigned_banc_m = set(); assigned_mcns = set(); m_resolved = []
    for _, row in m_sorted.iterrows():
        if row['BANC'] in assigned_banc_m or row['MCNS'] in assigned_mcns: continue
        assigned_banc_m.add(row['BANC']); assigned_mcns.add(row['MCNS'])
        m_resolved.append(row)
    m_resolved = pd.DataFrame(m_resolved)

    f_scores = f_resolved[['BANC','FAFB','match_score']].rename(columns={'match_score':'f_score'})
    m_scores = m_resolved[['BANC','MCNS','match_score']].rename(columns={'match_score':'m_score'})
    new_triplets = f_scores.merge(m_scores, on='BANC')
    new_triplets['total_score'] = new_triplets['f_score'] + new_triplets['m_score']
    new_triplets = new_triplets.sort_values('total_score', ascending=False)[['BANC','FAFB','MCNS']]
else:
    new_triplets = pd.DataFrame(columns=['BANC','FAFB','MCNS'])

expanded_pool = pd.concat([core_df[['BANC','FAFB','MCNS']], new_triplets], ignore_index=True)
print(f"\n2. Assembled Expanded Pool: {len(expanded_pool):,} candidates (Core {len(core_df)} + {len(new_triplets)} New)")

# --- 3. Simulated Annealing Pruner ---
print(f"\n3. Loading Edge Lists for Topological Pruning...")
def find_edge_list(name_pattern):
    for root, dirs, files in os.walk('/kaggle/input'):
        for f in files:
            if name_pattern in f and f.endswith('.csv'):
                return os.path.join(root, f)
    return None

banc_edge_file = find_edge_list('banc_')
fafb_edge_file = find_edge_list('fafb_')
mcns_edge_file = find_edge_list('mcns_')

if not all([banc_edge_file, fafb_edge_file, mcns_edge_file]):
    print("Edge lists not found! Falling back to local DATA1 paths...")
    banc_edge_file = f'{DATA1}/banc_626_edge_list.csv'
    fafb_edge_file = f'{DATA1}/fafb_783_edge_list.csv'
    mcns_edge_file = f'{DATA1}/mcns_0.9_edge_list.csv'

banc_df = pd.read_csv(banc_edge_file, header=None, names=['src','tgt'], dtype=str)
fafb_df = pd.read_csv(fafb_edge_file, header=None, names=['src','tgt'], dtype=str)
mcns_df = pd.read_csv(mcns_edge_file, header=None, names=['src','tgt'], dtype=str)

banc_to_idx = {b: i for i, b in enumerate(expanded_pool['BANC'])}
fafb_to_idx = {f: i for i, f in enumerate(expanded_pool['FAFB'])}
mcns_to_idx = {m: i for i, m in enumerate(expanded_pool['MCNS'])}

banc_int = banc_df[banc_df['src'].isin(banc_to_idx) & banc_df['tgt'].isin(banc_to_idx)]
fafb_int = fafb_df[fafb_df['src'].isin(fafb_to_idx) & fafb_df['tgt'].isin(fafb_to_idx)]
mcns_int = mcns_df[mcns_df['src'].isin(mcns_to_idx) & mcns_df['tgt'].isin(mcns_to_idx)]

banc_idx_edges = set((banc_to_idx[s], banc_to_idx[t]) for s,t in zip(banc_int['src'], banc_int['tgt']))
fafb_idx_edges = set((fafb_to_idx[s], fafb_to_idx[t]) for s,t in zip(fafb_int['src'], fafb_int['tgt']))
mcns_idx_edges = set((mcns_to_idx[s], mcns_to_idx[t]) for s,t in zip(mcns_int['src'], mcns_int['tgt']))

edges_data = (list(banc_idx_edges), list(fafb_idx_edges), list(mcns_idx_edges), len(expanded_pool))

def sa_worker(params):
    alpha, seed, edges_data = params
    np.random.seed(seed)
    random.seed(seed)
    
    b_edges, f_edges, m_edges, num_filtered = edges_data
    b_e = set(b_edges)
    f_e = set(f_edges)
    m_e = set(m_edges)
    all_active = b_e | f_e | m_e
    
    conflicts = defaultdict(int)
    adj = defaultdict(list)
    total_conflicts = 0

    for (i,j) in all_active:
        adj[i].append((i,j))
        adj[j].append((i,j))
        in_b = (i,j) in b_e
        in_f = (i,j) in f_e
        in_m = (i,j) in m_e
        if not (in_b == in_f == in_m):
            total_conflicts += 1
            conflicts[i] += 1
            conflicts[j] += 1

    active = set(range(num_filtered))
    
    while total_conflicts > 0:
        if alpha == float('inf'):
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
                in_b = edge in b_e
                in_f = edge in f_e
                in_m = edge in m_e
                if not (in_b == in_f == in_m):
                    total_conflicts -= 1
                    conflicts[i] -= 1
                    conflicts[j] -= 1
                
                all_active.discard(edge)
                b_e.discard(edge)
                f_e.discard(edge)
                m_e.discard(edge)
                
        if worst in conflicts:
            del conflicts[worst]
            
    return (alpha, seed, len(active), sorted(list(active)))

print("\n4. Launching Parallel Simulated Annealing Pruners...")
alphas = [3, 5, 8, float('inf')]
seeds = [42, 100, 200]
tasks = []
for a in alphas:
    for s in seeds:
        if a == float('inf') and s != 42: continue
        tasks.append((a, s, edges_data))

t_start = time.time()
global_best_n = 0
best_active = None

# Fallback for kaggle if multiprocessing fails
try:
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        for res in pool.imap_unordered(sa_worker, tasks):
            a, s, final_n, active_list = res
            print(f" -> [Alpha={a} | Seed={s}] Converged at N = {final_n:,}")
            if final_n > global_best_n:
                global_best_n = final_n
                best_active = active_list
except Exception as e:
    print(f"Multiprocessing failed ({e}), running sequentially...")
    for t in tasks:
        a, s, final_n, active_list = sa_worker(t)
        print(f" -> [Alpha={a} | Seed={s}] Converged at N = {final_n:,}")
        if final_n > global_best_n:
            global_best_n = final_n
            best_active = active_list

print(f"\n==========================================")
print(f"🏆 ULTIMATE SMART GROWTH FOUND: N = {global_best_n:,}")
print(f"Total time: {time.time()-t_start:.1f}s")
print(f"==========================================")

final_df = expanded_pool.iloc[best_active].copy()
out_file = f'/kaggle/working/submission_ULTIMATE_SMART_{global_best_n}.csv'
final_df.to_csv(out_file, index=False)
print(f"Saved to {out_file}!")
