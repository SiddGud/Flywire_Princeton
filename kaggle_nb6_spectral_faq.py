"""
RESTRICTED-FOCUS SEEDED GRAPH MATCHING (FAQ)
============================================
This notebook implements the Continuous Relaxation strategy to bridge biological gaps.
It uses scipy's Fast Approximate QAP (FAQ) algorithm on the boundary "halo" of the
15k-node core to probabilistically bridge over segmentation noise.
"""
import pandas as pd
import numpy as np
import time, os, sys
from collections import defaultdict, deque
from scipy.optimize import quadratic_assignment

print("==================================================")
print("  PHASE 6: SPECTRAL FAQ CONTINUOUS RELAXATION")
print("==================================================")

# ── 1. Load Data ────────────────────────────────────────────────────────
DATA1 = None
for root, dirs, files in os.walk('/kaggle/input'):
    if 'fafb_783_edge_list.csv' in files:
        DATA1 = root
        break

if DATA1 is None:
    print("ERROR: Could not find dataset files!")
    sys.exit(1)

print("Loading edge lists...")
fafb_df = pd.read_csv(f'{DATA1}/fafb_783_edge_list.csv', header=None, names=['src','tgt'], dtype=str)
banc_df = pd.read_csv(f'{DATA1}/banc_626_edge_list.csv', header=None, names=['src','tgt'], dtype=str)
mcns_df = pd.read_csv(f'{DATA1}/mcns_0.9_edge_list.csv', header=None, names=['src','tgt'], dtype=str)

banc_set = set(zip(banc_df['src'], banc_df['tgt']))
fafb_set = set(zip(fafb_df['src'], fafb_df['tgt']))
mcns_set = set(zip(mcns_df['src'], mcns_df['tgt']))

banc_undirected = defaultdict(set)
for s, t in banc_set:
    banc_undirected[s].add(t); banc_undirected[t].add(s)

fafb_undirected = defaultdict(set)
for s, t in fafb_set:
    fafb_undirected[s].add(t); fafb_undirected[t].add(s)

mcns_undirected = defaultdict(set)
for s, t in mcns_set:
    mcns_undirected[s].add(t); mcns_undirected[t].add(s)

# ── 2. Load the 15k Core ────────────────────────────────────────────────
core = {}
best_file = None
best_size = 0

print("Finding best submission seed...")
for root, dirs, files in os.walk('/kaggle/input'):
    for f in files:
        if f.lower().endswith('.csv') and 'submission' in f.lower():
            try:
                df = pd.read_csv(os.path.join(root, f), dtype=str)
                sz = len(df)
                if sz > best_size:
                    best_size = sz
                    best_file = os.path.join(root, f)
            except: pass

if not best_file:
    print("ERROR: No seed submission found! Please upload the 15k core.")
    sys.exit(1)

print(f"Loading {best_file} ({best_size} nodes) as fixed seeds...")
df = pd.read_csv(best_file, dtype=str)
for b, f, m in zip(df['BANC'], df['FAFB'], df['MCNS']):
    core[str(b)] = (str(f), str(m))

# Extract strict connected component just to be safe
def extract_lwcc(core_dict):
    b_nodes = set(core_dict.keys())
    adj = defaultdict(set)
    for u, v in banc_set:
        if u in b_nodes and v in b_nodes:
            adj[u].add(v)
            adj[v].add(u)
    if not b_nodes: return {}
    start = list(b_nodes)[0]
    visited = set([start])
    q = deque([start])
    while q:
        curr = q.popleft()
        for nb in adj[curr]:
            if nb not in visited:
                visited.add(nb)
                q.append(nb)
    return {k: core_dict[k] for k in visited}

core = extract_lwcc(core)
print(f"Verified strict connected core: {len(core)} nodes.")
global_best = len(core)

# ── 3. The Continuous Relaxation Loop ───────────────────────────────────
# We repeatedly extract the boundary "halo", align it continuously, and snap it back.
MAX_HALO_SIZE = 1500

for iteration in range(50):
    t0 = time.time()
    
    # Identify Boundary in BANC
    b_core_set = set(core.keys())
    f_core_set = {f for f, m in core.values()}
    m_core_set = {m for f, m in core.values()}
    
    b_boundary = set()
    for b in b_core_set:
        for nb in banc_undirected[b]:
            if nb not in b_core_set:
                b_boundary.add(nb)
                
    # If boundary is too large, sample it to avoid OOM in dense FAQ matrix
    b_boundary = list(b_boundary)
    np.random.shuffle(b_boundary)
    b_boundary = b_boundary[:MAX_HALO_SIZE]
    
    if not b_boundary:
        print("No boundary left to search!")
        break
        
    # Gather potential FAFB and MCNS boundary nodes
    f_boundary = set()
    for f in f_core_set:
        for nb in fafb_undirected[f]:
            if nb not in f_core_set:
                f_boundary.add(nb)
                
    m_boundary = set()
    for m in m_core_set:
        for nb in mcns_undirected[m]:
            if nb not in m_core_set:
                m_boundary.add(nb)
                
    f_boundary = list(f_boundary)
    m_boundary = list(m_boundary)
    
    # We must match BANC to an intersection of FAFB and MCNS. 
    # To keep FAQ 2D, we will optimize BANC against FAFB first, 
    # then rigorously verify MCNS validity discretely.
    
    np.random.shuffle(f_boundary)
    f_boundary = f_boundary[:MAX_HALO_SIZE * 2] # Provide extra candidates
    
    if not f_boundary:
        break
        
    N1 = len(b_boundary)
    N2 = len(f_boundary)
    N_max = max(N1, N2)
    
    # Build Dense Adjacency Matrices for FAQ
    A = np.zeros((N_max, N_max), dtype=np.float32)
    B = np.zeros((N_max, N_max), dtype=np.float32)
    
    b_idx = {n: i for i, n in enumerate(b_boundary)}
    f_idx = {n: i for i, n in enumerate(f_boundary)}
    
    # Populate A (BANC)
    for i, u in enumerate(b_boundary):
        for j, v in enumerate(b_boundary):
            if (u, v) in banc_set: A[i, j] = 1.0
            
    # Populate B (FAFB)
    for i, u in enumerate(f_boundary):
        for j, v in enumerate(f_boundary):
            if (u, v) in fafb_set: B[i, j] = 1.0
            
    # Add Seed Affinities (Ghost Edges to Core)
    # If b_i is connected to core node c, and f_j is connected to f_core node c, 
    # we artificially boost their edge weight (Continuous Relaxation)
    for i, b in enumerate(b_boundary):
        b_core_neighbors = banc_undirected[b].intersection(b_core_set)
        if not b_core_neighbors: continue
        for j, f in enumerate(f_boundary):
            f_core_neighbors = fafb_undirected[f].intersection(f_core_set)
            
            # Count matching core neighbors
            matches = sum(1 for bn in b_core_neighbors if core[bn][0] in f_core_neighbors)
            if matches > 0:
                A[i, i] += matches * 0.5
                B[j, j] += matches * 0.5
                
    # Run Continuous Relaxation (FAQ)
    print(f"  Iter {iteration:2d} | Running FAQ on {N1} BANC x {N2} FAFB boundary nodes...", end='', flush=True)
    try:
        res = quadratic_assignment(A, B, method='faq', options={'maximize': True, 'maxiter': 30})
        col_ind = res.col_ind
    except Exception as e:
        print(f" FAQ failed: {e}")
        continue
        
    # Extract Proposed Matches
    proposed_b = []
    proposed_f = []
    for i in range(N1):
        j = col_ind[i]
        if j < N2:
            proposed_b.append(b_boundary[i])
            proposed_f.append(f_boundary[j])
            
    # Strict Discrete Proofreading
    # We now take the probabilistic matches and strictly verify them against FAFB AND MCNS
    added = 0
    for b, f in zip(proposed_b, proposed_f):
        if b in core: continue
        
        # Find valid MCNS match
        b_in = {u for u in banc_df[banc_df['tgt'] == b]['src']}
        b_out = {v for v in banc_df[banc_df['src'] == b]['tgt']}
        
        # Check FAFB strict isomorphism
        valid_f = True
        for core_b in b_in.intersection(b_core_set):
            if (core[core_b][0], f) not in fafb_set: valid_f = False; break
        for core_b in b_out.intersection(b_core_set):
            if (f, core[core_b][0]) not in fafb_set: valid_f = False; break
            
        if not valid_f: continue
        
        # Search MCNS boundary for a valid strict match
        best_m = None
        for m in m_boundary:
            valid_m = True
            for core_b in b_in.intersection(b_core_set):
                if (core[core_b][1], m) not in mcns_set: valid_m = False; break
            for core_b in b_out.intersection(b_core_set):
                if (m, core[core_b][1]) not in mcns_set: valid_m = False; break
            if valid_m:
                best_m = m
                break
                
        if best_m:
            core[b] = (f, best_m)
            b_core_set.add(b)
            f_core_set.add(f)
            m_core_set.add(best_m)
            added += 1
            
    t1 = time.time()
    
    # Save if improved
    if len(core) > global_best:
        core = extract_lwcc(core) # Double check strict connectivity
        global_best = len(core)
        print(f" +{added} nodes! | *** NEW RECORD: {global_best} *** | {t1-t0:.1f}s")
        out_csv = f'/kaggle/working/submission_NB6_SPECTRAL_{global_best}.csv'
        pd.DataFrame([{'BANC':b,'FAFB':f,'MCNS':m} for b,(f,m) in core.items()]).to_csv(out_csv, index=False)
    else:
        print(f" +0 nodes. | Time: {t1-t0:.1f}s")

print("Notebook 6 Execution Complete.")
