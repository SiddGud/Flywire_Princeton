import pandas as pd
import numpy as np
import time

FAFB_FILE = 'fafb_783_edge_list.csv'
BANC_FILE = 'banc_626_edge_list.csv'
MCNS_FILE = 'mcns_0.9_edge_list.csv'
TRIPLETS_FILE = 'expanded_triplets.csv'

print("1. Loading candidate triplets...")
df_triplets = pd.read_csv(TRIPLETS_FILE, dtype=str)
print(f"Candidates to process: {len(df_triplets):,}")

def build_dense_matrix(fname, target_nodes):
    print(f"Loading {fname}...")
    t0 = time.time()
    df = pd.read_csv(fname, header=None, names=['src','tgt'], dtype=str)
    df = df[df['src'] != df['tgt']].drop_duplicates()
    
    nodes = list(target_nodes)
    n2i = {n: i for i, n in enumerate(nodes)}
    
    mask = df['src'].isin(n2i) & df['tgt'].isin(n2i)
    df = df[mask]
    
    rows = df['src'].map(n2i).values
    cols = df['tgt'].map(n2i).values
    
    N = len(nodes)
    mat = np.zeros((N, N), dtype=bool)
    
    if len(rows) > 0:
        mat[rows, cols] = True
        
    print(f"  -> Extracted {len(df):,} internal edges in {time.time()-t0:.1f}s")
    return mat

print("\n2. Building Dense Numpy Matrices (Fast Memory)...")
mat_f = build_dense_matrix(FAFB_FILE, df_triplets['FAFB'])
mat_b = build_dense_matrix(BANC_FILE, df_triplets['BANC'])
mat_m = build_dense_matrix(MCNS_FILE, df_triplets['MCNS'])

print("\n3. Starting Ultra-Fast CPU Pruning...")
N = len(df_triplets)
alive = np.ones(N, dtype=bool)

iteration = 0
t_start = time.time()

while True:
    # Compute violations only for alive nodes
    v_fb = mat_f ^ mat_b
    v_fm = mat_f ^ mat_m
    v_bm = mat_b ^ mat_m
    total_v = v_fb | v_fm | v_bm
    
    # Mask out dead nodes
    total_v[~alive, :] = False
    total_v[:, ~alive] = False
    
    num_violations = total_v.sum()
    
    if num_violations == 0:
        final_N = alive.sum()
        print(f"\n✅ CONVERGED! Perfect Isomorphism Reached.")
        print(f"Final N = {final_N:,}")
        break
        
    row_v = total_v.sum(axis=1) + total_v.sum(axis=0)
    # Ignore dead nodes
    row_v[~alive] = -1
    
    worst_idx = np.argmax(row_v)
    alive[worst_idx] = False
    
    iteration += 1
    if iteration % 500 == 0:
        print(f" Iter {iteration}: {alive.sum():,} nodes remaining | {num_violations:,} violations | Time: {time.time()-t_start:.1f}s")

print("\n4. Saving submission CSV...")
df_final = df_triplets[alive].copy()
out_name = f'submission_NBLAST_FAST_{len(df_final)}.csv'
df_final.to_csv(out_name, index=False)
print(f"Saved exactly {len(df_final)} rows to {out_name}")
