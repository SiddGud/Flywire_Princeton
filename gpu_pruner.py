import pandas as pd
import numpy as np
import torch
import time

# ==========================================
# CONFIGURATION
# ==========================================
# Ensure you upload the 3 edge list CSVs and 'expanded_triplets.csv' to Kaggle
FAFB_FILE = 'fafb_783_edge_list.csv'
BANC_FILE = 'banc_626_edge_list.csv'
MCNS_FILE = 'mcns_0.9_edge_list.csv'
TRIPLETS_FILE = 'expanded_triplets.csv'

# Check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if device.type == 'cpu':
    print("WARNING: GPU is not enabled! Turn on Accelerator -> GPU T4 in Kaggle settings.")

# ==========================================
# LOAD DATA
# ==========================================
print("\n1. Loading candidate triplets...")
df_triplets = pd.read_csv(TRIPLETS_FILE, dtype=str)
# Force 1-to-1 bijection just in case
df_triplets.drop_duplicates(subset=['FAFB'], keep='first', inplace=True)
df_triplets.drop_duplicates(subset=['BANC'], keep='first', inplace=True)
df_triplets.drop_duplicates(subset=['MCNS'], keep='first', inplace=True)
print(f"Candidates to process: {len(df_triplets):,}")

def build_dense_tensor(fname, target_nodes):
    print(f"Loading {fname}...")
    t0 = time.time()
    # Read edges
    df = pd.read_csv(fname, header=None, names=['src','tgt'], dtype=str)
    df = df[df['src'] != df['tgt']].drop_duplicates()
    
    # Create mapping from node_id to tensor index (0 to N-1)
    nodes = list(target_nodes)
    n2i = {n: i for i, n in enumerate(nodes)}
    
    # Filter edges to only those where BOTH src and tgt are in our candidate pool
    mask = df['src'].isin(n2i) & df['tgt'].isin(n2i)
    df = df[mask]
    
    rows = df['src'].map(n2i).values
    cols = df['tgt'].map(n2i).values
    
    # Build a dense PyTorch boolean tensor
    N = len(nodes)
    mat = torch.zeros((N, N), dtype=torch.bool, device=device)
    
    # Batch assign edges (requires long tensors for indexing)
    if len(rows) > 0:
        mat[torch.tensor(rows, dtype=torch.long), torch.tensor(cols, dtype=torch.long)] = True
        
    print(f"  -> Extracted {len(df):,} internal edges in {time.time()-t0:.1f}s")
    return mat

print("\n2. Building GPU tensors (this will take ~1.9GB VRAM per matrix)...")
mat_f = build_dense_tensor(FAFB_FILE, df_triplets['FAFB'])
mat_b = build_dense_tensor(BANC_FILE, df_triplets['BANC'])
mat_m = build_dense_tensor(MCNS_FILE, df_triplets['MCNS'])

# ==========================================
# PRUNING ALGORITHM (GPU ACCELERATED)
# ==========================================
print("\n3. Starting GPU-Accelerated Pruning...")
# Keep track of which nodes are still "alive" in the subgraph
N = len(df_triplets)
alive = torch.ones(N, dtype=torch.bool, device=device)

iteration = 0
t_start = time.time()

while True:
    # To simulate the induced subgraph of only alive nodes, 
    # we don't need to rebuild the matrix. We just zero out dead rows/cols.
    # A cleaner mathematical way: find violations across the whole matrix,
    # but only count violations where BOTH nodes are currently alive.
    
    # Diff matrices (XOR is equivalent to != for booleans)
    v_fb = mat_f ^ mat_b
    v_fm = mat_f ^ mat_m
    v_bm = mat_b ^ mat_m
    total_v = v_fb | v_fm | v_bm
    
    # Mask out edges involving dead nodes
    total_v[~alive, :] = False
    total_v[:, ~alive] = False
    
    num_violations = total_v.sum().item()
    
    if num_violations == 0:
        final_N = alive.sum().item()
        print(f"\n✅ CONVERGED! Perfect Isomorphism Reached.")
        print(f"Final N = {final_N:,}")
        break
        
    # Find the alive node involved in the most violations
    # Sum violations across rows AND columns
    row_v = total_v.sum(dim=1) + total_v.sum(dim=0)
    
    # Ignore dead nodes in argmax
    row_v[~alive] = -1
    
    worst_idx = torch.argmax(row_v).item()
    alive[worst_idx] = False
    
    iteration += 1
    if iteration % 500 == 0:
        print(f" Iter {iteration}: {alive.sum().item():,} nodes remaining | {num_violations:,} violations | Time: {time.time()-t_start:.1f}s")

# ==========================================
# SAVE RESULTS
# ==========================================
print("\n4. Saving submission CSV...")
# Filter dataframe to only alive nodes
alive_cpu = alive.cpu().numpy()
df_final = df_triplets[alive_cpu].copy()
out_name = f'submission_KAGGLE_{len(df_final)}.csv'
df_final.to_csv(out_name, index=False)
print(f"Saved exactly {len(df_final)} rows to {out_name}")
