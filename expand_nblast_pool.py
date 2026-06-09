import pandas as pd
import pyarrow.feather as feather
import numpy as np
import scipy.sparse as sp
import time

print("Loading BANC metadata...")
t0 = time.time()
meta = feather.read_feather('banc_888_meta.feather')

# 1. Collect all FAFB matches
f1 = meta[['root_626', 'fafb_match']].dropna().rename(columns={'fafb_match': 'FAFB'})
f2 = meta[['root_626', 'fafb_nblast_match']].dropna().rename(columns={'fafb_nblast_match': 'FAFB'})
f_all = pd.concat([f1, f2]).drop_duplicates()

# 2. Collect all MCNS matches
m1 = meta[['root_626', 'malecns_match']].dropna().rename(columns={'malecns_match': 'MCNS'})
m2 = meta[['root_626', 'malecns_nblast_match']].dropna().rename(columns={'malecns_nblast_match': 'MCNS'})
m_all = pd.concat([m1, m2]).drop_duplicates()

# 3. Merge into triplets
triplets = f_all.merge(m_all, on='root_626')
triplets.rename(columns={'root_626': 'BANC'}, inplace=True)
print(f"Raw potential triplets: {len(triplets):,} (loaded in {time.time()-t0:.1f}s)")

# 4. Strict 1-to-1 Bijection resolution
print("Resolving bijections...")
triplets = triplets.sample(frac=1, random_state=42) # Randomize so greedy keeps random one
triplets.drop_duplicates(subset=['FAFB'], keep='first', inplace=True)
triplets.drop_duplicates(subset=['BANC'], keep='first', inplace=True)
triplets.drop_duplicates(subset=['MCNS'], keep='first', inplace=True)
print(f"Unique bijection candidates: {len(triplets):,}")

# Save to disk just in case
triplets.to_csv("expanded_triplets.csv", index=False)

# 5. Fast Pruning
print("\nLoading sparse matrices for pruning...")
def load_graph(fname):
    df = pd.read_csv(fname, header=None, names=['src','tgt'], dtype=str)
    df = df[df['src'] != df['tgt']].drop_duplicates()
    nodes = sorted(set(df['src']) | set(df['tgt']))
    n2i = {n: i for i, n in enumerate(nodes)}
    rows = df['src'].map(n2i).values
    cols = df['tgt'].map(n2i).values
    data = np.ones(len(rows), dtype=bool)
    mat = sp.csr_matrix((data, (rows, cols)), shape=(len(nodes), len(nodes)), dtype=bool)
    return mat, n2i

t0 = time.time()
g_f, n2i_f = load_graph('fafb_783_edge_list.csv')
g_b, n2i_b = load_graph('banc_626_edge_list.csv')
g_m, n2i_m = load_graph('mcns_0.9_edge_list.csv')
print(f"Graphs loaded in {time.time()-t0:.1f}s")

# Filter triplets to those where nodes exist in the edge lists
print("\nFiltering triplets...")
valid = []
for idx, row in triplets.iterrows():
    if row['FAFB'] in n2i_f and row['BANC'] in n2i_b and row['MCNS'] in n2i_m:
        valid.append(row)
df = pd.DataFrame(valid)
print(f"Valid graph candidates: {len(df):,}")

if len(df) == 0:
    print("Error: No valid candidates.")
    exit()

def get_subgraphs(curr_df):
    idx_f = [n2i_f[n] for n in curr_df['FAFB']]
    idx_b = [n2i_b[n] for n in curr_df['BANC']]
    idx_m = [n2i_m[n] for n in curr_df['MCNS']]
    s_f = g_f[idx_f][:, idx_f].toarray()
    s_b = g_b[idx_b][:, idx_b].toarray()
    s_m = g_m[idx_m][:, idx_m].toarray()
    return s_f, s_b, s_m

print("\nStarting Greedy Pruning...")
iteration = 0
current_df = df.copy()

while True:
    s_f, s_b, s_m = get_subgraphs(current_df)
    v_fb = (s_f != s_b)
    v_fm = (s_f != s_m)
    v_bm = (s_b != s_m)
    
    total_v = v_fb | v_fm | v_bm
    num_violations = total_v.sum()
    
    if num_violations == 0:
        print(f"\n✅ CONVERGED! Final Isomorphic Subgraph N = {len(current_df):,}")
        break
        
    row_v = total_v.sum(axis=1) + total_v.sum(axis=0)
    worst_idx = np.argmax(row_v)
    
    current_df = current_df.drop(current_df.index[worst_idx]).reset_index(drop=True)
    iteration += 1
    
    if iteration % 500 == 0:
        print(f" Iter {iteration}: {len(current_df)} nodes remaining, {num_violations} violations")

current_df.to_csv(f'submission_NBLAST_EXPANDED_{len(current_df)}.csv', index=False)
print(f"Saved to submission_NBLAST_EXPANDED_{len(current_df)}.csv")
