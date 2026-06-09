import pandas as pd
import numpy as np
import pyarrow.feather as feather
import networkx as nx
from scipy.optimize import linear_sum_assignment
import time

t_start = time.time()

print("1. Loading metadata...")
meta = feather.read_feather('banc_888_meta.feather')
meta['root_626'] = meta['root_626'].astype(str)

print("2. Extracting FAFB and MCNS candidate mappings...")
# Extract all FAFB candidates and their associated cell types
f_cands = []
for col in ['fafb_match', 'fafb_nblast_match']:
    df = meta[['root_626', 'cell_type', col]].dropna(subset=[col, 'cell_type']).copy()
    df.rename(columns={col: 'FAFB'}, inplace=True)
    f_cands.append(df)
f_all = pd.concat(f_cands).groupby('FAFB')['cell_type'].first().reset_index() # just take first cell type associated

m_cands = []
for col in ['malecns_match', 'malecns_nblast_match']:
    df = meta[['root_626', 'cell_type', col]].dropna(subset=[col, 'cell_type']).copy()
    df.rename(columns={col: 'MCNS'}, inplace=True)
    m_cands.append(df)
m_all = pd.concat(m_cands).groupby('MCNS')['cell_type'].first().reset_index()

b_all = meta.dropna(subset=['cell_type'])[['root_626', 'cell_type']].copy()
b_all.rename(columns={'root_626': 'BANC'}, inplace=True)

# Group by cell type
b_grouped = b_all.groupby('cell_type')['BANC'].apply(list).to_dict()
f_grouped = f_all.groupby('cell_type')['FAFB'].apply(list).to_dict()
m_grouped = m_all.groupby('cell_type')['MCNS'].apply(list).to_dict()

all_cell_types = set(b_grouped.keys()) & set(f_grouped.keys()) & set(m_grouped.keys())
print(f"Found {len(all_cell_types)} shared cell types.")

print("3. Loading edge lists for topology...")
fafb_df = pd.read_csv('fafb_783_edge_list.csv', header=None, names=['src','tgt'], dtype=str)
banc_df = pd.read_csv('banc_626_edge_list.csv', header=None, names=['src','tgt'], dtype=str)
mcns_df = pd.read_csv('mcns_0.9_edge_list.csv', header=None, names=['src','tgt'], dtype=str)

print("4. Computing In/Out Degrees...")
def get_degrees(df):
    in_deg = df.groupby('tgt').size().to_dict()
    out_deg = df.groupby('src').size().to_dict()
    return in_deg, out_deg

b_in, b_out = get_degrees(banc_df)
f_in, f_out = get_degrees(fafb_df)
m_in, m_out = get_degrees(mcns_df)

print("5. Computing PageRank (this takes ~1 min per graph)...")
Gb = nx.from_pandas_edgelist(banc_df, 'src', 'tgt', create_using=nx.DiGraph())
pr_b = nx.pagerank(Gb, alpha=0.85)
del Gb

Gf = nx.from_pandas_edgelist(fafb_df, 'src', 'tgt', create_using=nx.DiGraph())
pr_f = nx.pagerank(Gf, alpha=0.85)
del Gf

Gm = nx.from_pandas_edgelist(mcns_df, 'src', 'tgt', create_using=nx.DiGraph())
pr_m = nx.pagerank(Gm, alpha=0.85)
del Gm

print("6. Bipartite Hungarian Alignment within cell-type buckets...")
def get_features(node, pr_dict, in_deg, out_deg):
    return np.array([
        pr_dict.get(node, 0.0) * 1e5, # scale PageRank up
        in_deg.get(node, 0),
        out_deg.get(node, 0)
    ])

b_f_pairs = []
b_m_pairs = []

matched_b = set()
matched_f = set()
matched_m = set()

for ct in all_cell_types:
    b_nodes = b_grouped[ct]
    f_nodes = f_grouped[ct]
    m_nodes = m_grouped[ct]
    
    if not b_nodes or not f_nodes or not m_nodes: continue
    
    # BANC vs FAFB
    cost_bf = np.zeros((len(b_nodes), len(f_nodes)))
    for i, b in enumerate(b_nodes):
        fb = get_features(b, pr_b, b_in, b_out)
        for j, f in enumerate(f_nodes):
            ff = get_features(f, pr_f, f_in, f_out)
            # L1 distance
            cost_bf[i, j] = np.sum(np.abs(fb - ff))
            
    row_ind, col_ind = linear_sum_assignment(cost_bf)
    for r, c in zip(row_ind, col_ind):
        b, f = b_nodes[r], f_nodes[c]
        if b not in matched_b and f not in matched_f:
            b_f_pairs.append((b, f))
            matched_b.add(b); matched_f.add(f)
            
    # BANC vs MCNS
    cost_bm = np.zeros((len(b_nodes), len(m_nodes)))
    for i, b in enumerate(b_nodes):
        fb = get_features(b, pr_b, b_in, b_out)
        for j, m in enumerate(m_nodes):
            fm = get_features(m, pr_m, m_in, m_out)
            cost_bm[i, j] = np.sum(np.abs(fb - fm))
            
    row_ind, col_ind = linear_sum_assignment(cost_bm)
    for r, c in zip(row_ind, col_ind):
        b, m = b_nodes[r], m_nodes[c]
        if m not in matched_m:
            b_m_pairs.append((b, m))
            matched_m.add(m)

print("7. Merging into Triplet Pool...")
df_bf = pd.DataFrame(b_f_pairs, columns=['BANC', 'FAFB'])
df_bm = pd.DataFrame(b_m_pairs, columns=['BANC', 'MCNS'])

triplets = pd.merge(df_bf, df_bm, on='BANC', how='inner')

# Deduplicate strictly
triplets = triplets.drop_duplicates(subset=['BANC'])
triplets = triplets.drop_duplicates(subset=['FAFB'])
triplets = triplets.drop_duplicates(subset=['MCNS'])

print(f"Final Topological Triplet Pool Size: {len(triplets):,}")
triplets.to_csv('topological_triplets.csv', index=False)
print(f"Saved to topological_triplets.csv. Total time: {time.time()-t_start:.0f}s")
