import pandas as pd
import numpy as np
import networkx as nx
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
import time

t_start = time.time()

print("1. Loading Edge Lists...")
fafb_df = pd.read_csv('fafb_783_edge_list.csv', header=None, names=['src','tgt'], dtype=str)
banc_df = pd.read_csv('banc_626_edge_list.csv', header=None, names=['src','tgt'], dtype=str)
mcns_df = pd.read_csv('mcns_0.9_edge_list.csv', header=None, names=['src','tgt'], dtype=str)

print("2. Extracting all unique nodes...")
banc_nodes = list(set(banc_df['src']) | set(banc_df['tgt']))
fafb_nodes = list(set(fafb_df['src']) | set(fafb_df['tgt']))
mcns_nodes = list(set(mcns_df['src']) | set(mcns_df['tgt']))

print(f"Total Nodes: BANC={len(banc_nodes)}, FAFB={len(fafb_nodes)}, MCNS={len(mcns_nodes)}")

print("3. Computing Topology (In/Out Degrees)...")
def get_degrees(df):
    in_deg = df.groupby('tgt').size().to_dict()
    out_deg = df.groupby('src').size().to_dict()
    return in_deg, out_deg

b_in, b_out = get_degrees(banc_df)
f_in, f_out = get_degrees(fafb_df)
m_in, m_out = get_degrees(mcns_df)

print("4. Computing PageRank (this takes ~3 mins total)...")
def compute_pr(df):
    G = nx.from_pandas_edgelist(df, 'src', 'tgt', create_using=nx.DiGraph())
    pr = nx.pagerank(G, alpha=0.85)
    del G
    return pr

pr_b = compute_pr(banc_df)
pr_f = compute_pr(fafb_df)
pr_m = compute_pr(mcns_df)

print("5. Preparing Feature Matrix for K-Means...")
def extract_features(nodes, pr, in_d, out_d):
    X = np.zeros((len(nodes), 3))
    for i, n in enumerate(nodes):
        X[i, 0] = pr.get(n, 0)
        X[i, 1] = in_d.get(n, 0)
        X[i, 2] = out_d.get(n, 0)
    # Log transform to handle power-law distributions
    X[:, 0] = np.log1p(X[:, 0] * 1e6) 
    X[:, 1] = np.log1p(X[:, 1])
    X[:, 2] = np.log1p(X[:, 2])
    return X

X_b = extract_features(banc_nodes, pr_b, b_in, b_out)
X_f = extract_features(fafb_nodes, pr_f, f_in, f_out)
X_m = extract_features(mcns_nodes, pr_m, m_in, m_out)

# Pool all features
X_all = np.vstack([X_b, X_f, X_m])
scaler = StandardScaler()
X_all_scaled = scaler.fit_transform(X_all)

# Split them back
X_b_scaled = X_all_scaled[:len(banc_nodes)]
X_f_scaled = X_all_scaled[len(banc_nodes):len(banc_nodes)+len(fafb_nodes)]
X_m_scaled = X_all_scaled[len(banc_nodes)+len(fafb_nodes):]

print("6. Running K-Means Clustering (K=5000)...")
kmeans = MiniBatchKMeans(n_clusters=5000, random_state=42, batch_size=10000, n_init="auto")
kmeans.fit(X_all_scaled)

labels_b = kmeans.predict(X_b_scaled)
labels_f = kmeans.predict(X_f_scaled)
labels_m = kmeans.predict(X_m_scaled)

# Group into buckets
def group_by_label(nodes, labels, features):
    buckets = {k: ([], []) for k in range(5000)}
    for n, l, feat in zip(nodes, labels, features):
        buckets[l][0].append(n)
        buckets[l][1].append(feat)
    return buckets

b_buckets = group_by_label(banc_nodes, labels_b, X_b_scaled)
f_buckets = group_by_label(fafb_nodes, labels_f, X_f_scaled)
m_buckets = group_by_label(mcns_nodes, labels_m, X_m_scaled)

print("7. Bipartite Hungarian Alignment within K-Means buckets...")
b_f_pairs = []
b_m_pairs = []

matched_b = set()
matched_f = set()
matched_m = set()

for k in range(5000):
    b_nodes, b_feats = b_buckets[k]
    f_nodes, f_feats = f_buckets[k]
    m_nodes, m_feats = m_buckets[k]
    
    if not b_nodes or not f_nodes or not m_nodes: continue
    
    # BANC vs FAFB
    b_f = np.array(b_feats); f_f = np.array(f_feats)
    # L1 distance matrix
    cost_bf = np.sum(np.abs(b_f[:, None, :] - f_f[None, :, :]), axis=2)
            
    row_ind, col_ind = linear_sum_assignment(cost_bf)
    for r, c in zip(row_ind, col_ind):
        b, f = b_nodes[r], f_nodes[c]
        if b not in matched_b and f not in matched_f:
            b_f_pairs.append((b, f))
            matched_b.add(b); matched_f.add(f)
            
    # BANC vs MCNS
    m_f = np.array(m_feats)
    cost_bm = np.sum(np.abs(b_f[:, None, :] - m_f[None, :, :]), axis=2)
            
    row_ind, col_ind = linear_sum_assignment(cost_bm)
    for r, c in zip(row_ind, col_ind):
        b, m = b_nodes[r], m_nodes[c]
        if m not in matched_m:
            b_m_pairs.append((b, m))
            matched_m.add(m)

print("8. Merging into Triplet Pool...")
df_bf = pd.DataFrame(b_f_pairs, columns=['BANC', 'FAFB'])
df_bm = pd.DataFrame(b_m_pairs, columns=['BANC', 'MCNS'])

triplets = pd.merge(df_bf, df_bm, on='BANC', how='inner')

# Deduplicate strictly
triplets = triplets.drop_duplicates(subset=['BANC'])
triplets = triplets.drop_duplicates(subset=['FAFB'])
triplets = triplets.drop_duplicates(subset=['MCNS'])

print(f"Final K-Means Triplet Pool Size: {len(triplets):,}")
triplets.to_csv('kmeans_triplets.csv', index=False)
print(f"Saved to kmeans_triplets.csv. Total time: {time.time()-t_start:.0f}s")
