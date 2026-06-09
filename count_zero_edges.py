import pandas as pd

sub = pd.read_csv('submission_ITER_GROW_13081.csv', dtype=str)
banc_df = pd.read_csv('banc_626_edge_list.csv', header=None, names=['src','tgt'], dtype=str)
fafb_df = pd.read_csv('fafb_783_edge_list.csv', header=None, names=['src','tgt'], dtype=str)
mcns_df = pd.read_csv('mcns_0.9_edge_list.csv', header=None, names=['src','tgt'], dtype=str)

b_nodes = set(sub['BANC']); f_nodes = set(sub['FAFB']); m_nodes = set(sub['MCNS'])

# Find which nodes participate in at least 1 internal edge
b_connected = set()
for s, t in zip(banc_df['src'], banc_df['tgt']):
    if s in b_nodes and t in b_nodes:
        b_connected.add(s)
        b_connected.add(t)

f_connected = set()
for s, t in zip(fafb_df['src'], fafb_df['tgt']):
    if s in f_nodes and t in f_nodes:
        f_connected.add(s)
        f_connected.add(t)

m_connected = set()
for s, t in zip(mcns_df['src'], mcns_df['tgt']):
    if s in m_nodes and t in m_nodes:
        m_connected.add(s)
        m_connected.add(t)

print(f"Total nodes: {len(sub)}")
print(f"Nodes with >= 1 internal edge in BANC: {len(b_connected)}")
print(f"Nodes with >= 1 internal edge in FAFB: {len(f_connected)}")
print(f"Nodes with >= 1 internal edge in MCNS: {len(m_connected)}")

# Nodes with 0 internal edges in ALL datasets
b2f = dict(zip(sub['BANC'], sub['FAFB']))
b2m = dict(zip(sub['BANC'], sub['MCNS']))

zero_edge_count = 0
zero_edge_banc = []
for _, row in sub.iterrows():
    b, f, m = row['BANC'], row['FAFB'], row['MCNS']
    if b not in b_connected and f not in f_connected and m not in m_connected:
        zero_edge_count += 1
        zero_edge_banc.append(b)

print(f"\nNodes with 0 internal edges in ALL 3 datasets: {zero_edge_count}")
print(f"Nodes with >= 1 internal edge in at least 1 dataset: {len(sub) - zero_edge_count}")

# What if we remove all zero-edge nodes?
clean = sub[~sub['BANC'].isin(set(zero_edge_banc))].reset_index(drop=True)
print(f"\nIf we remove all 0-edge nodes: N = {len(clean)}")
