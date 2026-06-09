import pandas as pd
import pyarrow.feather as feather
import networkx as nx
import matplotlib.pyplot as plt
import os
import collections

# Auto-find the highest scoring CSV
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

if not CORE_FILE or not META_FILE:
    print("Files not found! Make sure your 19k+ csv and banc_888_meta.feather are uploaded.")
    exit()

print(f"Loading {CORE_FILE}...")
core_df = pd.read_csv(CORE_FILE, dtype=str)
banc_nodes = set(core_df['BANC'])

print(f"Loading metadata {META_FILE}...")
meta = feather.read_feather(META_FILE)
meta['root_626'] = meta['root_626'].astype(str)

# Filter metadata to only neurons in our core
core_meta = meta[meta['root_626'].isin(banc_nodes)]

print("\n" + "="*50)
print("  BIOLOGICAL ANALYSIS OF THE CORE")
print("="*50)

# 1. Cell Class Distribution
print("\n--- Top Cell Classes ---")
class_counts = core_meta['cell_class'].value_counts().head(10)
print(class_counts)

# 2. Cell Type Distribution
print("\n--- Top Specific Cell Types ---")
type_counts = core_meta['cell_type'].value_counts().head(10)
print(type_counts)

# 3. Neurotransmitter Breakdown
print("\n--- Neurotransmitters ---")
if 'nt_type' in core_meta.columns:
    nt_counts = core_meta['nt_type'].value_counts().head(5)
    print(nt_counts)
else:
    print("Neurotransmitter data not available in this metadata file.")

print("\n" + "="*50)
print("  NETWORK GRAPH VISUALIZATION & 3D MESH TARGETS")
print("="*50)

# Load edges to find hubs
edge_file = None
for root, dirs, files in os.walk('/kaggle/input'):
    for f in files:
        if 'banc_626_edge_list' in f:
            edge_file = os.path.join(root, f)
            break
if not edge_file:
    edge_file = '/kaggle/input/datasets/siddhantgudwani/dataset/banc_626_edge_list.csv'

print("Loading edge list to calculate hubs...")
edges_df = pd.read_csv(edge_file, header=None, names=['src','tgt'], dtype=str)

# Filter to internal core edges only
core_edges = edges_df[edges_df['src'].isin(banc_nodes) & edges_df['tgt'].isin(banc_nodes)]

# Calculate degrees
degree = collections.defaultdict(int)
for s, t in zip(core_edges['src'], core_edges['tgt']):
    degree[s] += 1
    degree[t] += 1

# Get top 50 hubs
top_hubs = sorted(degree.items(), key=lambda x: x[1], reverse=True)[:50]
hub_ids = set([x[0] for x in top_hubs])

# Create Subgraph
G = nx.Graph()
for s, t in zip(core_edges['src'], core_edges['tgt']):
    if s in hub_ids and t in hub_ids:
        G.add_edge(s, t)

# Visualize
plt.figure(figsize=(12, 12))
pos = nx.spring_layout(G, k=0.5, seed=42)
nx.draw(G, pos, node_size=150, node_color='skyblue', edge_color='gray', alpha=0.7, with_labels=False)
plt.title("Top 50 Core Hubs of the Conserved Neuronal Circuit", fontsize=16)
plt.savefig('/kaggle/working/circuit_visualization.png', dpi=300, bbox_inches='tight')
print("Network graph saved to /kaggle/working/circuit_visualization.png")

print("\n--- TOP 10 ROOT IDs FOR CODEX 3D MESHES ---")
print("Paste these IDs into the FlyWire Codex UI to capture your 3D brain images:")
for i, (node_id, deg) in enumerate(top_hubs[:10]):
    cell_info = core_meta[core_meta['root_626'] == node_id]
    c_type = cell_info['cell_type'].values[0] if len(cell_info) > 0 and pd.notna(cell_info['cell_type'].values[0]) else "Unknown"
    print(f"{i+1}. BANC ID: {node_id} (Type: {c_type}, Internal Connections: {deg})")
