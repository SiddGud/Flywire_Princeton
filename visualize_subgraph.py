import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import pyarrow.feather as feather

# Load submission nodes
print("Loading subgraph nodes...")
sub = pd.read_csv('submission_CT_FINE_5092.csv', dtype=str)
banc_nodes = set(sub['BANC'].tolist())

# Load edges
print("Loading edges...")
banc_df = pd.read_csv('banc_626_edge_list.csv', header=None, names=['src','tgt'], dtype=str)
mask = banc_df['src'].isin(banc_nodes) & banc_df['tgt'].isin(banc_nodes)
subgraph_edges = banc_df[mask]

# Create graph
print("Creating NetworkX graph...")
G = nx.from_pandas_edgelist(subgraph_edges, 'src', 'tgt', create_using=nx.DiGraph())
print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

# Add missing nodes (isolated nodes without edges)
for node in banc_nodes:
    if node not in G:
        G.add_node(node)

# Load metadata for coloring
print("Loading metadata for node coloring...")
meta = feather.read_feather('banc_888_meta.feather')
meta['root_626'] = meta['root_626'].astype(str)
node_meta = meta[meta['root_626'].isin(banc_nodes)]
cell_types = dict(zip(node_meta['root_626'], node_meta['cell_type']))

# Color mapping based on cell type keywords
colors = []
for node in G.nodes():
    ct = str(cell_types.get(node, '')).lower()
    if 'lc' in ct: colors.append('#e74c3c') # Red for visual projection
    elif 'kc' in ct: colors.append('#3498db') # Blue for Kenyon cells
    elif 'mbon' in ct: colors.append('#9b59b6') # Purple for MBON
    elif 'dn' in ct: colors.append('#2ecc71') # Green for Descending
    else: colors.append('#95a5a6') # Gray for other interneurons

# Draw graph
print("Drawing graph (this may take a minute)...")
plt.figure(figsize=(14, 14), facecolor='white')
# Using spring layout for a nice organic look
pos = nx.spring_layout(G, k=0.3, iterations=50, seed=42)

# Draw edges and nodes
nx.draw_networkx_nodes(G, pos, node_size=10, node_color=colors, alpha=0.9, linewidths=0)
nx.draw_networkx_edges(G, pos, alpha=0.4, width=0.5, arrowsize=3, edge_color='#bdc3c7')

# Legend
import matplotlib.patches as mpatches
legend_patches = [
    mpatches.Patch(color='#e74c3c', label='Lobula Columnar (VPN)'),
    mpatches.Patch(color='#3498db', label='Kenyon Cells (MB)'),
    mpatches.Patch(color='#9b59b6', label='Mushroom Body Output (MBON)'),
    mpatches.Patch(color='#2ecc71', label='Descending Neurons (DN)'),
    mpatches.Patch(color='#95a5a6', label='Interneurons (LH, etc.)')
]
plt.legend(handles=legend_patches, loc='upper right', frameon=False, fontsize=12)
plt.title(f"Visual-Associative Isomorphic Subgraph (N = {len(banc_nodes):,})", fontsize=18)

plt.axis('off')
plt.tight_layout()

out_path = r'C:\Users\sahaj\.gemini\antigravity-ide\brain\5917e1ca-d626-4665-a6bb-ecf93b4152b6\subgraph_network.png'
plt.savefig(out_path, dpi=300, bbox_inches='tight')
print(f"Graph saved to {out_path}!")
