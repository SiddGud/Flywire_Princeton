# Local Visualization Guide

This document explains how to reproduce the two key visualizations locally after cloning the repository.

---

## 1. Network Graph (Force-Directed 2D/3D)

The interactive 3D network visualization of the 14,484-node subgraph is hosted at:
**[https://siddgud.github.io/14k_interactive_3d_network/14k_interactive_3d_network.html](https://siddgud.github.io/14k_interactive_3d_network/14k_interactive_3d_network.html)**

To generate a local version from `network.csv`:

```bash
git clone https://github.com/SiddGud/Flywire_Princeton.git
cd Flywire_Princeton
pip install pandas networkx plotly
python - <<'EOF'
import pandas as pd
import networkx as nx
import plotly.graph_objects as go

# Load matched triplets
df = pd.read_csv("network.csv", dtype=str)

# Build graph using BANC neuron IDs as nodes
# Edges from the BANC edge list (requires fafb_783_edge_list.csv locally)
# For a quick local demo: build from matched node set only
nodes = df["BANC"].tolist()
G = nx.DiGraph()
G.add_nodes_from(nodes)

# If you have the edge list locally:
# edges_df = pd.read_csv("banc_626_edge_list.csv", header=None, names=["src","tgt"], dtype=str)
# node_set = set(nodes)
# for _, row in edges_df.iterrows():
#     if row["src"] in node_set and row["tgt"] in node_set:
#         G.add_edge(row["src"], row["tgt"])

print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
EOF
```

The full interactive visualization was generated on Kaggle using the `kaggle_advanced_network_visualizer.py` script, which builds a 3D force-directed layout using node positions computed from the BANC edge list and renders it with Plotly.

---

## 2. 3D Mesh View (Neuroglancer / FlyWire Codex Platform)

The 3D mesh renderings (shown in `figures/`) were produced using the FlyWire Neuroglancer platform. To reproduce:

### Step 1: Extract FAFB neuron IDs from `network.csv`

```python
import pandas as pd
df = pd.read_csv("network.csv", dtype=str)
fafb_ids = df["FAFB"].tolist()
print(",".join(fafb_ids[:100]))  # first 100 for a test
```

### Step 2: Edit the Neuroglancer JSON state

The Neuroglancer viewer at `https://codex.flywire.ai` uses JSON state files to define which neurons are rendered. A template state file is structured as:

```json
{
  "layers": [
    {
      "type": "segmentation",
      "source": "precomputed://gs://flywire_v141_m783",
      "segments": ["720575940603231916", "720575940622670240", "..."],
      "name": "FAFB neurons"
    }
  ],
  "navigation": {
    "pose": { "position": { "voxelCoordinates": [34000, 40000, 3000] } },
    "zoomFactor": 8
  },
  "layout": "3d"
}
```

Replace the `"segments"` array with your FAFB IDs from `network.csv`. The `neuroglancer_14484_full.json` file (generated locally during the project) contained the complete list of all 14,484 FAFB IDs used to produce the figures in this repository.

### Step 3: Load into Neuroglancer

1. Go to `https://neuroglancer-demo.appspot.com/` or `https://codex.flywire.ai`
2. Click the `{}` button (JSON state editor) in the top right
3. Paste your edited JSON
4. The 14,484 neurons will render as a 3D mesh

The screenshots in `figures/` were taken from this viewer after loading all 14,484 FAFB IDs.

---

## 3. Gephi Graph Export

The `14k_subgraph_gephi.gexf` file (generated locally, not in the repo due to size) contains the full 14,484-node subgraph in GEXF format, which can be opened in [Gephi](https://gephi.org/) for custom layout and analysis:

```python
import pandas as pd
import networkx as nx

df = pd.read_csv("network.csv", dtype=str)
# ... build graph from edge list ...
nx.write_gexf(G, "14k_subgraph_gephi.gexf")
```
<!-- repo structure finalized -->
<!-- submission complete -->
