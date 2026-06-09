# ============================================================
# FlyWire 2026 Challenge — Full Pipeline
# Run this in Google Colab (GPU runtime recommended: A100)
# ============================================================
# SETUP: Before running:
# 1. Upload all 5 CSV files to your Google Drive under:
#    My Drive/flywire_challenge/
# 2. Set runtime to GPU (Runtime → Change runtime type → A100)
# 3. Run all cells top to bottom
# ============================================================

# ── Cell 1: Mount Drive & Install Libraries ──────────────────
from google.colab import drive
drive.mount('/content/drive')

!pip install -q networkx scipy numpy pandas matplotlib seaborn
!pip install -q python-igraph  # Fast C-based graph library

import os
DATA_DIR = "/content/drive/MyDrive/flywire_challenge/"
print("Files in Drive folder:")
print(os.listdir(DATA_DIR))

# ── Cell 2: Load All 5 Edge Lists ────────────────────────────
import pandas as pd
import numpy as np
import scipy.sparse as sp
import time

DATASETS = {
    "FAFB": "fafb_783_edge_list.csv",
    "BANC": "banc_626_edge_list.csv",
    "MANC": "manc_1.2.1_edge_list.csv",
    "MAOL": "maol_1.1_edge_list.csv",
    "MCNS": "mcns_0.9_edge_list.csv",
}

dfs = {}
for name, fname in DATASETS.items():
    t = time.time()
    df = pd.read_csv(DATA_DIR + fname)
    df.columns = ["src", "tgt"]
    df = df[df["src"] != df["tgt"]].drop_duplicates()
    dfs[name] = df
    print(f"{name}: {len(df):,} edges loaded in {time.time()-t:.1f}s")

# ── Cell 3: Build Compact Integer-Indexed Sparse Matrices ────
# Converts huge neuron IDs to compact integer indices 0..N-1
# This is essential for memory efficiency and matrix operations

def build_sparse_graph(df, node_index=None):
    """
    Build a scipy CSR sparse adjacency matrix.
    Returns: (matrix, node_to_idx dict, idx_to_node list)
    """
    nodes = sorted(set(df["src"]) | set(df["tgt"]))
    if node_index is None:
        node_to_idx = {n: i for i, n in enumerate(nodes)}
    else:
        node_to_idx = node_index
        nodes = list(node_index.keys())

    N = len(node_to_idx)
    rows = df["src"].map(node_to_idx).dropna().astype(int)
    cols = df["tgt"].map(node_to_idx).dropna().astype(int)
    # Keep only valid mappings
    valid = rows.notna() & cols.notna()
    data = np.ones(valid.sum(), dtype=np.int8)

    mat = sp.csr_matrix((data, (rows[valid], cols[valid])), shape=(N, N))
    return mat, node_to_idx, nodes

graphs = {}
for name, df in dfs.items():
    mat, n2i, i2n = build_sparse_graph(df)
    graphs[name] = {"mat": mat, "n2i": n2i, "i2n": i2n, "df": df}
    nnz = mat.nnz
    print(f"{name}: {mat.shape[0]:,} nodes, {nnz:,} edges, "
          f"memory: {mat.data.nbytes / 1e6:.1f}MB")

# ── Cell 4: Download BANC Metadata (Cross-Dataset Match IDs) ─
# The BANC dataset has pre-computed matches to FAFB neurons!
# This is the KEY SHORTCUT that makes the challenge tractable.

# Try to download from FlyWire Codex directly
# (You need to be logged in to Codex — use your Google account token)
# Alternative: Download manually from Harvard Dataverse and upload to Drive

# Method A: Try direct Codex API (may require auth token)
import urllib.request, json

BANC_METADATA_PATH = DATA_DIR + "banc_metadata.csv"

if not os.path.exists(BANC_METADATA_PATH):
    print("BANC metadata not found in Drive.")
    print("Please download from: https://codex.flywire.ai/api/download")
    print("Look for 'BANC cell info' or 'annotations' table")
    print("Then upload to your Google Drive flywire_challenge folder")
    print("\nAlternatively, install CAVEclient to fetch programmatically:")
    print("  !pip install caveclient")
    print("  from caveclient import CAVEclient")
    print("  c = CAVEclient('brain_and_nerve_cord')")
    print("  df = c.annotation.query_table('neuron_information')")
else:
    banc_meta = pd.read_csv(BANC_METADATA_PATH)
    print(f"BANC metadata loaded: {len(banc_meta):,} rows")
    print(f"Columns: {list(banc_meta.columns)}")

    # Look for cross-dataset match columns
    match_cols = [c for c in banc_meta.columns if 'match' in c.lower() or 'fafb' in c.lower()]
    print(f"Cross-dataset match columns: {match_cols}")

# ── Cell 5: Core Strategy — FAFB + BANC + MCNS ───────────────
# Using BANC as the bridge:
#   - BANC has fafb_783_match_id → tells us BANC↔FAFB pairs
#   - Use cell_type labels to extend to MCNS
#   - Find common induced subgraph among those matched neurons

# IF you have BANC metadata with fafb_783_match_id:
def build_seed_mapping_from_banc_metadata(banc_meta):
    """
    Extract the pre-computed BANC↔FAFB neuron correspondence.
    Returns a dict: {banc_id: fafb_id} for all confirmed matches.
    """
    # Try common column name patterns
    banc_id_col = None
    fafb_col = None

    for col in banc_meta.columns:
        if 'root' in col.lower() and 'id' in col.lower():
            banc_id_col = col
        if 'fafb' in col.lower() and 'match' in col.lower():
            fafb_col = col

    if banc_id_col and fafb_col:
        valid = banc_meta[[banc_id_col, fafb_col]].dropna()
        mapping = dict(zip(valid[banc_id_col], valid[fafb_col]))
        print(f"Found {len(mapping):,} BANC→FAFB confirmed matches!")
        return mapping
    else:
        print(f"Columns found: {list(banc_meta.columns)}")
        print("Could not find match ID columns automatically — check column names above")
        return {}

# ── Cell 6: Alternative — MAOL + MCNS Direct Match (by ID) ──
# We discovered MAOL neurons ARE the same physical neurons as MCNS optic lobe
# 99.9% of MAOL node IDs appear in MCNS
# This gives us a FREE matching for 51,613 neurons!

maol_nodes = set(dfs["MAOL"]["src"]) | set(dfs["MAOL"]["tgt"])
mcns_nodes = set(dfs["MCNS"]["src"]) | set(dfs["MCNS"]["tgt"])
shared_maol_mcns = maol_nodes & mcns_nodes

print(f"MAOL∩MCNS shared neurons: {len(shared_maol_mcns):,}")

# Build edge sets for these shared neurons
maol_edges = set(zip(dfs["MAOL"]["src"], dfs["MAOL"]["tgt"]))
mcns_edges = set(zip(dfs["MCNS"]["src"], dfs["MCNS"]["tgt"]))

# Edges between shared neurons in MAOL
maol_internal = frozenset((s,t) for s,t in maol_edges if s in shared_maol_mcns and t in shared_maol_mcns)
# Edges between shared neurons in MCNS
mcns_internal = frozenset((s,t) for s,t in mcns_edges if s in shared_maol_mcns and t in shared_maol_mcns)

# Conflict: edge in one but not the other
conflicts_maol = maol_internal - mcns_internal  # weak MAOL edges (1-4 syn)
conflicts_mcns = mcns_internal - maol_internal  # MCNS-only edges

# Nodes involved in any conflict
conflict_nodes = set()
for s, t in conflicts_maol: conflict_nodes.add(s); conflict_nodes.add(t)
for s, t in conflicts_mcns: conflict_nodes.add(s); conflict_nodes.add(t)

safe_nodes_maol_mcns = shared_maol_mcns - conflict_nodes
print(f"Conflict-free neurons (MAOL∩MCNS perfect match): {len(safe_nodes_maol_mcns):,}")

# These safe_nodes have IDENTICAL edge structure in MAOL and MCNS
# This is already a valid common induced subgraph between MAOL and MCNS!
safe_edges = {(s,t) for s,t in maol_internal if s in safe_nodes_maol_mcns and t in safe_nodes_maol_mcns}
print(f"Edges in this common subgraph: {len(safe_edges):,}")
print(f"\n🎯 MAOL+MCNS common induced subgraph: {len(safe_nodes_maol_mcns):,} neurons, {len(safe_edges):,} edges")
print(f"   Now we need to extend this to a 3rd dataset (FAFB or BANC)!")

# ── Cell 7: Extend to 3rd Dataset via Cell Type Matching ──────
# The safe_nodes are optic lobe neurons in both MAOL and MCNS
# FAFB also contains optic lobe neurons
# Use cell type labels to find FAFB neurons that match these

# This requires metadata — will add once BANC/Codex metadata is available
# For now, compute degree signatures as a proxy for cell type

def compute_degree_signature(df, node_set):
    """
    For each node, compute (in_degree, out_degree) as a structural fingerprint.
    Nodes with same fingerprint are candidates for matching.
    """
    in_deg  = df[df["tgt"].isin(node_set)].groupby("tgt").size().rename("in_deg")
    out_deg = df[df["src"].isin(node_set)].groupby("src").size().rename("out_deg")
    sig = pd.concat([in_deg, out_deg], axis=1).fillna(0).astype(int)
    sig["signature"] = list(zip(sig["in_deg"], sig["out_deg"]))
    return sig

print("\nComputing degree signatures for MAOL safe nodes...")
maol_sig = compute_degree_signature(dfs["MAOL"], safe_nodes_maol_mcns)
print(f"Signatures computed for {len(maol_sig):,} neurons")
print("\nTop 10 most common (in_deg, out_deg) patterns:")
print(maol_sig["signature"].value_counts().head(10))

print("\n\n=== NEXT STEPS ===")
print("1. Download BANC metadata → get fafb_783_match_id links")
print("2. Use those links + degree signatures to find FAFB matches")
print("3. Run McSplit on the pruned candidate set")
print("4. Verify isomorphism and generate submission CSV")
