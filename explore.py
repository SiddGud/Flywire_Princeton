"""
FlyWire Connectome Explorer
============================
Step 1: Load all 5 edge lists and understand what we're working with.

What this script does:
  1. Loads each CSV file into a pandas DataFrame
  2. Builds a directed graph (DiGraph) for each dataset
  3. Prints key statistics about each graph
  4. Saves a summary so we know what we're dealing with before analysis
"""

import pandas as pd
import networkx as nx
import time
import os

# ─────────────────────────────────────────────
# Configuration: file paths and dataset names
# ─────────────────────────────────────────────
DATASETS = {
    "FAFB": "fafb_783_edge_list.csv",
    "BANC": "banc_626_edge_list.csv",
    "MANC": "manc_1.2.1_edge_list.csv",
    "MAOL": "maol_1.1_edge_list.csv",
    "MCNS": "mcns_0.9_edge_list.csv",
}

# ─────────────────────────────────────────────
# Helper: load one edge list and build a graph
# ─────────────────────────────────────────────
def load_graph(name, filepath):
    print(f"\n{'='*50}")
    print(f"  Loading {name} from {filepath} ...")
    t0 = time.time()

    # Read the CSV
    df = pd.read_csv(filepath)
    t1 = time.time()
    print(f"  ✓ CSV loaded in {t1-t0:.1f}s  |  Rows: {len(df):,}")

    # Show the first few rows so we understand the structure
    print(f"\n  First 3 rows:")
    print(df.head(3).to_string(index=False))

    # Rename columns to standard names for easier handling
    df.columns = ["source", "target"]

    # Remove any self-loops (neuron connecting to itself — not meaningful)
    before = len(df)
    df = df[df["source"] != df["target"]]
    removed = before - len(df)
    if removed > 0:
        print(f"  ⚠ Removed {removed:,} self-loops")

    # Remove duplicate edges (same connection listed twice)
    before = len(df)
    df = df.drop_duplicates()
    removed = before - len(df)
    if removed > 0:
        print(f"  ⚠ Removed {removed:,} duplicate edges")

    # Build the directed graph
    print(f"\n  Building directed graph...")
    t2 = time.time()
    G = nx.from_pandas_edgelist(df, source="source", target="target",
                                 create_using=nx.DiGraph())
    t3 = time.time()
    print(f"  ✓ Graph built in {t3-t2:.1f}s")

    return G, df


# ─────────────────────────────────────────────
# Helper: print detailed stats about a graph
# ─────────────────────────────────────────────
def print_stats(name, G):
    print(f"\n  📊 Statistics for {name}:")
    print(f"     Neurons (nodes):          {G.number_of_nodes():>12,}")
    print(f"     Connections (edges):       {G.number_of_edges():>12,}")

    # Degree = how many connections a neuron has
    # In-degree  = how many neurons send signals TO this one
    # Out-degree = how many neurons this one sends signals TO
    in_degrees  = [d for _, d in G.in_degree()]
    out_degrees = [d for _, d in G.out_degree()]

    avg_in  = sum(in_degrees)  / len(in_degrees)
    avg_out = sum(out_degrees) / len(out_degrees)
    max_in  = max(in_degrees)
    max_out = max(out_degrees)

    print(f"     Avg in-degree  (inputs):   {avg_in:>12.1f}")
    print(f"     Max in-degree  (inputs):   {max_in:>12,}")
    print(f"     Avg out-degree (outputs):  {avg_out:>12.1f}")
    print(f"     Max out-degree (outputs):  {max_out:>12,}")

    # Density = what fraction of all possible connections actually exist?
    # 0 = no connections, 1 = every neuron connects to every other
    density = nx.density(G)
    print(f"     Graph density:             {density:>12.6f}  (very sparse is normal!)")

    # Weakly connected components = groups of neurons that are connected
    # (ignoring direction). Like "islands" in the network.
    wcc = list(nx.weakly_connected_components(G))
    print(f"     Connected components:      {len(wcc):>12,}")
    largest_wcc = max(wcc, key=len)
    print(f"     Largest component size:    {len(largest_wcc):>12,}  neurons")
    pct = 100 * len(largest_wcc) / G.number_of_nodes()
    print(f"     (that's {pct:.1f}% of all neurons in one big connected cluster)")

    return {
        "name": name,
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "avg_in_degree": round(avg_in, 2),
        "avg_out_degree": round(avg_out, 2),
        "max_in_degree": max_in,
        "max_out_degree": max_out,
        "density": round(density, 8),
        "num_components": len(wcc),
        "largest_component": len(largest_wcc),
    }


# ─────────────────────────────────────────────
# Helper: find neurons shared between datasets
# ─────────────────────────────────────────────
def check_overlap(graphs):
    """
    Check if any neuron IDs appear in multiple datasets.
    FAFB/BANC use 18-digit IDs, MANC/MAOL/MCNS use small integers —
    so there may be ID collisions even if they're different neurons!
    This is important to know before we start matching.
    """
    print(f"\n{'='*50}")
    print("  🔍 Checking neuron ID overlap between datasets...")
    print("  (This tells us if we can use IDs directly to match neurons)")

    names = list(graphs.keys())
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            n1, n2 = names[i], names[j]
            nodes1 = set(graphs[n1].nodes())
            nodes2 = set(graphs[n2].nodes())
            shared = nodes1 & nodes2
            print(f"\n  {n1} ∩ {n2}:")
            print(f"    {n1} has {len(nodes1):,} unique neuron IDs")
            print(f"    {n2} has {len(nodes2):,} unique neuron IDs")
            print(f"    Shared IDs: {len(shared):,}")
            if len(shared) > 0:
                print(f"    ⚠ WARNING: ID collision — same numbers may mean different neurons!")
                print(f"    Sample shared IDs: {list(shared)[:5]}")
            else:
                print(f"    ✓ No ID collision — these use completely different ID spaces")


# ─────────────────────────────────────────────
# Main: run everything
# ─────────────────────────────────────────────
def main():
    print("\n" + "="*50)
    print("  🧠 FlyWire Connectome Data Explorer")
    print("="*50)

    graphs = {}
    summary_rows = []

    # Check that all files exist
    for name, fname in DATASETS.items():
        if not os.path.exists(fname):
            print(f"  ❌ Missing file: {fname}")
            print(f"     Make sure you're running this from: C:\\Users\\sahaj\\OneDrive\\Desktop\\neww")
            return

    # Load each dataset
    for name, fname in DATASETS.items():
        G, df = load_graph(name, fname)
        graphs[name] = G
        stats = print_stats(name, G)
        summary_rows.append(stats)

    # Print a comparison table
    print(f"\n\n{'='*50}")
    print("  📋 COMPARISON SUMMARY TABLE")
    print("="*50)
    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.set_index("name")
    print(summary_df[["nodes", "edges", "avg_in_degree", "max_in_degree",
                       "density", "num_components", "largest_component"]].to_string())

    # Save summary to CSV
    summary_df.to_csv("dataset_summary.csv")
    print(f"\n  ✓ Summary saved to dataset_summary.csv")

    # Check ID overlap (crucial for understanding how to match neurons)
    check_overlap(graphs)

    print(f"\n\n{'='*50}")
    print("  ✅ Exploration complete!")
    print("  Next step: fetch cell type metadata from FlyWire Codex")
    print("="*50)


if __name__ == "__main__":
    main()
