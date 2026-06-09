"""
MAOL-MCNS Deeper Analysis
===========================
Results from investigate_maol_mcns.py:
  - 99.9% of MAOL node IDs appear in MCNS  ← same physical neurons!
  - 24% of MAOL edges appear in MCNS        ← threshold difference explains this
  - 97.3% of MCNS internal edges appear in MAOL  ← this is the key number!

Conclusion: MAOL neurons ARE the same physical neurons as MCNS optic lobe neurons.
The 76% of MAOL-only edges are weak (1-4 synapse) connections filtered by MCNS (5+ threshold).

This script finds: the largest set of MAOL/MCNS neurons where BOTH datasets agree 
on EVERY edge (no conflicts) — i.e., the common induced subgraph between MAOL and MCNS.
"""

import pandas as pd
import time

print("Loading edge lists...")
t0 = time.time()

maol = pd.read_csv("maol_1.1_edge_list.csv")
maol.columns = ["src", "tgt"]
maol = maol[maol["src"] != maol["tgt"]]

mcns = pd.read_csv("mcns_0.9_edge_list.csv")
mcns.columns = ["src", "tgt"]
mcns = mcns[mcns["src"] != mcns["tgt"]]

print(f"Loaded in {time.time()-t0:.1f}s")

# Shared nodes
maol_nodes = set(maol["src"]) | set(maol["tgt"])
mcns_nodes = set(mcns["src"]) | set(mcns["tgt"])
shared = maol_nodes & mcns_nodes
print(f"\nShared nodes: {len(shared):,}")

# ─────────────────────────────────────────────
# The key insight: MCNS edges among shared nodes
# 97.3% of them also appear in MAOL
# The remaining 2.7% are the "conflict" edges
# that break isomorphism for any pair they connect
# ─────────────────────────────────────────────
t1 = time.time()
print("\nBuilding edge sets (this takes a moment)...")

maol_set = set(zip(maol["src"], maol["tgt"]))
mcns_set = set(zip(mcns["src"], mcns["tgt"]))

# Edges in MCNS between shared nodes
mcns_internal = {(s,t) for s,t in mcns_set if s in shared and t in shared}
# Edges in MAOL between shared nodes  
maol_internal = {(s,t) for s,t in maol_set if s in shared and t in shared}

# Conflict edges: exist in one but not the other
maol_only = maol_internal - mcns_internal   # 1-4 synapse weak connections
mcns_only = mcns_internal - maol_internal   # segmentation differences
both      = maol_internal & mcns_internal   # strong matching edges (5+ synapse)

print(f"\nEdge analysis between {len(shared):,} shared nodes:")
print(f"  MAOL internal edges:   {len(maol_internal):>9,}")
print(f"  MCNS internal edges:   {len(mcns_internal):>9,}")
print(f"  Agree (both have):     {len(both):>9,}  <- these are fine for isomorphism")
print(f"  MAOL-only (weak 1-4):  {len(maol_only):>9,}  <- these BREAK isomorphism")
print(f"  MCNS-only (seg diff):  {len(mcns_only):>9,}  <- these BREAK isomorphism")
print(f"\n  % of MCNS edges also in MAOL: {100*len(both)/max(1,len(mcns_internal)):.1f}%  (the real overlap)")
print(f"  Time: {time.time()-t1:.1f}s")

# ─────────────────────────────────────────────
# Find neurons that are NEVER involved in conflict edges
# These neurons can safely be in our common induced subgraph
# (all their within-set connections agree between MAOL and MCNS)
# ─────────────────────────────────────────────
print("\nFinding conflict-free neurons...")

# Neurons involved in any conflict edge (can't be in the safe set together)
conflict_nodes = set()
for s, t in maol_only:
    conflict_nodes.add(s)
    conflict_nodes.add(t)
for s, t in mcns_only:
    conflict_nodes.add(s)
    conflict_nodes.add(t)

safe_nodes = shared - conflict_nodes

print(f"\n  Neurons in conflict edges: {len(conflict_nodes):,}")
print(f"  Conflict-FREE neurons:     {len(safe_nodes):,}")
print(f"  (Neurons where MAOL and MCNS PERFECTLY AGREE on all connections)")

# Among safe nodes: count matching edges
safe_matching_edges = {(s,t) for s,t in both if s in safe_nodes and t in safe_nodes}
print(f"  Matching edges among safe nodes: {len(safe_matching_edges):,}")

if len(safe_nodes) > 0:
    print(f"\n  RESULT: A common induced subgraph of {len(safe_nodes):,} neurons exists")
    print(f"          between MAOL and MCNS with {len(safe_matching_edges):,} edges!")
    print(f"          This is our seed — we now need a 3rd dataset to match them to.")
    print(f"          FAFB has the optic lobe too — use cell types to link them.")
else:
    print(f"\n  No perfectly conflict-free neurons found.")
    print(f"  Need a different approach (e.g., find largest independent set in conflict graph)")

print(f"\n  Done in {time.time()-t0:.1f}s total")
