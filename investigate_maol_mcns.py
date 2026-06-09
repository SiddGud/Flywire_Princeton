"""
MAOL vs MCNS Deep Investigation
=================================
We found that 51,613 out of 51,668 MAOL neurons (99.9%) share the SAME
integer IDs with MCNS neurons. This could mean:

Hypothesis A: MAOL neurons ARE physically the same neurons as MCNS optic lobe neurons
              (same fly, same dataset, just MAOL is a subvolume of MCNS)
              => Edge IDs should heavily overlap too

Hypothesis B: The ID collision is coincidental (both just use small integers starting ~10000)
              => Edge structure would be totally different

This script figures out which hypothesis is correct by checking edge overlap.
"""

import pandas as pd
import time

PY = "C:\\Users\\sahaj\\AppData\\Local\\Programs\\Python\\Python311\\python.exe"

print("Loading MAOL and MCNS edge lists...")

t0 = time.time()
maol = pd.read_csv("maol_1.1_edge_list.csv")
maol.columns = ["src", "tgt"]
maol = maol[maol["src"] != maol["tgt"]]          # remove self-loops
maol_edges = set(zip(maol["src"], maol["tgt"]))  # set of (src, tgt) tuples
print(f"MAOL loaded: {len(maol):,} edges in {time.time()-t0:.1f}s")

t1 = time.time()
mcns = pd.read_csv("mcns_0.9_edge_list.csv")
mcns.columns = ["src", "tgt"]
mcns = mcns[mcns["src"] != mcns["tgt"]]
mcns_edges = set(zip(mcns["src"], mcns["tgt"]))
print(f"MCNS loaded: {len(mcns):,} edges in {time.time()-t1:.1f}s")

# ─────────────────────────────────────────────
# 1. Check: shared neuron IDs
# ─────────────────────────────────────────────
maol_nodes = set(maol["src"]) | set(maol["tgt"])
mcns_nodes = set(mcns["src"]) | set(mcns["tgt"])
shared_nodes = maol_nodes & mcns_nodes

print(f"\n{'='*55}")
print(f"  NODE (Neuron) ID Overlap:")
print(f"  MAOL nodes:          {len(maol_nodes):>8,}")
print(f"  MCNS nodes:          {len(mcns_nodes):>8,}")
print(f"  Shared node IDs:     {len(shared_nodes):>8,}  ({100*len(shared_nodes)/len(maol_nodes):.1f}% of MAOL)")

# ─────────────────────────────────────────────
# 2. Check: shared edges (same src→tgt pair)
# ─────────────────────────────────────────────
shared_edges = maol_edges & mcns_edges

print(f"\n  EDGE Overlap:")
print(f"  MAOL edges:          {len(maol_edges):>8,}")
print(f"  MCNS edges:          {len(mcns_edges):>8,}")
print(f"  Shared edges:        {len(shared_edges):>8,}  ({100*len(shared_edges)/len(maol_edges):.1f}% of MAOL edges)")

# ─────────────────────────────────────────────
# 3. For shared nodes: how many MAOL edges are also in MCNS?
#    (This isolates the threshold effect)
# ─────────────────────────────────────────────
print(f"\n{'='*55}")
print(f"  WITHIN shared nodes: edges that appear in both...")

# MAOL edges where BOTH endpoints are in shared_nodes
maol_internal = {(s, t) for s, t in maol_edges if s in shared_nodes and t in shared_nodes}
# MCNS edges where BOTH endpoints are in shared_nodes
mcns_internal  = {(s, t) for s, t in mcns_edges if s in shared_nodes and t in shared_nodes}
both_internal  = maol_internal & mcns_internal

print(f"  MAOL internal edges: {len(maol_internal):>8,}  (edges between shared nodes in MAOL)")
print(f"  MCNS internal edges: {len(mcns_internal):>8,}  (edges between shared nodes in MCNS)")
print(f"  Edges in BOTH:       {len(both_internal):>8,}  ({100*len(both_internal)/max(1,len(maol_internal)):.1f}% of MAOL internal)")

# ─────────────────────────────────────────────
# 4. Interpret the results
# ─────────────────────────────────────────────
print(f"\n{'='*55}")
print(f"  INTERPRETATION:")

edge_overlap_pct = 100 * len(both_internal) / max(1, len(maol_internal))

if edge_overlap_pct > 80:
    print(f"  ✅ HYPOTHESIS A CONFIRMED: MAOL IS a physical subset of MCNS!")
    print(f"     Same fly, same neurons — MAOL just has lower threshold (1+ vs 5+)")
    print(f"     The {100-edge_overlap_pct:.1f}% of MAOL edges NOT in MCNS = weak 1-4 synapse connections")
    print(f"\n  STRATEGIC IMPLICATION:")
    print(f"     MAOL + MCNS matching is nearly trivial by ID for strong edges!")
    print(f"     We can restrict MAOL to edges that also appear in MCNS")
    print(f"     → The MCNS-restricted MAOL subgraph IS a valid common subgraph")
elif edge_overlap_pct > 30:
    print(f"  ⚠ PARTIAL MATCH: Moderate edge overlap ({edge_overlap_pct:.1f}%)")
    print(f"     The threshold difference (1+ vs 5+) causes significant divergence")
    print(f"     MAOL+MCNS pairing is tricky but not impossible")
else:
    print(f"  ❌ HYPOTHESIS B: ID collision is coincidental. Only {edge_overlap_pct:.1f}% overlap.")
    print(f"     MAOL and MCNS are truly different datasets — avoid this pairing")

# ─────────────────────────────────────────────
# 5. If Hypothesis A is true: what's the usable common subgraph size?
# ─────────────────────────────────────────────
if len(both_internal) > 0:
    # Nodes involved in matching edges
    matching_nodes = set()
    for s, t in both_internal:
        matching_nodes.add(s)
        matching_nodes.add(t)
    print(f"\n  USABLE MATCHING NEURONS:")
    print(f"     Neurons involved in matching edges: {len(matching_nodes):,}")
    print(f"     These form the seed of our common subgraph!")

    # Show sample matching edges
    print(f"\n  Sample matching edges (MAOL ID == MCNS ID):")
    for i, (s, t) in enumerate(list(both_internal)[:5]):
        print(f"     {s} → {t}")

print(f"\n{'='*55}")
print(f"  Investigation complete!")
