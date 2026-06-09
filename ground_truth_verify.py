"""
INDEPENDENT GROUND-TRUTH VERIFIER v2
Written from scratch - no code reuse from any previous script.
Checks ALL 6 directions of isomorphism + bijection + self-loops.
"""
import pandas as pd
import sys

print("=" * 65)
print("  GROUND-TRUTH INDEPENDENT VERIFICATION (v2)")
print("=" * 65)

# Load submission
fname = 'submission_CONNECTED_9788.csv'
df = pd.read_csv(fname, dtype=str)
N = len(df)
print(f"\nFile: {fname}")
print(f"Rows: {N}")

# ── TEST 1: Column check ────────────────────────────────────────
cols = list(df.columns)
assert cols == ['BANC', 'FAFB', 'MCNS'], f"Bad columns: {cols}"
print("\nTEST 1 - Columns: ['BANC', 'FAFB', 'MCNS'] ... OK")

# ── TEST 2: No nulls ────────────────────────────────────────────
nulls = df.isnull().sum().sum()
print(f"TEST 2 - Null values: {nulls} ... {'OK' if nulls == 0 else 'FAIL'}")

# ── TEST 3: Bijection (all unique per column) ───────────────────
banc_ids = df['BANC'].tolist()
fafb_ids = df['FAFB'].tolist()
mcns_ids = df['MCNS'].tolist()

banc_dupes = N - len(set(banc_ids))
fafb_dupes = N - len(set(fafb_ids))
mcns_dupes = N - len(set(mcns_ids))
print(f"TEST 3 - Duplicates: BANC={banc_dupes}, FAFB={fafb_dupes}, MCNS={mcns_dupes} ... ", end="")
if banc_dupes == 0 and fafb_dupes == 0 and mcns_dupes == 0:
    print("OK")
else:
    print("FAIL")
    sys.exit(1)

# ── Build mappings (row-based) ──────────────────────────────────
# Row i says: BANC[i] <-> FAFB[i] <-> MCNS[i]
banc_to_row = {}
fafb_to_row = {}
mcns_to_row = {}
for i in range(N):
    banc_to_row[banc_ids[i]] = i
    fafb_to_row[fafb_ids[i]] = i
    mcns_to_row[mcns_ids[i]] = i

# ── Load raw edge lists ─────────────────────────────────────────
print("\nLoading edge lists...")
banc_el = pd.read_csv('banc_626_edge_list.csv', header=None, names=['s', 't'], dtype=str)
fafb_el = pd.read_csv('fafb_783_edge_list.csv', header=None, names=['s', 't'], dtype=str)
mcns_el = pd.read_csv('mcns_0.9_edge_list.csv', header=None, names=['s', 't'], dtype=str)

# ── Extract internal edges as ROW-INDEX pairs ───────────────────
# This is the key: we convert everything to row indices so we can
# directly compare across datasets.
print("Extracting internal edges as row-index pairs...")

banc_node_set = set(banc_ids)
fafb_node_set = set(fafb_ids)
mcns_node_set = set(mcns_ids)

# BANC internal edges -> set of (row_i, row_j)
banc_edges = set()
for s, t in zip(banc_el['s'], banc_el['t']):
    if s in banc_to_row and t in banc_to_row:
        banc_edges.add((banc_to_row[s], banc_to_row[t]))

# FAFB internal edges -> set of (row_i, row_j)
fafb_edges = set()
for s, t in zip(fafb_el['s'], fafb_el['t']):
    if s in fafb_to_row and t in fafb_to_row:
        fafb_edges.add((fafb_to_row[s], fafb_to_row[t]))

# MCNS internal edges -> set of (row_i, row_j)
mcns_edges = set()
for s, t in zip(mcns_el['s'], mcns_el['t']):
    if s in mcns_to_row and t in mcns_to_row:
        mcns_edges.add((mcns_to_row[s], mcns_to_row[t]))

print(f"Internal edges (row-index): BANC={len(banc_edges)}, FAFB={len(fafb_edges)}, MCNS={len(mcns_edges)}")

# ── TEST 4: Perfect isomorphism ─────────────────────────────────
# Because we converted ALL edges to the SAME row-index space,
# isomorphism means: banc_edges == fafb_edges == mcns_edges
# This is the simplest, most bulletproof check possible.

print("\nTEST 4 - ISOMORPHISM (all 3 edge sets must be identical in row-index space):")

bf_match = (banc_edges == fafb_edges)
bm_match = (banc_edges == mcns_edges)
fm_match = (fafb_edges == mcns_edges)

print(f"  BANC == FAFB? {bf_match}")
print(f"  BANC == MCNS? {bm_match}")
print(f"  FAFB == MCNS? {fm_match}")

if bf_match and bm_match and fm_match:
    print(f"\n  ALL THREE EDGE SETS ARE IDENTICAL.")
    print(f"  {len(banc_edges)} edges perfectly preserved across all 3 connectomes.")
    print(f"\n  >>> PERFECT DIRECTED ISOMORPHISM: CONFIRMED <<<")
else:
    # Show exactly what differs
    only_b = banc_edges - fafb_edges - mcns_edges
    only_f = fafb_edges - banc_edges - mcns_edges
    only_m = mcns_edges - banc_edges - fafb_edges
    in_bf_not_m = (banc_edges & fafb_edges) - mcns_edges
    in_bm_not_f = (banc_edges & mcns_edges) - fafb_edges
    in_fm_not_b = (fafb_edges & mcns_edges) - banc_edges

    print(f"\n  DIFFERENCES FOUND:")
    print(f"    Only in BANC: {len(only_b)}")
    print(f"    Only in FAFB: {len(only_f)}")
    print(f"    Only in MCNS: {len(only_m)}")
    print(f"    In BANC+FAFB but not MCNS: {len(in_bf_not_m)}")
    print(f"    In BANC+MCNS but not FAFB: {len(in_bm_not_f)}")
    print(f"    In FAFB+MCNS but not BANC: {len(in_fm_not_b)}")
    print(f"\n  >>> ISOMORPHISM BROKEN <<<")

# ── TEST 5: Self-loops ──────────────────────────────────────────
banc_self = sum(1 for i, j in banc_edges if i == j)
fafb_self = sum(1 for i, j in fafb_edges if i == j)
mcns_self = sum(1 for i, j in mcns_edges if i == j)
print(f"\nTEST 5 - Self-loops: BANC={banc_self}, FAFB={fafb_self}, MCNS={mcns_self}")

# ── TEST 6: Connected nodes ─────────────────────────────────────
connected = set()
for i, j in banc_edges:
    connected.add(i); connected.add(j)
print(f"\nTEST 6 - Nodes with >= 1 edge: {len(connected)} / {N}")
if len(connected) == N:
    print("  Every node participates in at least 1 edge. OK")
else:
    print(f"  WARNING: {N - len(connected)} nodes have 0 edges")

print("\n" + "=" * 65)
print(f"  FINAL VERDICT: N={N}, Edges={len(banc_edges)}")
print("=" * 65)
