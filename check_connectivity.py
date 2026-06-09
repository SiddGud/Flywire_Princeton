"""
Check if submission is weakly connected.
If not, extract the Largest Weakly Connected Component (LWCC) and save it.
Weakly connected = connected when edge directions are ignored.
"""
import csv
import glob
import os
from collections import defaultdict, deque

# Find best submission file
def sort_key(x):
    base = x.replace('_CLEAN','').replace('_FINAL','')
    try: n = int(base.split('_')[-1].split('.')[0])
    except: n = 0
    bonus = 2 if ('FINAL' in x or 'CLEAN' in x) else 1
    return (n, bonus)

files = (glob.glob('submission_S3_*.csv') + glob.glob('submission_S2_*.csv') +
         glob.glob('submission_MAX_*.csv'))
if not files:
    print("No submission files found!"); exit()
files.sort(key=sort_key, reverse=True)
fname = files[0]
print(f"Checking: {fname}\n")

# Load submission
df = []
with open(fname, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        df.append({'BANC': row['BANC'], 'FAFB': row['FAFB'], 'MCNS': row['MCNS']})

N = len(df)
print(f"Total rows: {N:,}")

banc_nodes = set(r['BANC'] for r in df)

# Load BANC edge list and build undirected adjacency for weak connectivity
print("Loading BANC edges...")
adj = defaultdict(set)
with open('banc_626_edge_list.csv', 'r') as f:
    for line in f:
        s, t = line.strip().split(',')
        if s in banc_nodes and t in banc_nodes:
            adj[s].add(t)
            adj[t].add(s)  # undirected for weak connectivity check

# BFS to find all connected components
print("Finding connected components...")
visited = set()
components = []

for node in banc_nodes:
    if node not in visited:
        # BFS from this node
        comp = []
        queue = deque([node])
        visited.add(node)
        while queue:
            curr = queue.popleft()
            comp.append(curr)
            for nbr in adj[curr]:
                if nbr not in visited:
                    visited.add(nbr)
                    queue.append(nbr)
        components.append(comp)

components.sort(key=len, reverse=True)
print(f"\nFound {len(components):,} weakly connected components")
print(f"Largest component: {len(components[0]):,} nodes")
if len(components) > 1:
    print(f"2nd largest:       {len(components[1]):,} nodes")
    print(f"3rd largest:       {len(components[2]):,} nodes" if len(components) > 2 else "")

lwcc_nodes = set(components[0])
print(f"\nNodes in LWCC: {len(lwcc_nodes):,} / {N:,}")

if len(lwcc_nodes) == N:
    print("\nSUBMISSION IS ALREADY WEAKLY CONNECTED! No changes needed.")
else:
    print(f"\nRemoving {N - len(lwcc_nodes):,} nodes not in LWCC...")
    clean = [r for r in df if r['BANC'] in lwcc_nodes]
    out_name = fname.replace('.csv', '_CONNECTED.csv')
    with open(out_name, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['BANC', 'FAFB', 'MCNS'])
        writer.writeheader()
        writer.writerows(clean)
    print(f"Saved: {out_name} ({len(clean):,} rows)")
    print("\nRun verify_max.py on the _CONNECTED file before submitting!")
