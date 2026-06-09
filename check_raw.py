import csv
from collections import defaultdict, deque

print('Building adjacency and edge sets...')
adj = defaultdict(set)
banc_set = set()
with open('banc_626_edge_list.csv') as f:
    for line in f:
        s, t = line.strip().split(',')
        adj[s].add(t); adj[t].add(s)
        banc_set.add((s, t))

fafb_set = set()
with open('fafb_783_edge_list.csv') as f:
    for line in f:
        s, t = line.strip().split(',')
        fafb_set.add((s, t))

mcns_set = set()
with open('mcns_0.9_edge_list.csv') as f:
    for line in f:
        s, t = line.strip().split(',')
        mcns_set.add((s, t))

print('Loading submission_raw.csv...')
df = []
with open('submission_raw.csv') as f:
    for row in csv.DictReader(f):
        df.append(row)

N = len(df)
print(f'Rows: {N:,}')
print(f'BANC unique: {len(set(r["BANC"] for r in df)):,}')
print(f'FAFB unique: {len(set(r["FAFB"] for r in df)):,}')
print(f'MCNS unique: {len(set(r["MCNS"] for r in df)):,}')

# Extract LWCC
nodes = set(r['BANC'] for r in df)
visited, comps = set(), []
for node in nodes:
    if node not in visited:
        comp, q = [], deque([node])
        visited.add(node)
        while q:
            c = q.popleft(); comp.append(c)
            for nb in adj[c]:
                if nb in nodes and nb not in visited:
                    visited.add(nb); q.append(nb)
        comps.append(comp)
comps.sort(key=len, reverse=True)
lwcc = set(comps[0])
print(f'\nLWCC: {len(lwcc):,} nodes ({len(comps):,} components total)')

# Build mappings for LWCC
b2f = {r['BANC']: r['FAFB'] for r in df if r['BANC'] in lwcc}
b2m = {r['BANC']: r['MCNS'] for r in df if r['BANC'] in lwcc}
f2b = {v: k for k, v in b2f.items()}
m2b = {v: k for k, v in b2m.items()}

# Check isomorphism on LWCC
be = set()
for s, t in banc_set:
    if s in lwcc and t in lwcc:
        be.add((s, t))

fe_banc = set()
for s, t in fafb_set:
    bs, bt = f2b.get(s), f2b.get(t)
    if bs and bt:
        fe_banc.add((bs, bt))

me_banc = set()
for s, t in mcns_set:
    bs, bt = m2b.get(s), m2b.get(t)
    if bs and bt:
        me_banc.add((bs, bt))

print(f'BANC internal edges: {len(be):,}')
print(f'FAFB==BANC: {fe_banc == be}')
print(f'MCNS==BANC: {me_banc == be}')

if fe_banc == be == me_banc:
    print(f'\nPERFECT ISOMORPHISM ON LWCC - SAVING...')
    lwcc_rows = [r for r in df if r['BANC'] in lwcc]
    with open('submission_raw_LWCC.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['BANC', 'FAFB', 'MCNS'])
        writer.writeheader()
        writer.writerows(lwcc_rows)
    print(f'Saved submission_raw_LWCC.csv ({len(lwcc_rows):,} rows)')
else:
    print('ISOMORPHISM VIOLATED on LWCC - file may be invalid')
    print(f'Extra in FAFB: {len(fe_banc - be)}')
    print(f'Extra in MCNS: {len(me_banc - be)}')
    print(f'Missing in FAFB: {len(be - fe_banc)}')
    print(f'Missing in MCNS: {len(be - me_banc)}')
