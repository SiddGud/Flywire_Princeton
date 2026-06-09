"""
fast_grow.py
============
Fast vectorized Grow-from-Core.

Takes the SA result (4,780 nodes) and tries to add back
any of the 9,771 leftover candidates that are zero-violation.

Uses pre-built adjacency sets for O(d) per-candidate checking
instead of DataFrame queries inside a loop.
"""

import pandas as pd
import time
from collections import defaultdict

SA_FILE = 'submission_HUNGARIAN_SA_4780.csv'
ALL_FILE = 'hungarian_triplets.csv'

print("Loading SA core and all Hungarian triplets...")
# Check what SA file was actually created
import glob
sa_files = sorted(glob.glob('submission_HUNGARIAN_SA_*.csv'))
print(f"Found SA files: {sa_files}")

if sa_files:
    # Use the largest one
    SA_FILE = max(sa_files, key=lambda f: int(f.split('_')[-1].replace('.csv','')))
    print(f"Using: {SA_FILE}")
    sa_df = pd.read_csv(SA_FILE, dtype=str)
else:
    # Fall back: re-run the SA greedy pass quickly
    print("No SA file found — loading hungarian_triplets and running greedy...")
    from collections import defaultdict
    
    filtered = pd.read_csv(ALL_FILE, dtype=str)
    fafb_df = pd.read_csv('fafb_783_edge_list.csv'); fafb_df.columns = ['src','tgt']
    banc_df = pd.read_csv('banc_626_edge_list.csv'); banc_df.columns = ['src','tgt']
    mcns_df = pd.read_csv('mcns_0.9_edge_list.csv'); mcns_df.columns = ['src','tgt']
    
    banc_to_idx = {b: i for i,b in enumerate(filtered['BANC'].tolist())}
    fafb_to_idx = {f: i for i,f in enumerate(filtered['FAFB'].tolist())}
    mcns_to_idx = {m: i for i,m in enumerate(filtered['MCNS'].tolist())}
    
    def be(df, a, b):
        df_s = df.astype(str)
        m = df_s['src'].isin(a) & df_s['tgt'].isin(b)
        return set((a[s],b[t]) for s,t in zip(df_s[m]['src'],df_s[m]['tgt']) if s in a and t in b)
    
    be_b = be(banc_df, banc_to_idx, banc_to_idx)
    be_f = be(fafb_df, fafb_to_idx, fafb_to_idx)
    be_m = be(mcns_df, mcns_to_idx, mcns_to_idx)
    
    all_e = be_b | be_f | be_m
    conflicts = defaultdict(int)
    adj = defaultdict(list)
    total = 0
    for (i,j) in all_e:
        adj[i].append((i,j)); adj[j].append((i,j))
        ib, inf, im = (i,j) in be_b, (i,j) in be_f, (i,j) in be_m
        if not (ib == inf == im):
            total += 1; conflicts[i] += 1; conflicts[j] += 1
    
    active = set(range(len(filtered)))
    while total > 0:
        worst = max(conflicts, key=conflicts.get)
        active.discard(worst)
        for e in adj[worst]:
            if e in all_e:
                ib, inf, im = e in be_b, e in be_f, e in be_m
                if not (ib == inf == im):
                    total -= 1; conflicts[e[0]] -= 1; conflicts[e[1]] -= 1
                all_e.discard(e); be_b.discard(e); be_f.discard(e); be_m.discard(e)
        if worst in conflicts: del conflicts[worst]
    
    sa_df = filtered.iloc[sorted(active)].copy().reset_index(drop=True)
    sa_df.to_csv(f'submission_HUNGARIAN_SA_{len(sa_df)}.csv', index=False)
    print(f"Greedy completed: N = {len(sa_df)}")

print(f"SA core: {len(sa_df):,} nodes")

all_df = pd.read_csv(ALL_FILE, dtype=str)
print(f"Full Hungarian pool: {len(all_df):,} nodes")

# Candidates not in SA core
in_core = set(sa_df['BANC'].unique())
candidates = all_df[~all_df['BANC'].isin(in_core)].copy()
print(f"Candidates to try: {len(candidates):,}")

# Load edge lists once
print("\nLoading edge lists...")
t0 = time.time()
fafb_df = pd.read_csv('fafb_783_edge_list.csv'); fafb_df.columns = ['src','tgt']
banc_df = pd.read_csv('banc_626_edge_list.csv'); banc_df.columns = ['src','tgt']
mcns_df = pd.read_csv('mcns_0.9_edge_list.csv'); mcns_df.columns = ['src','tgt']
print(f"Loaded in {time.time()-t0:.1f}s")

fafb_df = fafb_df.astype(str)
banc_df = banc_df.astype(str)
mcns_df = mcns_df.astype(str)

# Build fast adjacency sets: node -> set of neighbours
print("Building adjacency sets for core neurons...")
t0 = time.time()

# Build lookup: BANC node -> set of BANC neighbors
banc_adj = defaultdict(set)
for _, row in banc_df.iterrows():
    banc_adj[row['src']].add(row['tgt'])
    banc_adj[row['tgt']].add(row['src'])

fafb_adj = defaultdict(set)
for _, row in fafb_df.iterrows():
    fafb_adj[row['src']].add(row['tgt'])
    fafb_adj[row['tgt']].add(row['src'])

mcns_adj = defaultdict(set)
for _, row in mcns_df.iterrows():
    mcns_adj[row['src']].add(row['tgt'])
    mcns_adj[row['tgt']].add(row['src'])

print(f"Adjacency sets built in {time.time()-t0:.1f}s")

# Build core sets for fast lookup
core_banc_list = sa_df['BANC'].tolist()
core_fafb_list = sa_df['FAFB'].tolist()
core_mcns_list = sa_df['MCNS'].tolist()

# Map BANC -> (FAFB, MCNS) for each core node
banc_to_triplet = {b: (f, m) for b, f, m in zip(core_banc_list, core_fafb_list, core_mcns_list)}

print(f"\nTrying to grow from {len(sa_df):,} core nodes...")
t0 = time.time()
added = 0

for _, row in candidates.iterrows():
    b, f, m = str(row['BANC']), str(row['FAFB']), str(row['MCNS'])
    
    # Get neighbors of the new candidate in each graph
    b_neighbors = banc_adj.get(b, set())
    f_neighbors = fafb_adj.get(f, set())
    m_neighbors = mcns_adj.get(m, set())
    
    violation = False
    for cb, (cf, cm) in banc_to_triplet.items():
        # Does new node connect to this core node in each graph?
        e_b = cb in b_neighbors
        e_f = cf in f_neighbors
        e_m = cm in m_neighbors
        if not (e_b == e_f == e_m):
            violation = True
            break
    
    if not violation:
        # Add to core
        banc_to_triplet[b] = (f, m)
        added += 1
        
        if added % 50 == 0:
            print(f"  Added {added} so far, total core now {len(sa_df)+added:,}")

final_n = len(sa_df) + added
print(f"\n✅ Grow-from-Core complete!")
print(f"   SA core:  {len(sa_df):,}")
print(f"   Added:    {added:,}")
print(f"   Final N:  {final_n:,}")

# Build final DataFrame
new_rows = [{'BANC': b, 'FAFB': f, 'MCNS': m} for b, (f, m) in banc_to_triplet.items()]
final_df = pd.DataFrame(new_rows)

# Verify bijection
assert final_df['BANC'].nunique() == len(final_df)
assert final_df['FAFB'].nunique() == len(final_df)
assert final_df['MCNS'].nunique() == len(final_df)
print("   Bijection verified ✅")

out_name = f'submission_GROWN_{final_n}.csv'
final_df.to_csv(out_name, index=False)
print(f"   Saved to {out_name}")
print(f"\nTime: {time.time()-t0:.1f}s")
