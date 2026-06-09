import pandas as pd
import numpy as np

print('='*60)
print('  COMPREHENSIVE INDEPENDENT AUDIT')
print('='*60)

sub = pd.read_csv('submission_ITER_GROW_13083.csv', dtype=str)
Nv = len(sub)
print(f'\nTotal rows: {Nv}')

# CHECK 1: Are all IDs unique (bijection)?
b_unique = sub['BANC'].nunique()
f_unique = sub['FAFB'].nunique()
m_unique = sub['MCNS'].nunique()
print('\nCHECK 1 - BIJECTION (all IDs must be unique):')
b1 = "PASS" if b_unique == Nv else "FAIL"
f1 = "PASS" if f_unique == Nv else "FAIL"
m1 = "PASS" if m_unique == Nv else "FAIL"
print(f'  BANC unique: {b_unique} / {Nv}  [{b1}]')
print(f'  FAFB unique: {f_unique} / {Nv}  [{f1}]')
print(f'  MCNS unique: {m_unique} / {Nv}  [{m1}]')

# Load edge lists
banc_df = pd.read_csv('banc_626_edge_list.csv', header=None, names=['src','tgt'], dtype=str)
fafb_df = pd.read_csv('fafb_783_edge_list.csv', header=None, names=['src','tgt'], dtype=str)
mcns_df = pd.read_csv('mcns_0.9_edge_list.csv', header=None, names=['src','tgt'], dtype=str)

# CHECK 2: Do IDs exist in edge lists?
all_banc = set(banc_df['src']) | set(banc_df['tgt'])
all_fafb = set(fafb_df['src']) | set(fafb_df['tgt'])
all_mcns = set(mcns_df['src']) | set(mcns_df['tgt'])

b_missing = len(set(sub['BANC']) - all_banc)
f_missing = len(set(sub['FAFB']) - all_fafb)
m_missing = len(set(sub['MCNS']) - all_mcns)

print('\nCHECK 2 - DO IDS EXIST IN EDGE LISTS?')
print(f'  BANC IDs not in edge list: {b_missing}')
print(f'  FAFB IDs not in edge list: {f_missing}')
print(f'  MCNS IDs not in edge list: {m_missing}')

# CHECK 3: Full directed isomorphism via edge-by-edge mapping
print('\nCHECK 3 - FULL DIRECTED ISOMORPHISM (edge-by-edge):')
b_nodes = set(sub['BANC'])
f_nodes = set(sub['FAFB'])
m_nodes = set(sub['MCNS'])

b2f = dict(zip(sub['BANC'], sub['FAFB']))
b2m = dict(zip(sub['BANC'], sub['MCNS']))
f2b = dict(zip(sub['FAFB'], sub['BANC']))
m2b = dict(zip(sub['MCNS'], sub['BANC']))

# Internal edges
be = set()
for s, t in zip(banc_df['src'], banc_df['tgt']):
    if s in b_nodes and t in b_nodes:
        be.add((s, t))

fe = set()
for s, t in zip(fafb_df['src'], fafb_df['tgt']):
    if s in f_nodes and t in f_nodes:
        fe.add((s, t))

me = set()
for s, t in zip(mcns_df['src'], mcns_df['tgt']):
    if s in m_nodes and t in m_nodes:
        me.add((s, t))

print(f'  Internal edges: BANC={len(be)}, FAFB={len(fe)}, MCNS={len(me)}')

# Check BANC -> FAFB and BANC -> MCNS
v_bf = 0
v_bm = 0
for (s, t) in be:
    if (b2f[s], b2f[t]) not in fe:
        v_bf += 1
    if (b2m[s], b2m[t]) not in me:
        v_bm += 1

# Check FAFB -> BANC
v_fb = 0
for (s, t) in fe:
    if (f2b[s], f2b[t]) not in be:
        v_fb += 1

# Check MCNS -> BANC
v_mb = 0
for (s, t) in me:
    if (m2b[s], m2b[t]) not in be:
        v_mb += 1

print(f'  BANC edge missing in FAFB: {v_bf}')
print(f'  BANC edge missing in MCNS: {v_bm}')
print(f'  FAFB edge missing in BANC: {v_fb}')
print(f'  MCNS edge missing in BANC: {v_mb}')

total_v = v_bf + v_bm + v_fb + v_mb
print(f'\n  TOTAL VIOLATIONS: {total_v}')
if total_v == 0:
    print('  PERFECT DIRECTED ISOMORPHISM CONFIRMED')
else:
    print('  ISOMORPHISM BROKEN')
