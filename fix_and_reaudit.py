import pandas as pd

sub = pd.read_csv('submission_ITER_GROW_13083.csv', dtype=str)
print(f"Before: {len(sub)} rows")

# Remove the 2 self-loop violating MCNS nodes
bad_mcns = {'37975', '917499'}
sub = sub[~sub['MCNS'].isin(bad_mcns)].reset_index(drop=True)
print(f"After removing 2 self-loop violators: {len(sub)} rows")

# Save
sub.to_csv('submission_ITER_GROW_13081.csv', index=False)
print("Saved to submission_ITER_GROW_13081.csv")

# Now run the FULL audit again
banc_df = pd.read_csv('banc_626_edge_list.csv', header=None, names=['src','tgt'], dtype=str)
fafb_df = pd.read_csv('fafb_783_edge_list.csv', header=None, names=['src','tgt'], dtype=str)
mcns_df = pd.read_csv('mcns_0.9_edge_list.csv', header=None, names=['src','tgt'], dtype=str)

Nv = len(sub)

# CHECK 1: Bijection
b_u = sub['BANC'].nunique()
f_u = sub['FAFB'].nunique()
m_u = sub['MCNS'].nunique()
print(f"\nCHECK 1 - BIJECTION:")
print(f"  BANC: {b_u}/{Nv} [{'PASS' if b_u==Nv else 'FAIL'}]")
print(f"  FAFB: {f_u}/{Nv} [{'PASS' if f_u==Nv else 'FAIL'}]")
print(f"  MCNS: {m_u}/{Nv} [{'PASS' if m_u==Nv else 'FAIL'}]")

# CHECK 2: Full directed isomorphism edge-by-edge (INCLUDING self-loops this time)
b_nodes = set(sub['BANC']); f_nodes = set(sub['FAFB']); m_nodes = set(sub['MCNS'])
b2f = dict(zip(sub['BANC'], sub['FAFB']))
b2m = dict(zip(sub['BANC'], sub['MCNS']))
f2b = dict(zip(sub['FAFB'], sub['BANC']))
m2b = dict(zip(sub['MCNS'], sub['BANC']))

be = set()
for s, t in zip(banc_df['src'], banc_df['tgt']):
    if s in b_nodes and t in b_nodes: be.add((s, t))

fe = set()
for s, t in zip(fafb_df['src'], fafb_df['tgt']):
    if s in f_nodes and t in f_nodes: fe.add((s, t))

me = set()
for s, t in zip(mcns_df['src'], mcns_df['tgt']):
    if s in m_nodes and t in m_nodes: me.add((s, t))

print(f"\nCHECK 2 - EDGES: BANC={len(be)}, FAFB={len(fe)}, MCNS={len(me)}")

v_bf = sum(1 for s,t in be if (b2f[s],b2f[t]) not in fe)
v_bm = sum(1 for s,t in be if (b2m[s],b2m[t]) not in me)
v_fb = sum(1 for s,t in fe if (f2b[s],f2b[t]) not in be)
v_mb = sum(1 for s,t in me if (m2b[s],m2b[t]) not in be)

print(f"  BANC->FAFB missing: {v_bf}")
print(f"  BANC->MCNS missing: {v_bm}")
print(f"  FAFB->BANC missing: {v_fb}")
print(f"  MCNS->BANC missing: {v_mb}")

total = v_bf + v_bm + v_fb + v_mb
print(f"\n  TOTAL VIOLATIONS: {total}")
if total == 0:
    print("  PERFECT DIRECTED ISOMORPHISM CONFIRMED")
else:
    print("  ISOMORPHISM STILL BROKEN")
