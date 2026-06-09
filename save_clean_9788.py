import pandas as pd

sub = pd.read_csv('submission_ITER_GROW_13081.csv', dtype=str)
banc_df = pd.read_csv('banc_626_edge_list.csv', header=None, names=['src','tgt'], dtype=str)
fafb_df = pd.read_csv('fafb_783_edge_list.csv', header=None, names=['src','tgt'], dtype=str)
mcns_df = pd.read_csv('mcns_0.9_edge_list.csv', header=None, names=['src','tgt'], dtype=str)

b_nodes = set(sub['BANC']); f_nodes = set(sub['FAFB']); m_nodes = set(sub['MCNS'])

# Find nodes with at least 1 internal edge
b_connected = set()
for s, t in zip(banc_df['src'], banc_df['tgt']):
    if s in b_nodes and t in b_nodes:
        b_connected.add(s); b_connected.add(t)

# Keep only connected nodes
clean = sub[sub['BANC'].isin(b_connected)].reset_index(drop=True)
clean.to_csv('submission_CONNECTED_9788.csv', index=False)
print(f"Saved clean submission: {len(clean)} rows")

# Full audit
b2f = dict(zip(clean['BANC'], clean['FAFB']))
b2m = dict(zip(clean['BANC'], clean['MCNS']))
f2b = dict(zip(clean['FAFB'], clean['BANC']))
m2b = dict(zip(clean['MCNS'], clean['BANC']))

cb = set(clean['BANC']); cf = set(clean['FAFB']); cm = set(clean['MCNS'])

be = set()
for s, t in zip(banc_df['src'], banc_df['tgt']):
    if s in cb and t in cb: be.add((s,t))
fe = set()
for s, t in zip(fafb_df['src'], fafb_df['tgt']):
    if s in cf and t in cf: fe.add((s,t))
me = set()
for s, t in zip(mcns_df['src'], mcns_df['tgt']):
    if s in cm and t in cm: me.add((s,t))

print(f"\nEdges: BANC={len(be)}, FAFB={len(fe)}, MCNS={len(me)}")

# Bijection
b_u = clean['BANC'].nunique(); f_u = clean['FAFB'].nunique(); m_u = clean['MCNS'].nunique()
N = len(clean)
print(f"Bijection: BANC={b_u}/{N} FAFB={f_u}/{N} MCNS={m_u}/{N}")

v_bf = sum(1 for s,t in be if (b2f[s],b2f[t]) not in fe)
v_bm = sum(1 for s,t in be if (b2m[s],b2m[t]) not in me)
v_fb = sum(1 for s,t in fe if (f2b[s],f2b[t]) not in be)
v_mb = sum(1 for s,t in me if (m2b[s],m2b[t]) not in be)

total = v_bf + v_bm + v_fb + v_mb
print(f"\nViolations: B->F={v_bf} B->M={v_bm} F->B={v_fb} M->B={v_mb}")
print(f"TOTAL VIOLATIONS: {total}")
if total == 0:
    print("PERFECT DIRECTED ISOMORPHISM CONFIRMED")
else:
    print("BROKEN")
