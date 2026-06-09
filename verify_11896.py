import pandas as pd

fname = 'submission_MULTI_11896.csv'
df = pd.read_csv(fname, dtype=str)
N = len(df)
print(f'Rows: {N}')

print(f'BANC unique: {df["BANC"].nunique()}/{N}')
print(f'FAFB unique: {df["FAFB"].nunique()}/{N}')
print(f'MCNS unique: {df["MCNS"].nunique()}/{N}')

banc_el = pd.read_csv('banc_626_edge_list.csv', header=None, names=['s','t'], dtype=str)
fafb_el = pd.read_csv('fafb_783_edge_list.csv', header=None, names=['s','t'], dtype=str)
mcns_el = pd.read_csv('mcns_0.9_edge_list.csv', header=None, names=['s','t'], dtype=str)

banc_ids = df['BANC'].tolist()
fafb_ids = df['FAFB'].tolist()
mcns_ids = df['MCNS'].tolist()
b2r = {b:i for i,b in enumerate(banc_ids)}
f2r = {f:i for i,f in enumerate(fafb_ids)}
m2r = {m:i for i,m in enumerate(mcns_ids)}

be = set(); fe = set(); me = set()
for s,t in zip(banc_el['s'], banc_el['t']):
    if s in b2r and t in b2r: be.add((b2r[s], b2r[t]))
for s,t in zip(fafb_el['s'], fafb_el['t']):
    if s in f2r and t in f2r: fe.add((f2r[s], f2r[t]))
for s,t in zip(mcns_el['s'], mcns_el['t']):
    if s in m2r and t in m2r: me.add((m2r[s], m2r[t]))

print(f'Edges: BANC={len(be)} FAFB={len(fe)} MCNS={len(me)}')
print(f'BANC==FAFB? {be==fe}')
print(f'BANC==MCNS? {be==me}')

connected = set()
for i,j in be:
    connected.add(i); connected.add(j)
print(f'Connected: {len(connected)}/{N}')

sl = sum(1 for i,j in be if i==j)
print(f'Self-loops: {sl}')

if be == fe == me and len(connected) == N and sl == 0:
    print('PERFECT ISOMORPHISM CONFIRMED')
else:
    print('ISSUES FOUND')
