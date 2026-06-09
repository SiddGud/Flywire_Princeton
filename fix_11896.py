import pandas as pd

df = pd.read_csv('submission_MULTI_11896.csv', dtype=str)
N = len(df)

banc_el = pd.read_csv('banc_626_edge_list.csv', header=None, names=['s','t'], dtype=str)
fafb_el = pd.read_csv('fafb_783_edge_list.csv', header=None, names=['s','t'], dtype=str)
mcns_el = pd.read_csv('mcns_0.9_edge_list.csv', header=None, names=['s','t'], dtype=str)

b2r = {b:i for i,b in enumerate(df['BANC'])}
f2r = {f:i for i,f in enumerate(df['FAFB'])}
m2r = {m:i for i,m in enumerate(df['MCNS'])}

be = set(); fe = set(); me = set()
for s,t in zip(banc_el['s'], banc_el['t']):
    if s in b2r and t in b2r: be.add((b2r[s], b2r[t]))
for s,t in zip(fafb_el['s'], fafb_el['t']):
    if s in f2r and t in f2r: fe.add((f2r[s], f2r[t]))
for s,t in zip(mcns_el['s'], mcns_el['t']):
    if s in m2r and t in m2r: me.add((m2r[s], m2r[t]))

# Find extra MCNS edges
extra = me - be
print(f"Extra MCNS edges (not in BANC): {len(extra)}")
bad_rows = set()
for i, j in extra:
    print(f"  Row {i} -> Row {j}  (self-loop: {i==j})")
    bad_rows.add(i)
    bad_rows.add(j)

print(f"\nRows to remove: {bad_rows}")

# Remove bad rows
clean = df.drop(index=list(bad_rows)).reset_index(drop=True)

# Re-verify: remove 0-edge nodes
b_nodes = set(clean['BANC'])
connected = set()
for s,t in zip(banc_el['s'], banc_el['t']):
    if s in b_nodes and t in b_nodes:
        connected.add(s); connected.add(t)
clean = clean[clean['BANC'].isin(connected)].reset_index(drop=True)

clean.to_csv('submission_MULTI_CLEAN.csv', index=False)
print(f"\nCleaned: {len(clean)} rows")

# Final verify
N2 = len(clean)
b2r2 = {b:i for i,b in enumerate(clean['BANC'])}
f2r2 = {f:i for i,f in enumerate(clean['FAFB'])}
m2r2 = {m:i for i,m in enumerate(clean['MCNS'])}

be2 = set(); fe2 = set(); me2 = set()
for s,t in zip(banc_el['s'], banc_el['t']):
    if s in b2r2 and t in b2r2: be2.add((b2r2[s], b2r2[t]))
for s,t in zip(fafb_el['s'], fafb_el['t']):
    if s in f2r2 and t in f2r2: fe2.add((f2r2[s], f2r2[t]))
for s,t in zip(mcns_el['s'], mcns_el['t']):
    if s in m2r2 and t in m2r2: me2.add((m2r2[s], m2r2[t]))

print(f"Edges: BANC={len(be2)} FAFB={len(fe2)} MCNS={len(me2)}")
print(f"BANC==FAFB? {be2==fe2}")
print(f"BANC==MCNS? {be2==me2}")
sl = sum(1 for i,j in be2 if i==j)
conn = set()
for i,j in be2: conn.add(i); conn.add(j)
print(f"Connected: {len(conn)}/{N2}, Self-loops: {sl}")
if be2==fe2==me2 and len(conn)==N2 and sl==0:
    print("PERFECT ISOMORPHISM CONFIRMED")
