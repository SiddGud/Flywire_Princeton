import csv
import glob
import os

# Find the latest MAX/S2/S3 submission file (prefer FINAL and CLEAN)
files = glob.glob('submission_MAX_*.csv') + glob.glob('submission_S2_*.csv') + glob.glob('submission_S3_*.csv')
if not files:
    print("No submission files found! Please download the file from Kaggle and place it here.")
    exit()

# Get the one with the highest N (CLEAN is tiebreaker, N is primary)
def sort_key(x):
    base = x.replace('_CLEAN', '').replace('_FINAL', '')
    try:
        n = int(base.split('_')[-1].split('.')[0])
    except:
        n = 0
    clean_bonus = 1 if 'CLEAN' in x else 0
    return (n, clean_bonus)
files.sort(key=sort_key, reverse=True)
fname = files[0]
print(f"Verifying {fname}...\n")

df = []
with open(fname, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        df.append({'BANC': row['BANC'], 'FAFB': row['FAFB'], 'MCNS': row['MCNS']})

N = len(df)
print(f'Rows: {N}')

banc_nodes = set(r['BANC'] for r in df)
fafb_nodes = set(r['FAFB'] for r in df)
mcns_nodes = set(r['MCNS'] for r in df)

print(f'BANC unique: {len(banc_nodes)}/{N}')
print(f'FAFB unique: {len(fafb_nodes)}/{N}')
print(f'MCNS unique: {len(mcns_nodes)}/{N}')

def load_edges(file):
    edges = []
    with open(file, 'r') as f:
        for line in f:
            s, t = line.strip().split(',')
            edges.append((s, t))
    return edges

banc_el = load_edges('banc_626_edge_list.csv')
fafb_el = load_edges('fafb_783_edge_list.csv')
mcns_el = load_edges('mcns_0.9_edge_list.csv')

b2r = {r['BANC']: i for i, r in enumerate(df)}
f2r = {r['FAFB']: i for i, r in enumerate(df)}
m2r = {r['MCNS']: i for i, r in enumerate(df)}

be = set()
fe = set()
me = set()

for s, t in banc_el:
    if s in b2r and t in b2r: be.add((b2r[s], b2r[t]))
for s, t in fafb_el:
    if s in f2r and t in f2r: fe.add((f2r[s], f2r[t]))
for s, t in mcns_el:
    if s in m2r and t in m2r: me.add((m2r[s], m2r[t]))

print(f'\nEdges: BANC={len(be)} FAFB={len(fe)} MCNS={len(me)}')
print(f'BANC==FAFB? {be==fe}')
print(f'BANC==MCNS? {be==me}')

connected = set()
for i,j in be:
    connected.add(i); connected.add(j)
print(f'Connected nodes (BANC): {len(connected)}/{N}')

sl = sum(1 for i,j in be if i==j)
print(f'Self-loops (BANC): {sl}')

# Check for extra/missing edges and clean if needed
if be == fe == me and len(connected) == N and sl == 0:
    print('\nPERFECT ISOMORPHISM CONFIRMED! ✅')
else:
    print('\nISSUES FOUND. Cleaning up...')
    bad_rows = set()
    
    # Extra edges in FAFB
    for i, j in fe - be:
        bad_rows.add(i); bad_rows.add(j)
    # Extra edges in MCNS
    for i, j in me - be:
        bad_rows.add(i); bad_rows.add(j)
    # Missing edges
    for i, j in be - fe:
        bad_rows.add(i); bad_rows.add(j)
    for i, j in be - me:
        bad_rows.add(i); bad_rows.add(j)
        
    print(f"Found {len(bad_rows)} nodes violating isomorphism. Removing them...")
    
    # Create clean df
    clean_df = [df[i] for i in range(len(df)) if i not in bad_rows]
    
    # Remove nodes that are now 0-edge due to removals
    b_nodes = set(r['BANC'] for r in clean_df)
    conn2 = set()
    for s, t in banc_el:
        if s in b_nodes and t in b_nodes:
            conn2.add(s)
            conn2.add(t)
            
    final_df = [r for r in clean_df if r['BANC'] in conn2]
    
    out_name = fname.replace('.csv', '_CLEAN.csv')
    with open(out_name, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['BANC', 'FAFB', 'MCNS'])
        writer.writeheader()
        writer.writerows(final_df)
    print(f"Cleaned file saved to {out_name} (Rows: {len(final_df)})")
    
    print("\nPlease re-run this script to verify the CLEAN file.")
