import pandas as pd
import numpy as np
import time, os

DATA1 = '/kaggle/input/datasets/siddhantgudwani/dataset'
DATA2 = '/kaggle/working'

print("Loading edge lists...")
fafb_df = pd.read_csv(f'{DATA1}/fafb_783_edge_list.csv', header=None, names=['src','tgt'], dtype=str)
banc_df = pd.read_csv(f'{DATA1}/banc_626_edge_list.csv', header=None, names=['src','tgt'], dtype=str)
mcns_df = pd.read_csv(f'{DATA1}/mcns_0.9_edge_list.csv', header=None, names=['src','tgt'], dtype=str)

print("Loading K-Means triplets...")
if not os.path.exists(f'{DATA2}/kmeans_triplets.csv'):
    print("❌ kmeans_triplets.csv not found! Please run the generate_kmeans_triplets script first!")
    exit()

triplets = pd.read_csv(f'{DATA2}/kmeans_triplets.csv', dtype=str)
print(f"Original Pool: {len(triplets):,} triplets")

b2i = {b:i for i,b in enumerate(triplets['BANC'].tolist())}
f2i = {f:i for i,f in enumerate(triplets['FAFB'].tolist())}
m2i = {m:i for i,m in enumerate(triplets['MCNS'].tolist())}

def idx_edges(df, ia, ib):
    mask = df['src'].isin(ia) & df['tgt'].isin(ib)
    return [(ia[s], ib[t]) for s,t in zip(df[mask]['src'], df[mask]['tgt'])]

be = idx_edges(banc_df, b2i, b2i)
fe = idx_edges(fafb_df, f2i, f2i)
me = idx_edges(mcns_df, m2i, m2i)

# ========================================================
# STRATEGY 1: PURGE THE ZEROES
# ========================================================
print("\nPurging completely disconnected (0-edge) nodes from the pool...")
connected_banc_nodes = set(u for u,v in be) | set(v for u,v in be)
connected_fafb_nodes = set(u for u,v in fe) | set(v for u,v in fe)
connected_mcns_nodes = set(u for u,v in me) | set(v for u,v in me)

# A node must have at least one connection in ALL THREE datasets to even be considered!
valid_indices = []
for i, row in triplets.iterrows():
    if i in connected_banc_nodes and i in connected_fafb_nodes and i in connected_mcns_nodes:
        valid_indices.append(i)

triplets_filtered = triplets.iloc[valid_indices].reset_index(drop=True)
print(f"Surviving Pool (Connected Nodes Only): {len(triplets_filtered):,} triplets")

if len(triplets_filtered) == 0:
    print("Result: 0 nodes survived the filter! Pure math found NO connected valid matches.")
    exit()

# Re-index for SA
b2i = {b:i for i,b in enumerate(triplets_filtered['BANC'].tolist())}
f2i = {f:i for i,f in enumerate(triplets_filtered['FAFB'].tolist())}
m2i = {m:i for i,m in enumerate(triplets_filtered['MCNS'].tolist())}

be_filtered = idx_edges(banc_df, b2i, b2i)
fe_filtered = idx_edges(fafb_df, f2i, f2i)
me_filtered = idx_edges(mcns_df, m2i, m2i)

print(f"\nRunning Simulated Annealing on the surviving {len(triplets_filtered)} connected candidates...")
from collections import defaultdict
np.random.seed(42)
be_set, fe_set, me_set = set(map(tuple, be_filtered)), set(map(tuple, fe_filtered)), set(map(tuple, me_filtered))
all_e = be_set | fe_set | me_set
conflicts = defaultdict(int); adj = defaultdict(list); total = 0
for (i,j) in all_e:
    adj[i].append((i,j)); adj[j].append((i,j))
    ib,iff,im = (i,j) in be_set,(i,j) in fe_set,(i,j) in me_set
    if not (ib==iff==im): total+=1; conflicts[i]+=1; conflicts[j]+=1
active = set(range(len(triplets_filtered)))

while total > 0:
    top = sorted(conflicts.items(), key=lambda x:x[1], reverse=True)[:50]
    ns=[x[0] for x in top]; cs=np.array([x[1] for x in top],dtype=np.float64)
    w=cs**2; worst=int(np.random.choice(ns, p=w/w.sum()))
    active.discard(worst)
    for e in list(adj[worst]):
        if e in all_e:
            ib,iff,im = e in be_set,e in fe_set,e in me_set
            if not (ib==iff==im): total-=1; conflicts[e[0]]-=1; conflicts[e[1]]-=1
            all_e.discard(e); be_set.discard(e); fe_set.discard(e); me_set.discard(e)
    if worst in conflicts: del conflicts[worst]

best_n = len(active)
print("\n" + "="*60)
print(f"  FINAL CONNECTED N: {best_n}")
print("="*60)
