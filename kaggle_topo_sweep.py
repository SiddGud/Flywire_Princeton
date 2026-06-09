import pandas as pd
import numpy as np
import time, multiprocessing, os
from collections import defaultdict

# ── Auto-find dataset2 path ──────────────────────────────────────
DATA1 = '/kaggle/input/datasets/siddhantgudwani/dataset'
DATA2 = None
for root, dirs, files in os.walk('/kaggle/input'):
    for f in files:
        if 'topological_triplets' in f:
            DATA2 = root
            break
    if DATA2: break

if DATA2 is None:
    # fallback
    for candidate in ['/kaggle/input/dataset2', '/kaggle/input/datasets/siddhantgudwani/dataset2']:
        if os.path.exists(candidate):
            DATA2 = candidate; break

print(f"Edge lists: {DATA1}")
print(f"Triplets:   {DATA2}")
print(f"Files in dataset2: {os.listdir(DATA2) if DATA2 else 'NOT FOUND'}")

FAFB_FILE = f'{DATA1}/fafb_783_edge_list.csv'
BANC_FILE  = f'{DATA1}/banc_626_edge_list.csv'
MCNS_FILE  = f'{DATA1}/mcns_0.9_edge_list.csv'
TOPO_FILE  = f'{DATA2}/topological_triplets.csv'
OUT_DIR    = '/kaggle/working'
CURRENT_BEST = 5092

# ── SA worker (pure function, no file I/O) ───────────────────────
def sa_worker(params):
    from collections import defaultdict
    import numpy as np
    alpha, seed, banc_e, fafb_e, mcns_e, num_nodes = params
    np.random.seed(seed)
    be = set(map(tuple, banc_e))
    fe = set(map(tuple, fafb_e))
    me = set(map(tuple, mcns_e))
    all_e = be | fe | me
    conflicts = defaultdict(int); adj = defaultdict(list); total = 0
    for (i,j) in all_e:
        adj[i].append((i,j)); adj[j].append((i,j))
        ib,iff,im = (i,j) in be,(i,j) in fe,(i,j) in me
        if not (ib==iff==im): total+=1; conflicts[i]+=1; conflicts[j]+=1
    active = set(range(num_nodes))
    while total > 0:
        if alpha==float('inf') or not conflicts:
            worst = max(conflicts, key=conflicts.get)
        else:
            top = sorted(conflicts.items(), key=lambda x:x[1], reverse=True)[:50]
            ns=[x[0] for x in top]; cs=np.array([x[1] for x in top],dtype=np.float64)
            w=cs**alpha; worst=int(np.random.choice(ns, p=w/w.sum()))
        active.discard(worst)
        for e in list(adj[worst]):
            if e in all_e:
                ib,iff,im = e in be,e in fe,e in me
                if not (ib==iff==im): total-=1; conflicts[e[0]]-=1; conflicts[e[1]]-=1
                all_e.discard(e); be.discard(e); fe.discard(e); me.discard(e)
        if worst in conflicts: del conflicts[worst]
    return (alpha, seed, len(active), sorted(list(active)))

if __name__ == '__main__':
    print("\n" + "="*60)
    print(f"  TOPOLOGICAL ALIGNMENT SWEEP  |  Best to beat: {CURRENT_BEST}")
    print("="*60)

    # Load edge lists
    print("\nLoading edge lists...")
    t0 = time.time()
    fafb_df = pd.read_csv(FAFB_FILE, header=None, names=['src','tgt'], dtype=str)
    banc_df = pd.read_csv(BANC_FILE,  header=None, names=['src','tgt'], dtype=str)
    mcns_df = pd.read_csv(MCNS_FILE,  header=None, names=['src','tgt'], dtype=str)
    print(f"Loaded in {time.time()-t0:.1f}s | BANC={len(banc_df):,} FAFB={len(fafb_df):,} MCNS={len(mcns_df):,}")

    banc_str = set(zip(banc_df['src'], banc_df['tgt']))
    fafb_str = set(zip(fafb_df['src'], fafb_df['tgt']))
    mcns_str = set(zip(mcns_df['src'], mcns_df['tgt']))

    print(f"\nLoading Topological triplets from {TOPO_FILE}...")
    triplets = pd.read_csv(TOPO_FILE, dtype=str)
    print(f"Massive Pool: {len(triplets):,} triplets")

    b2i = {b:i for i,b in enumerate(triplets['BANC'].tolist())}
    f2i = {f:i for i,f in enumerate(triplets['FAFB'].tolist())}
    m2i = {m:i for i,m in enumerate(triplets['MCNS'].tolist())}

    def idx_edges(df, ia, ib):
        mask = df['src'].isin(ia) & df['tgt'].isin(ib)
        return [(ia[s], ib[t]) for s,t in zip(df[mask]['src'], df[mask]['tgt'])]

    print("Building index edges...")
    t0 = time.time()
    be = idx_edges(banc_df, b2i, b2i)
    fe = idx_edges(fafb_df, f2i, f2i)
    me = idx_edges(mcns_df, m2i, m2i)
    N  = len(triplets)
    print(f"Done in {time.time()-t0:.1f}s | be={len(be):,} fe={len(fe):,} me={len(me):,}")

    # Broad sweep since it's a completely new pool structure
    alphas = [2, 5, 8, 10, 15, float('inf')]
    tasks  = [(a, s, be, fe, me, N)
              for a in alphas for s in range(45)
              if not (a==float('inf') and s>0)]
    ncores = multiprocessing.cpu_count()
    print(f"\nDispatching {len(tasks)} SA configs across {ncores} cores...")

    best_n=0; best_active=None; best_p=None
    t_sa = time.time()
    with multiprocessing.Pool(processes=ncores) as pool:
        for res in pool.imap_unordered(sa_worker, tasks):
            a, s, n, active = res
            a_str = "Greedy" if a==float('inf') else f"a={a}"
            beat  = "  *** NEW BEST ***" if n > best_n else ""
            print(f"  [{a_str:6s} s={s:2d}] N={n:,}{beat}")
            if n > best_n: best_n=n; best_active=active; best_p=(a_str,s)

    print(f"\n BEST SA: N={best_n}  ({best_p})")
    print(f"SA time: {time.time()-t_sa:.0f}s")

    sa_df = triplets.iloc[best_active].copy().reset_index(drop=True)
    sa_df.to_csv(f'{OUT_DIR}/topo_sa_{best_n}.csv', index=False)

    print(f"\nGrowing from {best_n:,} core...")
    candidates = triplets[~triplets['BANC'].isin(set(sa_df['BANC']))].copy()
    core = {str(b):(str(f),str(m)) for b,f,m in zip(sa_df['BANC'],sa_df['FAFB'],sa_df['MCNS'])}
    added=0; t_g=time.time()
    for _,row in candidates.iterrows():
        b,f,m = str(row['BANC']),str(row['FAFB']),str(row['MCNS'])
        ok=True
        for cb,(cf,cm) in core.items():
            if ((b,cb) in banc_str)!=((f,cf) in fafb_str) or \
               ((b,cb) in banc_str)!=((m,cm) in mcns_str): ok=False; break
            if ((cb,b) in banc_str)!=((cf,f) in fafb_str) or \
               ((cb,b) in banc_str)!=((cm,m) in mcns_str): ok=False; break
        if ok:
            core[b]=(f,m); added+=1
            if added%10==0: print(f"  +{added} grown → {best_n+added:,} ({time.time()-t_g:.0f}s)")

    final_n = best_n + added
    rows = [{'BANC':b,'FAFB':f,'MCNS':m} for b,(f,m) in core.items()]
    final_df = pd.DataFrame(rows)

    print(f"\n{'='*60}")
    print(f"  SA:      {best_n:,}")
    print(f"  Grown:   +{added:,}")
    print(f"  FINAL N: {final_n:,}  (vs best {CURRENT_BEST:,}, +{final_n-CURRENT_BEST:,})")
    print(f"{'='*60}")

    out = f'{OUT_DIR}/submission_TOPO_{final_n}.csv'
    final_df.to_csv(out, index=False)
    print(f"Saved: {out}")

    print("\nVerifying...")
    Nv=len(final_df)
    bi={n:i for i,n in enumerate(final_df['BANC'].tolist())}
    fi={n:i for i,n in enumerate(final_df['FAFB'].tolist())}
    mi={n:i for i,n in enumerate(final_df['MCNS'].tolist())}
    mb=np.zeros((Nv,Nv),dtype=bool)
    mf=np.zeros((Nv,Nv),dtype=bool)
    mm=np.zeros((Nv,Nv),dtype=bool)
    for df,mat,ia,ib in [(banc_df,mb,bi,bi),(fafb_df,mf,fi,fi),(mcns_df,mm,mi,mi)]:
        mask=df['src'].isin(ia)&df['tgt'].isin(ib)
        d=df[mask]; r=d['src'].map(ia).values; c=d['tgt'].map(ib).values; mat[r,c]=True
    v=((mb^mf)|(mb^mm)); np.fill_diagonal(v,False)
    print(f"Violations: {v.sum()} | Edges BANC={mb.sum()} FAFB={mf.sum()} MCNS={mm.sum()}")
    print("PERFECT ISOMORPHISM ✅" if v.sum()==0 else f"VIOLATIONS: {v.sum()} ❌")
