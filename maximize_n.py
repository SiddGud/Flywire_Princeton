"""
MAXIMUM N SEARCH — All strategies
===================================
1. Random restart pruning (10 seeds) — different tie-breaking, different local maxima
2. Local search from N=3,873 — "remove 1 to gain many" swaps
3. MAOL check — can we add MAOL as 4th dataset for free?
4. Cell-type-sorted growing — smarter candidate ordering
"""
import pandas as pd
import pyarrow.feather as feather
import time, random
from collections import defaultdict

print("=" * 65)
print("  MAXIMUM N SEARCH — ALL STRATEGIES")
print("=" * 65)

# ─── Load everything ─────────────────────────────────────────
print("\nLoading edge lists...")
def load_str(f):
    df = pd.read_csv(f, dtype=str, header=None)
    df.columns = ["src","tgt"]
    df["src"]=df["src"].str.strip(); df["tgt"]=df["tgt"].str.strip()
    return df

t0=time.time()
fafb_df = load_str("fafb_783_edge_list.csv")
banc_df = load_str("banc_626_edge_list.csv")
mcns_df  = load_str("mcns_0.9_edge_list.csv")
maol_df  = load_str("maol_1.1_edge_list.csv") if __import__('os').path.exists("maol_1.1_edge_list.csv") else None
print(f"Loaded in {time.time()-t0:.1f}s | MAOL: {'found' if maol_df is not None else 'not found'}")

fafb_es=set(zip(fafb_df["src"],fafb_df["tgt"]))
banc_es=set(zip(banc_df["src"],banc_df["tgt"]))
mcns_es=set(zip(mcns_df["src"],mcns_df["tgt"]))
fafb_all=set(fafb_df["src"])|set(fafb_df["tgt"])
banc_all=set(banc_df["src"])|set(banc_df["tgt"])
mcns_all=set(mcns_df["src"])|set(mcns_df["tgt"])
maol_all=set(maol_df["src"])|set(maol_df["tgt"]) if maol_df is not None else set()
maol_es=set(zip(maol_df["src"],maol_df["tgt"])) if maol_df is not None else set()

# ─── Load ALL candidate triplets (manual + NBLAST) ────────────
meta = feather.read_feather("banc_888_meta.feather")
def build_pool(fc, mc):
    p=meta[(meta["sexually_dimorphic"]=="isomorphic")&meta[fc].notna()&meta[mc].notna()].copy()
    p["BANC"]=p["root_626"].astype(str).str.strip()
    p["FAFB"]=p[fc].astype(str).str.strip()
    p["MCNS"]=p[mc].astype(str).str.split(".").str[0]
    return p[["BANC","FAFB","MCNS","cell_type","super_class"]] if "cell_type" in meta.columns else p[["BANC","FAFB","MCNS"]]

pool1=build_pool("fafb_match","malecns_match")
pool2=build_pool("fafb_nblast_match","malecns_nblast_match")
all_cands=pd.concat([pool1,pool2]).drop_duplicates(subset=["BANC"]).reset_index(drop=True)
all_cands=all_cands[all_cands["BANC"].isin(banc_all)&all_cands["FAFB"].isin(fafb_all)&all_cands["MCNS"].isin(mcns_all)].reset_index(drop=True)
print(f"Total candidate pool: {len(all_cands):,}")

# ─── Build bijection-clean unique triplets (for pruning) ──────
valid=pool1[pool1["BANC"].isin(banc_all)&pool1["FAFB"].isin(fafb_all)&pool1["MCNS"].isin(mcns_all)].copy()
valid=valid.drop_duplicates(subset=["BANC"]).drop_duplicates(subset=["FAFB"]).drop_duplicates(subset=["MCNS"]).reset_index(drop=True)
print(f"Unique bijection triplets (pruning pool): {len(valid):,}")

def build_edge_idx(df, b_list, f_list, m_list):
    b2i={b:i for i,b in enumerate(b_list)}
    f2i={f:i for i,f in enumerate(f_list)}
    m2i={m:i for i,m in enumerate(m_list)}
    bi=banc_df[banc_df["src"].isin(b2i)&banc_df["tgt"].isin(b2i)]
    fi=fafb_df[fafb_df["src"].isin(f2i)&fafb_df["tgt"].isin(f2i)]
    mi=mcns_df[mcns_df["src"].isin(m2i)&mcns_df["tgt"].isin(m2i)]
    be=set((b2i[s],b2i[t]) for s,t in zip(bi["src"],bi["tgt"]))
    fe=set((f2i[s],f2i[t]) for s,t in zip(fi["src"],fi["tgt"]))
    me=set((m2i[s],m2i[t]) for s,t in zip(mi["src"],mi["tgt"]))
    return be,fe,me

def prune_random(df, seed=42):
    """Greedy pruner with RANDOM tie-breaking — different seeds give different results"""
    rng=random.Random(seed)
    bl=df["BANC"].tolist(); fl=df["FAFB"].tolist(); ml=df["MCNS"].tolist()
    be,fe,me=build_edge_idx(df,bl,fl,ml)
    active=set(range(len(df)))
    for iteration in range(999999):
        all_e=be|fe|me
        cfct=defaultdict(int); total=0
        for (i,j) in all_e:
            if i not in active or j not in active: continue
            if not((((i,j) in be)==((i,j) in fe)==((i,j) in me))):
                total+=1; cfct[i]+=1; cfct[j]+=1
        if total==0: break
        mx=max(cfct.values())
        candidates_worst=[k for k,v in cfct.items() if v==mx]
        worst=rng.choice(candidates_worst)  # RANDOM tie-breaking!
        active.discard(worst)
        be={e for e in be if e[0] in active and e[1] in active}
        fe={e for e in fe if e[0] in active and e[1] in active}
        me={e for e in me if e[0] in active and e[1] in active}
    return df.iloc[sorted(active)][["BANC","FAFB","MCNS"]].copy().reset_index(drop=True)

def grow(base_df, cand_df, label=""):
    cb=base_df["BANC"].tolist(); cf=base_df["FAFB"].tolist(); cm=base_df["MCNS"].tolist()
    cbs=set(cb); cfs=set(cf); cms=set(cm)
    new_c=cand_df[~cand_df["BANC"].isin(cbs)].reset_index(drop=True)
    added=0
    for _,row in new_c.iterrows():
        rb,rf,rm=row["BANC"],row["FAFB"],row["MCNS"]
        if rb in cbs or rf in cfs or rm in cms: continue
        ok=True
        for j in range(len(cb)):
            for sb,sf,sm,tb,tf,tm in [(rb,rf,rm,cb[j],cf[j],cm[j]),(cb[j],cf[j],cm[j],rb,rf,rm)]:
                if not(((sb,tb)in banc_es)==((sf,tf)in fafb_es)==((sm,tm)in mcns_es)):
                    ok=False; break
            if not ok: break
        if ok:
            cb.append(rb); cf.append(rf); cm.append(rm)
            cbs.add(rb); cfs.add(rf); cms.add(rm)
            added+=1
    result=pd.DataFrame({"BANC":cb,"FAFB":cf,"MCNS":cm})
    print(f"  {label}: pruned={len(base_df)} → grew +{added} → N={len(result):,}")
    return result

# ─── STRATEGY 1: Random restart pruning (10 seeds) ───────────
print("\n=== STRATEGY 1: RANDOM RESTART PRUNING (10 seeds) ===")
best_n=0; best_df=None
for seed in range(10):
    t=time.time()
    pruned=prune_random(valid, seed=seed)
    grown=grow(pruned, all_cands, label=f"seed={seed}")
    if len(grown)>best_n:
        best_n=len(grown)
        best_df=grown.copy()
    print(f"    → seed={seed}: N={len(grown):,} (pruned={len(pruned):,}) [{time.time()-t:.0f}s]")

print(f"\nBest from random restarts: N={best_n:,}")
if best_df is not None:
    best_df.to_csv("submission_RANDOM_BEST.csv", index=False)

# ─── STRATEGY 2: Local search from current best ───────────────
print("\n=== STRATEGY 2: LOCAL SEARCH (remove 1 gain many) ===")
current=pd.read_csv("submission_DEFINITIVE.csv", dtype=str)
print(f"Starting from N={len(current):,}")

# For efficiency: build a "blocking" map
# For each candidate C not in set, which neurons in the set block it?
cb_cur=current["BANC"].tolist(); cf_cur=current["FAFB"].tolist(); cm_cur=current["MCNS"].tolist()
cbs_cur=set(cb_cur); cfs_cur=set(cf_cur); cms_cur=set(cm_cur)

# Only check candidates NOT in current set
rest=all_cands[~all_cands["BANC"].isin(cbs_cur)&~all_cands["FAFB"].isin(cfs_cur)&~all_cands["MCNS"].isin(cms_cur)].reset_index(drop=True)
print(f"Candidates not in current set: {len(rest):,}")

# For each candidate, find which neuron SOLELY blocks it
blockers=defaultdict(list)  # blocker_idx -> list of candidates it's blocking
print("Finding blocking relationships (this takes a few minutes)...")
t=time.time()
for idx,row in rest.iterrows():
    rb,rf,rm=row["BANC"],row["FAFB"],row["MCNS"]
    if rb in cbs_cur or rf in cfs_cur or rm in cms_cur: continue
    blocking_neurons=[]
    for j in range(len(cb_cur)):
        bb,bf,bm=cb_cur[j],cf_cur[j],cm_cur[j]
        violated=False
        for sb,sf,sm,tb,tf,tm in [(rb,rf,rm,bb,bf,bm),(bb,bf,bm,rb,rf,rm)]:
            if not(((sb,tb)in banc_es)==((sf,tf)in fafb_es)==((sm,tm)in mcns_es)):
                violated=True; break
        if violated: blocking_neurons.append(j)
    if len(blocking_neurons)==1:  # Solely blocked by one neuron
        blockers[blocking_neurons[0]].append(idx)
    if idx%1000==0: print(f"  Checked {idx}/{len(rest)} candidates ({time.time()-t:.0f}s)")

# Find best swap: remove one neuron, gain many
best_swap_gain=0; best_swap_idx=-1
for neuron_j, blocked_cands in blockers.items():
    gain=len(blocked_cands)-1  # Remove 1, gain len(blocked_cands)
    if gain>best_swap_gain:
        best_swap_gain=gain; best_swap_idx=neuron_j

print(f"\nBest swap: remove neuron #{best_swap_idx}, gain {best_swap_gain+1} new neurons (net +{best_swap_gain})")
if best_swap_gain>0:
    # Execute the best swap
    remove_b=cb_cur[best_swap_idx]; remove_f=cf_cur[best_swap_idx]; remove_m=cm_cur[best_swap_idx]
    swapped_df=current.drop(index=best_swap_idx).reset_index(drop=True)
    print(f"Removed neuron: BANC={remove_b}")
    swapped_grown=grow(swapped_df, all_cands, label="after_swap")
    if len(swapped_grown)>len(current):
        print(f"IMPROVEMENT: {len(current)} → {len(swapped_grown)} (+{len(swapped_grown)-len(current)})")
        swapped_grown.to_csv("submission_SWAPPED.csv", index=False)
    else:
        print(f"No improvement from swap")

# ─── STRATEGY 3: Check MAOL as 4th dataset ───────────────────
print("\n=== STRATEGY 3: MAOL AS 4th DATASET CHECK ===")
if len(maol_all)==0:
    # Load MAOL by finding the file
    import os, glob
    maol_files=glob.glob("*.csv")
    print("Available CSV files:", [f for f in maol_files if 'maol' in f.lower() or 'male' in f.lower()])
    # Try common names
    for fname in ["maol_1.1_edge_list.csv","male_optic_lobe.csv","maol.csv"]:
        if os.path.exists(fname):
            maol_df=load_str(fname)
            maol_es=set(zip(maol_df["src"],maol_df["tgt"]))
            maol_all=set(maol_df["src"])|set(maol_df["tgt"])
            print(f"Found MAOL: {fname}, {len(maol_all):,} neurons")
            break
    if len(maol_all)==0:
        print("MAOL file not found — listing all CSV files:")
        print([f for f in glob.glob("*.csv")])

if len(maol_all)>0:
    our_mcns=set(current["MCNS"])
    maol_overlap=our_mcns & maol_all
    print(f"Our MCNS neurons: {len(our_mcns):,} | In MAOL: {len(maol_overlap):,}")
    # Check: for our matched neurons in MAOL, do MAOL edges = MCNS edges?
    if len(maol_overlap)>100:
        sample_mcns=list(maol_overlap)[:200]
        sub=current[current["MCNS"].isin(sample_mcns)].reset_index(drop=True)
        m2i={m:i for i,m in enumerate(sub["MCNS"])}
        mi_mcns=mcns_df[mcns_df["src"].isin(m2i)&mcns_df["tgt"].isin(m2i)]
        mi_maol=maol_df[maol_df["src"].isin(m2i)&maol_df["tgt"].isin(m2i)]
        mes_mcns=set((m2i[s],m2i[t]) for s,t in zip(mi_mcns["src"],mi_mcns["tgt"]))
        mes_maol=set((m2i[s],m2i[t]) for s,t in zip(mi_maol["src"],mi_maol["tgt"]))
        maol_extra=mes_maol-mes_mcns  # edges in MAOL but not MCNS (1-4 syn)
        print(f"Sample 200 neurons: MCNS edges={len(mes_mcns)}, MAOL edges={len(mes_maol)}, MAOL-only={len(maol_extra)}")
        print(f"MAOL-only edges (would cause violations): {len(maol_extra)}")

# ─── FINAL SUMMARY ───────────────────────────────────────────
print("\n" + "="*65)
print("  FINAL SUMMARY")
print("="*65)
current_n=len(pd.read_csv("submission_DEFINITIVE.csv"))
random_n=best_n if best_df is not None else 0
print(f"Current best (DEFINITIVE): N={current_n:,}")
print(f"Random restart best:       N={random_n:,}")
overall_best=max(current_n, random_n)
print(f"\nOVERALL BEST N: {overall_best:,}")
