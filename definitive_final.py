"""
CORRECT SPOT CHECK + SAVE N=3,873
===================================
Bug: spot_check() was mixing BANC/FAFB/MCNS ID spaces.
BANC IDs (720575941...) never appear in FAFB edge set (720575940...)
so every edge looked violated. Fix: use INDEX-based comparison.
"""
import pandas as pd
import pyarrow.feather as feather
import time

def load_str(f):
    df = pd.read_csv(f, dtype=str, header=None)
    df.columns = ["src","tgt"]
    df["src"]=df["src"].str.strip(); df["tgt"]=df["tgt"].str.strip()
    return df

print("Loading edge lists...")
fafb_df = load_str("fafb_783_edge_list.csv")
banc_df = load_str("banc_626_edge_list.csv")
mcns_df  = load_str("mcns_0.9_edge_list.csv")

def spot_check_correct(df, n=400, label=""):
    """
    CORRECT spot check using INDEX-based edge comparison.
    Neuron i in BANC = neuron i in FAFB = neuron i in MCNS.
    Edge (i,j) must agree across all 3 datasets.
    """
    samp = df.sample(min(n, len(df)), random_state=42)
    b2i = {b:i for i,b in enumerate(samp["BANC"])}
    f2i = {f:i for i,f in enumerate(samp["FAFB"])}
    m2i = {m:i for i,m in enumerate(samp["MCNS"])}

    bi = banc_df[banc_df["src"].isin(b2i) & banc_df["tgt"].isin(b2i)]
    fi = fafb_df[fafb_df["src"].isin(f2i) & fafb_df["tgt"].isin(f2i)]
    mi = mcns_df[mcns_df["src"].isin(m2i) & mcns_df["tgt"].isin(m2i)]

    # KEY FIX: convert to index space so all three sets are comparable
    bes = set((b2i[s], b2i[t]) for s,t in zip(bi["src"],bi["tgt"]))
    fes = set((f2i[s], f2i[t]) for s,t in zip(fi["src"],fi["tgt"]))
    mes = set((m2i[s], m2i[t]) for s,t in zip(mi["src"],mi["tgt"]))

    all_e = bes | fes | mes
    viol = sum(1 for e in all_e if not((e in bes)==(e in fes)==(e in mes)))

    status = "PASS ✅" if viol==0 else f"FAIL ❌ ({viol}/{len(all_e)} violated)"
    print(f"  [{label}] N={len(df):,} | edges={len(all_e):,} | {status}")
    return viol==0

print("\n=== VERIFYING ALL SUBMISSION FILES ===")

# 1. Clean pipeline result (N=2,332)
clean = pd.read_csv("submission_CLEAN_FINAL.csv", dtype=str)
print(f"\nsubmission_CLEAN_FINAL.csv")
print(f"  Bijection: BANC={clean['BANC'].nunique()} FAFB={clean['FAFB'].nunique()} MCNS={clean['MCNS'].nunique()} (rows={len(clean)})")
spot_check_correct(clean, label="CLEAN_FINAL")

# 2. Rebuild N=3,873 (re-run the growing from definitive_final.py)
print("\n=== RE-RUNNING GROWING TO GET N=3,873 ===")
meta = feather.read_feather("banc_888_meta.feather")

def build_pool(fafb_col, mcns_col):
    p = meta[(meta["sexually_dimorphic"]=="isomorphic") &
              meta[fafb_col].notna() & meta[mcns_col].notna()].copy()
    p["BANC"] = p["root_626"].astype(str).str.strip()
    p["FAFB"] = p[fafb_col].astype(str).str.strip()
    p["MCNS"] = p[mcns_col].astype(str).str.split(".").str[0]
    return p[["BANC","FAFB","MCNS"]]

pool1 = build_pool("fafb_match","malecns_match")
pool2 = build_pool("fafb_nblast_match","malecns_nblast_match")
all_cands = pd.concat([pool1, pool2]).drop_duplicates(subset=["BANC"]).reset_index(drop=True)

fafb_all=set(fafb_df["src"])|set(fafb_df["tgt"])
banc_all=set(banc_df["src"])|set(banc_df["tgt"])
mcns_all=set(mcns_df["src"])|set(mcns_df["tgt"])

all_cands = all_cands[
    all_cands["BANC"].isin(banc_all) &
    all_cands["FAFB"].isin(fafb_all) &
    all_cands["MCNS"].isin(mcns_all)
].reset_index(drop=True)

fafb_es=set(zip(fafb_df["src"],fafb_df["tgt"]))
banc_es=set(zip(banc_df["src"],banc_df["tgt"]))
mcns_es=set(zip(mcns_df["src"],mcns_df["tgt"]))

base_b=set(clean["BANC"]); base_f=set(clean["FAFB"]); base_m=set(clean["MCNS"])
new_cands = all_cands[~all_cands["BANC"].isin(base_b)].reset_index(drop=True)

cur_b=clean["BANC"].tolist(); cur_f=clean["FAFB"].tolist(); cur_m=clean["MCNS"].tolist()
cur_bs=set(cur_b); cur_fs=set(cur_f); cur_ms=set(cur_m)
t=time.time(); added=0

for idx,row in new_cands.iterrows():
    cb,cf,cm = row["BANC"],row["FAFB"],row["MCNS"]
    if cb in cur_bs or cf in cur_fs or cm in cur_ms: continue
    ok=True
    for j in range(len(cur_b)):
        bb,bf,bm=cur_b[j],cur_f[j],cur_m[j]
        for sb,sf,sm,tb,tf,tm in [(cb,cf,cm,bb,bf,bm),(bb,bf,bm,cb,cf,cm)]:
            if not(((sb,tb)in banc_es)==((sf,tf)in fafb_es)==((sm,tm)in mcns_es)):
                ok=False; break
        if not ok: break
    if ok:
        cur_b.append(cb); cur_f.append(cf); cur_m.append(cm)
        cur_bs.add(cb); cur_fs.add(cf); cur_ms.add(cm)
        added+=1
        if added%200==0:
            print(f"  +{added} ({idx}/{len(new_cands)} checked, {time.time()-t:.0f}s)")

final=pd.DataFrame({"BANC":cur_b,"FAFB":cur_f,"MCNS":cur_m})
print(f"Grew by +{added} → N={len(final):,}")
assert final["BANC"].nunique()==len(final) and final["FAFB"].nunique()==len(final) and final["MCNS"].nunique()==len(final)
print("Bijection: SATISFIED ✅")

spot_check_correct(final, n=400, label="GROWN")
spot_check_correct(final, n=len(final), label="FULL (all neurons)")

final.to_csv("submission_DEFINITIVE.csv", index=False)
print(f"\n✅ submission_DEFINITIVE.csv saved — N={len(final):,}")
