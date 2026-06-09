"""
GROW + VERIFY the integrated pruner result (N=2,978)
"""
import pandas as pd
import pyarrow.feather as feather
import time

print("=" * 60)
print("  GROW + VERIFY (starting from N=2,978)")
print("=" * 60)

base = pd.read_csv("submission_BEST.csv", dtype=str)
print(f"Base: N={len(base):,} | BANC:{base['BANC'].nunique()} FAFB:{base['FAFB'].nunique()} MCNS:{base['MCNS'].nunique()}")
assert base['BANC'].nunique()==len(base) and base['FAFB'].nunique()==len(base) and base['MCNS'].nunique()==len(base)
print("Bijection confirmed ✅")

# Load ALL candidates (manual only — higher quality)
meta = feather.read_feather("banc_888_meta.feather")
pool = meta[(meta["sexually_dimorphic"]=="isomorphic") &
             meta["fafb_match"].notna() & meta["malecns_match"].notna()].copy()
pool["BANC"] = pool["root_626"].astype(str).str.strip()
pool["FAFB"] = pool["fafb_match"].astype(str).str.strip()
pool["MCNS"] = pool["malecns_match"].astype(str).str.split(".").str[0]

# Load edge lists
print("\nLoading edge lists...")
def load_str(f):
    df = pd.read_csv(f, dtype=str, header=None)
    df.columns = ["src","tgt"]
    df["src"]=df["src"].str.strip(); df["tgt"]=df["tgt"].str.strip()
    return df

fafb_df = load_str("fafb_783_edge_list.csv")
banc_df = load_str("banc_626_edge_list.csv")
mcns_df  = load_str("mcns_0.9_edge_list.csv")

fafb_ids=set(fafb_df["src"])|set(fafb_df["tgt"])
banc_ids=set(banc_df["src"])|set(banc_df["tgt"])
mcns_ids=set(mcns_df["src"])|set(mcns_df["tgt"])

# All valid candidates not in base
cands = pool[
    pool["BANC"].isin(banc_ids) & pool["FAFB"].isin(fafb_ids) & pool["MCNS"].isin(mcns_ids) &
    ~pool["BANC"].isin(set(base["BANC"]))
].drop_duplicates(subset=["BANC"]).reset_index(drop=True)
print(f"Candidates to try: {len(cands):,}")

# Full edge sets
fafb_es=set(zip(fafb_df["src"],fafb_df["tgt"]))
banc_es=set(zip(banc_df["src"],banc_df["tgt"]))
mcns_es=set(zip(mcns_df["src"],mcns_df["tgt"]))

# ─── SPOT CHECK first (verify base is truly 0 violations) ────
print("\nSpot-checking base (300 neurons)...")
samp = base.sample(300, random_state=99)
b2i={b:i for i,b in enumerate(samp["BANC"])}
f2i={f:i for i,f in enumerate(samp["FAFB"])}
m2i={m:i for i,m in enumerate(samp["MCNS"])}
bi=banc_df[banc_df["src"].isin(b2i)&banc_df["tgt"].isin(b2i)]
fi=fafb_df[fafb_df["src"].isin(f2i)&fafb_df["tgt"].isin(f2i)]
mi=mcns_df[mcns_df["src"].isin(m2i)&mcns_df["tgt"].isin(m2i)]
bes=set(zip(bi["src"],bi["tgt"])); fes=set(zip(fi["src"],fi["tgt"])); mes=set(zip(mi["src"],mi["tgt"]))
all_e=bes|fes|mes
viol=sum(1 for e in all_e if not((e in bes)==(e in fes)==(e in mes)))
print(f"Edge positions: {len(all_e):,} | Violations: {viol}")
print(f"SPOT CHECK: {'PASSED ✅' if viol==0 else 'FAILED ❌'}")

# ─── GROW ────────────────────────────────────────────────────
print(f"\nGrowing from {len(base):,}...")
cur_b=base["BANC"].tolist(); cur_f=base["FAFB"].tolist(); cur_m=base["MCNS"].tolist()
cur_b_s=set(cur_b); cur_f_s=set(cur_f); cur_m_s=set(cur_m)
t=time.time(); added=0

for idx, row in cands.iterrows():
    cb,cf,cm=row["BANC"],row["FAFB"],row["MCNS"]
    if cb in cur_b_s or cf in cur_f_s or cm in cur_m_s: continue
    ok=True
    for j in range(len(cur_b)):
        bb,bf,bm=cur_b[j],cur_f[j],cur_m[j]
        for sb,sf,sm,tb,tf,tm in [(cb,cf,cm,bb,bf,bm),(bb,bf,bm,cb,cf,cm)]:
            if not(((sb,tb)in banc_es)==((sf,tf)in fafb_es)==((sm,tm)in mcns_es)):
                ok=False; break
        if not ok: break
    if ok:
        cur_b.append(cb); cur_f.append(cf); cur_m.append(cm)
        cur_b_s.add(cb); cur_f_s.add(cf); cur_m_s.add(cm)
        added+=1
        if added%50==0: print(f"  +{added} ({idx}/{len(cands)} checked, {time.time()-t:.0f}s)")

final=pd.DataFrame({"BANC":cur_b,"FAFB":cur_f,"MCNS":cur_m})
assert final["BANC"].nunique()==len(final) and final["FAFB"].nunique()==len(final) and final["MCNS"].nunique()==len(final)
print(f"\n=== FINAL ===")
print(f"N = {len(final):,}  (was 2,978 + grew +{added:,})")
print("Bijection satisfied ✅")
final.to_csv("submission_FINAL.csv", index=False)
print("Saved to submission_FINAL.csv")
