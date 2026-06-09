"""
CLEAN PIPELINE FROM SCRATCH — Bijection enforced from the start
================================================================
The critical bug: BANC metadata has multiple BANC neurons pointing to
the SAME FAFB/MCNS neuron (different segmentation of same cell).
Must deduplicate by FAFB and MCNS before ANY processing.
"""
import pandas as pd
import pyarrow.feather as feather
import time
from collections import defaultdict

print("=" * 60)
print("  CLEAN PIPELINE (bijection from start)")
print("=" * 60)

# ─── Load metadata with deduplication ────────────────────────
print("\nLoading metadata...")
meta = feather.read_feather("banc_888_meta.feather")

valid_meta = meta[
    (meta["sexually_dimorphic"] == "isomorphic") &
    meta["fafb_match"].notna() &
    meta["malecns_match"].notna()
].copy()

valid_meta["BANC"] = valid_meta["root_626"].astype(str).str.strip()
valid_meta["FAFB"] = valid_meta["fafb_match"].astype(str).str.strip()
valid_meta["MCNS"] = valid_meta["malecns_match"].astype(str).str.split(".").str[0]

print(f"Before dedup: {len(valid_meta):,} rows")
print(f"  BANC unique: {valid_meta['BANC'].nunique():,}")
print(f"  FAFB unique: {valid_meta['FAFB'].nunique():,}")
print(f"  MCNS unique: {valid_meta['MCNS'].nunique():,}")

# Deduplicate: each FAFB and MCNS neuron can only appear ONCE
# Keep first occurrence (highest confidence per metadata sort)
valid_meta = valid_meta.drop_duplicates(subset=["BANC"])
valid_meta = valid_meta.drop_duplicates(subset=["FAFB"])
valid_meta = valid_meta.drop_duplicates(subset=["MCNS"])
valid_meta = valid_meta.reset_index(drop=True)
print(f"\nAfter dedup: {len(valid_meta):,} rows (all unique)")

# ─── Load challenge files ─────────────────────────────────────
print("\nLoading edge lists...")
t0 = time.time()
def load_str(f):
    df = pd.read_csv(f, dtype=str, header=None)
    df.columns = ["src","tgt"]
    df["src"] = df["src"].str.strip(); df["tgt"] = df["tgt"].str.strip()
    return df

fafb_df = load_str("fafb_783_edge_list.csv")
banc_df = load_str("banc_626_edge_list.csv")
mcns_df  = load_str("mcns_0.9_edge_list.csv")
print(f"Loaded in {time.time()-t0:.1f}s")

fafb_ids = set(fafb_df["src"])|set(fafb_df["tgt"])
banc_ids = set(banc_df["src"])|set(banc_df["tgt"])
mcns_ids  = set(mcns_df["src"])|set(mcns_df["tgt"])

# Filter to challenge files
valid = valid_meta[
    valid_meta["BANC"].isin(banc_ids) &
    valid_meta["FAFB"].isin(fafb_ids) &
    valid_meta["MCNS"].isin(mcns_ids)
].reset_index(drop=True)
print(f"Valid unique triplets in challenge files: {len(valid):,}")

# ─── Build edge sets ─────────────────────────────────────────
print("Building edge sets...")
t = time.time()
banc_list = valid["BANC"].tolist()
fafb_list = valid["FAFB"].tolist()
mcns_list  = valid["MCNS"].tolist()

b2i = {b:i for i,b in enumerate(banc_list)}
f2i = {f:i for i,f in enumerate(fafb_list)}
m2i = {m:i for i,m in enumerate(mcns_list)}

bi = banc_df[banc_df["src"].isin(b2i)&banc_df["tgt"].isin(b2i)]
fi = fafb_df[fafb_df["src"].isin(f2i)&fafb_df["tgt"].isin(f2i)]
mi = mcns_df[mcns_df["src"].isin(m2i)&mcns_df["tgt"].isin(m2i)]

banc_ei = set((b2i[s],b2i[t]) for s,t in zip(bi["src"],bi["tgt"]))
fafb_ei = set((f2i[s],f2i[t]) for s,t in zip(fi["src"],fi["tgt"]))
mcns_ei  = set((m2i[s],m2i[t]) for s,t in zip(mi["src"],mi["tgt"]))
print(f"Internal edges — BANC:{len(banc_ei):,} FAFB:{len(fafb_ei):,} MCNS:{len(mcns_ei):,}")
print(f"Built in {time.time()-t:.1f}s")

# ─── Count initial conflicts ──────────────────────────────────
all_ei = banc_ei | fafb_ei | mcns_ei
init_c = sum(1 for e in all_ei if not ((e in banc_ei)==(e in fafb_ei)==(e in mcns_ei)))
print(f"\nInitial conflicts: {init_c:,} / {len(all_ei):,}  ({100*init_c/max(1,len(all_ei)):.1f}%)")

# ─── Greedy pruning ───────────────────────────────────────────
print(f"\n=== PRUNING ===")
t = time.time()
active = set(range(len(valid)))
b_e, f_e, m_e = set(banc_ei), set(fafb_ei), set(mcns_ei)

for iteration in range(100000):
    all_e = b_e | f_e | m_e
    cfct = defaultdict(int)
    total = 0
    for (i,j) in all_e:
        if i not in active or j not in active: continue
        if not (((i,j) in b_e)==((i,j) in f_e)==((i,j) in m_e)):
            total += 1; cfct[i]+=1; cfct[j]+=1
    if total == 0:
        print(f"\n✅ Converged after {iteration} removals!")
        break
    worst = max(cfct, key=cfct.get)
    active.discard(worst)
    b_e={e for e in b_e if e[0] in active and e[1] in active}
    f_e={e for e in f_e if e[0] in active and e[1] in active}
    m_e={e for e in m_e if e[0] in active and e[1] in active}
    if iteration % 200 == 0:
        print(f"  iter {iteration}: {len(active):,} neurons, {total:,} conflicts ({time.time()-t:.0f}s)")

# ─── Save pruned result ───────────────────────────────────────
pruned = valid.iloc[sorted(active)][["BANC","FAFB","MCNS"]].copy()
print(f"\nPruned N = {len(pruned):,}")
pruned.to_csv("submission_PRUNED_CLEAN.csv", index=False)

# ─── Grow ────────────────────────────────────────────────────
print(f"\n=== GROWING FROM {len(pruned):,} ===")
fafb_es = set(zip(fafb_df["src"],fafb_df["tgt"]))
banc_es = set(zip(banc_df["src"],banc_df["tgt"]))
mcns_es  = set(zip(mcns_df["src"],mcns_df["tgt"]))

cur_b = pruned["BANC"].tolist()
cur_f = pruned["FAFB"].tolist()
cur_m  = pruned["MCNS"].tolist()
cur_b_s, cur_f_s, cur_m_s = set(cur_b), set(cur_f), set(cur_m)

# All unused valid candidates
rest = valid[~valid["BANC"].isin(cur_b_s)].reset_index(drop=True)
print(f"Candidates to try: {len(rest):,}")
t = time.time(); added = 0

for idx, row in rest.iterrows():
    cb, cf, cm = row["BANC"], row["FAFB"], row["MCNS"]
    if cb in cur_b_s or cf in cur_f_s or cm in cur_m_s: continue
    ok = True
    for j in range(len(cur_b)):
        bb,bf,bm = cur_b[j],cur_f[j],cur_m[j]
        for sb,sf,sm,tb,tf,tm in [(cb,cf,cm,bb,bf,bm),(bb,bf,bm,cb,cf,cm)]:
            if not (((sb,tb) in banc_es)==((sf,tf) in fafb_es)==((sm,tm) in mcns_es)):
                ok=False; break
        if not ok: break
    if ok:
        cur_b.append(cb); cur_f.append(cf); cur_m.append(cm)
        cur_b_s.add(cb); cur_f_s.add(cf); cur_m_s.add(cm)
        added += 1
        if added % 100 == 0:
            print(f"  +{added} ({idx}/{len(rest)} checked, {time.time()-t:.0f}s)")

final = pd.DataFrame({"BANC":cur_b,"FAFB":cur_f,"MCNS":cur_m})
print(f"\n=== FINAL RESULT ===")
print(f"N = {len(final):,} (pruned {len(pruned):,} + grew +{added:,})")
assert final["BANC"].nunique()==len(final) and final["FAFB"].nunique()==len(final) and final["MCNS"].nunique()==len(final)
print("All IDs unique — bijection satisfied ✅")
final.to_csv("submission_CLEAN_FINAL.csv", index=False)
print("Saved to submission_CLEAN_FINAL.csv")
