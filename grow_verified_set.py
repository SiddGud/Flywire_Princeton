"""
FIXED GROWING ALGORITHM
========================
Bug fixed: must ensure FAFB and MCNS IDs are UNIQUE across the set
(bijection requirement — each neuron maps to exactly one counterpart).
"""
import pandas as pd
import pyarrow.feather as feather
import time

print("=" * 60)
print("  FIXED GROWING ALGORITHM (bijection enforced)")
print("=" * 60)

# ─── Load verified base set ───────────────────────────────────
base = pd.read_csv("submission_FINAL_VALID.csv", dtype=str)
print(f"Base: {len(base):,} | BANC unique: {base['BANC'].nunique()} | "
      f"FAFB unique: {base['FAFB'].nunique()} | MCNS unique: {base['MCNS'].nunique()}")
assert base['BANC'].nunique() == len(base), "Base has duplicate BANC IDs!"
assert base['FAFB'].nunique() == len(base), "Base has duplicate FAFB IDs!"
assert base['MCNS'].nunique() == len(base), "Base has duplicate MCNS IDs!"
print("Base verified: all IDs unique ✅")

# ─── Load candidates ──────────────────────────────────────────
meta = feather.read_feather("banc_888_meta.feather")
pool1 = meta[(meta["sexually_dimorphic"]=="isomorphic") &
              meta["fafb_match"].notna() & meta["malecns_match"].notna()].copy()
pool1["BANC"] = pool1["root_626"].astype(str).str.strip()
pool1["FAFB"] = pool1["fafb_match"].astype(str).str.strip()
pool1["MCNS"] = pool1["malecns_match"].astype(str).str.split(".").str[0]

pool2 = meta[(meta["sexually_dimorphic"]=="isomorphic") &
              meta["fafb_nblast_match"].notna() & meta["malecns_nblast_match"].notna()].copy()
pool2["BANC"] = pool2["root_626"].astype(str).str.strip()
pool2["FAFB"] = pool2["fafb_nblast_match"].astype(str).str.strip()
pool2["MCNS"] = pool2["malecns_nblast_match"].astype(str).str.split(".").str[0]

# Manual matches first (higher quality), then NBLAST
all_cands = pd.concat([pool1[["BANC","FAFB","MCNS"]],
                        pool2[["BANC","FAFB","MCNS"]]]).drop_duplicates(subset=["BANC"]).reset_index(drop=True)

# ─── Load edge lists ──────────────────────────────────────────
print("Loading edge lists...")
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

# Filter candidates to those in challenge files
new_cands = all_cands[
    ~all_cands["BANC"].isin(set(base["BANC"])) &   # BANC not already in base
    all_cands["BANC"].isin(banc_ids) &
    all_cands["FAFB"].isin(fafb_ids) &
    all_cands["MCNS"].isin(mcns_ids)
].reset_index(drop=True)
print(f"Candidates in challenge files (not in base): {len(new_cands):,}")

# ─── Build full edge sets ────────────────────────────────────
print("Building edge sets...")
t = time.time()
fafb_es = set(zip(fafb_df["src"], fafb_df["tgt"]))
banc_es = set(zip(banc_df["src"], banc_df["tgt"]))
mcns_es  = set(zip(mcns_df["src"],  mcns_df["tgt"]))
print(f"Built in {time.time()-t:.1f}s")

# ─── Initialize current set with sets for fast lookup ────────
cur_b = base["BANC"].tolist()
cur_f = base["FAFB"].tolist()
cur_m  = base["MCNS"].tolist()
cur_b_set = set(cur_b)   # for O(1) lookup
cur_f_set = set(cur_f)   # BIJECTION: FAFB IDs must be unique
cur_m_set  = set(cur_m)  # BIJECTION: MCNS IDs must be unique

# ─── Grow with bijection enforcement ─────────────────────────
print(f"\nGrowing from {len(base):,}...")
t = time.time()
added = 0
rejected = 0

for idx, row in new_cands.iterrows():
    cb, cf, cm = row["BANC"], row["FAFB"], row["MCNS"]

    # BIJECTION CHECK: all 3 IDs must be new
    if cb in cur_b_set or cf in cur_f_set or cm in cur_m_set:
        rejected += 1
        continue

    # ISOMORPHISM CHECK: no conflicts with any existing neuron
    ok = True
    for j in range(len(cur_b)):
        bb, bf, bm = cur_b[j], cur_f[j], cur_m[j]
        for sb,sf,sm,tb,tf,tm in [(cb,cf,cm,bb,bf,bm),(bb,bf,bm,cb,cf,cm)]:
            if not (((sb,tb) in banc_es)==((sf,tf) in fafb_es)==((sm,tm) in mcns_es)):
                ok = False; break
        if not ok: break

    if ok:
        cur_b.append(cb); cur_f.append(cf); cur_m.append(cm)
        cur_b_set.add(cb); cur_f_set.add(cf); cur_m_set.add(cm)
        added += 1
        if added % 100 == 0:
            print(f"  +{added} neurons ({idx}/{len(new_cands)} checked, {time.time()-t:.0f}s)")
    else:
        rejected += 1

print(f"\nDone in {time.time()-t:.1f}s")
print(f"Added: {added:,} | Rejected: {rejected:,}")

# ─── Save and verify ──────────────────────────────────────────
final = pd.DataFrame({"BANC":cur_b,"FAFB":cur_f,"MCNS":cur_m})
print(f"\n=== RESULT ===")
print(f"N = {len(final):,}")
print(f"BANC unique: {final['BANC'].nunique()} | FAFB unique: {final['FAFB'].nunique()} | MCNS unique: {final['MCNS'].nunique()}")

assert final["BANC"].nunique() == len(final), "DUPLICATE BANC!"
assert final["FAFB"].nunique() == len(final), "DUPLICATE FAFB!"
assert final["MCNS"].nunique()  == len(final), "DUPLICATE MCNS!"
print("All IDs unique ✅ — bijection satisfied")

final.to_csv("submission_FIXED.csv", index=False)
print("Saved to submission_FIXED.csv")
