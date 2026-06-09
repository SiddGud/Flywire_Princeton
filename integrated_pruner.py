"""
INTEGRATED BIJECTION + ISOMORPHISM PRUNER
==========================================
Instead of deduplicating upfront (which loses candidates), we start
with all 10,808 neurons and let the pruner resolve BOTH types of conflict:
  1. Edge conflicts (edge exists in A but not B)
  2. Bijection conflicts (two neurons share same FAFB or MCNS ID)

The pruner removes the neuron causing the most TOTAL conflicts at each step.
This is smarter than pre-dedup because it picks the BETTER representative
when two BANC neurons map to the same FAFB/MCNS neuron.
"""
import pandas as pd
import pyarrow.feather as feather
import time
from collections import defaultdict

print("=" * 60)
print("  INTEGRATED PRUNER (edge + bijection conflicts)")
print("=" * 60)

# ─── Load all sex-invariant triplets (no pre-dedup) ──────────
meta = feather.read_feather("banc_888_meta.feather")

valid_meta = meta[
    (meta["sexually_dimorphic"] == "isomorphic") &
    meta["fafb_match"].notna() &
    meta["malecns_match"].notna()
].copy()

valid_meta["BANC"] = valid_meta["root_626"].astype(str).str.strip()
valid_meta["FAFB"] = valid_meta["fafb_match"].astype(str).str.strip()
valid_meta["MCNS"] = valid_meta["malecns_match"].astype(str).str.split(".").str[0]

print(f"Starting pool: {len(valid_meta):,} rows")
print(f"  BANC unique: {valid_meta['BANC'].nunique():,}")
print(f"  FAFB unique: {valid_meta['FAFB'].nunique():,}")
print(f"  MCNS unique: {valid_meta['MCNS'].nunique():,}")

# ─── Load edge lists ──────────────────────────────────────────
print("\nLoading edge lists...")
def load_str(f):
    df = pd.read_csv(f, dtype=str, header=None)
    df.columns = ["src","tgt"]
    df["src"] = df["src"].str.strip(); df["tgt"] = df["tgt"].str.strip()
    return df

fafb_df = load_str("fafb_783_edge_list.csv")
banc_df = load_str("banc_626_edge_list.csv")
mcns_df  = load_str("mcns_0.9_edge_list.csv")

fafb_ids = set(fafb_df["src"])|set(fafb_df["tgt"])
banc_ids = set(banc_df["src"])|set(banc_df["tgt"])
mcns_ids  = set(mcns_df["src"])|set(mcns_df["tgt"])

# Filter to challenge files, keep BANC unique (but allow FAFB/MCNS dups)
valid = valid_meta[
    valid_meta["BANC"].isin(banc_ids) &
    valid_meta["FAFB"].isin(fafb_ids) &
    valid_meta["MCNS"].isin(mcns_ids)
].drop_duplicates(subset=["BANC"]).reset_index(drop=True)  # BANC must be unique
print(f"In challenge files (BANC-unique): {len(valid):,}")

banc_list = valid["BANC"].tolist()
fafb_list = valid["FAFB"].tolist()
mcns_list  = valid["MCNS"].tolist()

# ─── Build edge index sets ────────────────────────────────────
print("Building edge index sets...")
t = time.time()
b2i = {b:i for i,b in enumerate(banc_list)}
f2i = {f:i for i,f in enumerate(fafb_list)}
m2i = {m:i for i,m in enumerate(mcns_list)}

bi = banc_df[banc_df["src"].isin(b2i) & banc_df["tgt"].isin(b2i)]
fi = fafb_df[fafb_df["src"].isin(f2i) & fafb_df["tgt"].isin(f2i)]
mi = mcns_df[mcns_df["src"].isin(m2i)  & mcns_df["tgt"].isin(m2i)]

banc_ei = set((b2i[s],b2i[t]) for s,t in zip(bi["src"],bi["tgt"]))
fafb_ei = set((f2i[s],f2i[t]) for s,t in zip(fi["src"],fi["tgt"]))
mcns_ei  = set((m2i[s],m2i[t]) for s,t in zip(mi["src"],mi["tgt"]))
print(f"Edges BANC:{len(banc_ei):,} FAFB:{len(fafb_ei):,} MCNS:{len(mcns_ei):,} ({time.time()-t:.1f}s)")

# ─── Build FAFB and MCNS duplicate maps ──────────────────────
# fafb_groups[fafb_id] = [list of neuron indices with that FAFB ID]
fafb_groups = defaultdict(list)
mcns_groups  = defaultdict(list)
for i, (f, m) in enumerate(zip(fafb_list, mcns_list)):
    fafb_groups[f].append(i)
    mcns_groups[m].append(i)

init_bijection_conflicts = sum(len(v)-1 for v in fafb_groups.values() if len(v)>1) + \
                            sum(len(v)-1 for v in mcns_groups.values()  if len(v)>1)
print(f"\nInitial bijection violations: {init_bijection_conflicts:,} extra neurons to remove")

# ─── Integrated pruning ───────────────────────────────────────
print("\n=== INTEGRATED PRUNING ===")
t0 = time.time()
active = set(range(len(valid)))
b_e, f_e, m_e = set(banc_ei), set(fafb_ei), set(mcns_ei)

for iteration in range(200000):
    all_e = b_e | f_e | m_e
    cfct = defaultdict(int)
    total_edge = 0
    total_bijection = 0

    # Edge conflicts
    for (i,j) in all_e:
        if i not in active or j not in active: continue
        if not (((i,j) in b_e)==((i,j) in f_e)==((i,j) in m_e)):
            total_edge += 1; cfct[i]+=1; cfct[j]+=1

    # Bijection conflicts (FAFB duplicates)
    for f_id, idxs in fafb_groups.items():
        active_idxs = [k for k in idxs if k in active]
        if len(active_idxs) > 1:
            for k in active_idxs:
                cfct[k] += len(active_idxs) - 1
                total_bijection += 1

    # Bijection conflicts (MCNS duplicates)
    for m_id, idxs in mcns_groups.items():
        active_idxs = [k for k in idxs if k in active]
        if len(active_idxs) > 1:
            for k in active_idxs:
                cfct[k] += len(active_idxs) - 1
                total_bijection += 1

    total = total_edge + total_bijection
    if total == 0:
        print(f"\n✅ Converged after {iteration} removals!")
        break
    worst = max(cfct, key=cfct.get)
    active.discard(worst)
    b_e = {e for e in b_e if e[0] in active and e[1] in active}
    f_e = {e for e in f_e if e[0] in active and e[1] in active}
    m_e = {e for e in m_e if e[0] in active and e[1] in active}

    if iteration % 200 == 0:
        elapsed = time.time()-t0
        print(f"  iter {iteration:5d}: {len(active):,} active, edge:{total_edge:,} bijection:{total_bijection:,} ({elapsed:.0f}s)")

# ─── Extract result ───────────────────────────────────────────
final = valid.iloc[sorted(active)][["BANC","FAFB","MCNS"]].copy()
print(f"\n=== PRUNED RESULT ===")
print(f"N = {len(final):,}")
print(f"BANC unique: {final['BANC'].nunique()} | FAFB unique: {final['FAFB'].nunique()} | MCNS unique: {final['MCNS'].nunique()}")
assert final["BANC"].nunique()==len(final) and final["FAFB"].nunique()==len(final) and final["MCNS"].nunique()==len(final)
print("Bijection satisfied ✅")
final.to_csv("submission_INTEGRATED.csv", index=False)

# ─── Compare ─────────────────────────────────────────────────
clean = pd.read_csv("submission_CLEAN_FINAL.csv", dtype=str)
print(f"\n=== COMPARISON ===")
print(f"Pre-dedup pipeline: N = {len(clean):,}")
print(f"Integrated pruner:  N = {len(final):,}")
if len(final) > len(clean):
    print(f"Integrated is BETTER by +{len(final)-len(clean):,} ✅")
    final.to_csv("submission_BEST.csv", index=False)
    print("Saved to submission_BEST.csv")
else:
    print(f"Pre-dedup pipeline wins, keeping that as best")
    clean.to_csv("submission_BEST.csv", index=False)
