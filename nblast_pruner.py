"""
NBLAST PRUNER — 24,970 candidates instead of 10,808
Same algorithm as clean_verify.py but using computational matches.
Expected final N: hopefully 7,000-10,000.
"""
import pandas as pd
import pyarrow.feather as feather
import time
from collections import defaultdict

print("=" * 60)
print("  NBLAST PRUNER (24,970 candidates)")
print("=" * 60)

# ─── Load metadata with NBLAST matches ───────────────────────
print("\nLoading metadata...")
meta = feather.read_feather('banc_888_meta.feather')

# Use NBLAST matches (computational, 3x coverage vs manual)
nblast = meta[
    (meta['sexually_dimorphic'] == 'isomorphic') &
    meta['fafb_nblast_match'].notna() &
    meta['malecns_nblast_match'].notna()
].copy()

nblast['BANC'] = nblast['root_626'].astype(str).str.strip()
nblast['FAFB'] = nblast['fafb_nblast_match'].astype(str).str.strip()
nblast['MCNS'] = nblast['malecns_nblast_match'].astype(str).str.split('.').str[0]

print(f"Sex-invariant NBLAST triplets: {len(nblast):,}")

# ─── Load challenge edge lists (all STRING) ───────────────────
print("Loading edge lists...")
t0 = time.time()

def load_str(f):
    df = pd.read_csv(f, dtype=str, header=None)
    df.columns = ['src','tgt']
    df['src'] = df['src'].str.strip()
    df['tgt'] = df['tgt'].str.strip()
    return df

fafb_df = load_str('fafb_783_edge_list.csv')
banc_df = load_str('banc_626_edge_list.csv')
mcns_df  = load_str('mcns_0.9_edge_list.csv')
print(f"Loaded in {time.time()-t0:.1f}s")

fafb_ids = set(fafb_df['src']) | set(fafb_df['tgt'])
banc_ids = set(banc_df['src']) | set(banc_df['tgt'])
mcns_ids  = set(mcns_df['src'])  | set(mcns_df['tgt'])

# ─── Filter to challenge files ────────────────────────────────
valid = nblast[
    nblast['BANC'].isin(banc_ids) &
    nblast['FAFB'].isin(fafb_ids) &
    nblast['MCNS'].isin(mcns_ids)
].reset_index(drop=True)

# Drop duplicates (BANC, FAFB, or MCNS IDs shouldn't appear twice)
valid = valid.drop_duplicates(subset=['BANC']).drop_duplicates(subset=['FAFB']).drop_duplicates(subset=['MCNS']).reset_index(drop=True)
print(f"Valid NBLAST triplets in all 3 challenge files: {len(valid):,}")

banc_list = valid['BANC'].tolist()
fafb_list = valid['FAFB'].tolist()
mcns_list  = valid['MCNS'].tolist()

# ─── Build index-based edge sets (string keys) ────────────────
print("Building internal edge sets...")
t = time.time()

b2i = {b: i for i, b in enumerate(banc_list)}
f2i = {f: i for i, f in enumerate(fafb_list)}
m2i = {m: i for i, m in enumerate(mcns_list)}

banc_int = banc_df[banc_df['src'].isin(b2i) & banc_df['tgt'].isin(b2i)]
fafb_int = fafb_df[fafb_df['src'].isin(f2i) & fafb_df['tgt'].isin(f2i)]
mcns_int  = mcns_df[mcns_df['src'].isin(m2i)  & mcns_df['tgt'].isin(m2i)]

banc_eidx = set((b2i[s], b2i[t]) for s,t in zip(banc_int['src'], banc_int['tgt']))
fafb_eidx = set((f2i[s], f2i[t]) for s,t in zip(fafb_int['src'], fafb_int['tgt']))
mcns_eidx  = set((m2i[s],  m2i[t])  for s,t in zip(mcns_int['src'],  mcns_int['tgt']))

print(f"Internal edges — BANC:{len(banc_eidx):,} FAFB:{len(fafb_eidx):,} MCNS:{len(mcns_eidx):,}")
print(f"Built in {time.time()-t:.1f}s")

# ─── Sanity check ─────────────────────────────────────────────
if len(banc_eidx) + len(fafb_eidx) + len(mcns_eidx) == 0:
    print("ERROR: Edge sets empty — ID mismatch!")
    print("BANC sample:", banc_list[:2], "| banc_df sample:", list(banc_df['src'])[:2])
    exit(1)

# ─── Count initial conflicts ──────────────────────────────────
all_eidx = banc_eidx | fafb_eidx | mcns_eidx
init_conflicts = sum(
    1 for (i,j) in all_eidx
    if not (((i,j) in banc_eidx) == ((i,j) in fafb_eidx) == ((i,j) in mcns_eidx))
)
print(f"\nInitial conflicts: {init_conflicts:,} / {len(all_eidx):,}")
print(f"Agreement rate: {100*(1-init_conflicts/max(1,len(all_eidx))):.2f}%")

# ─── Greedy pruning ───────────────────────────────────────────
print(f"\n=== GREEDY PRUNING (NBLAST candidates) ===")
t = time.time()
active = set(range(len(valid)))
b_e = set(banc_eidx)
f_e = set(fafb_eidx)
m_e = set(mcns_eidx)
iteration = 0

while True:
    all_e = b_e | f_e | m_e
    conflicts = defaultdict(int)
    total = 0
    for (i,j) in all_e:
        if i not in active or j not in active: continue
        b = (i,j) in b_e
        f = (i,j) in f_e
        m = (i,j) in m_e
        if not (b == f == m):
            total += 1
            conflicts[i] += 1
            conflicts[j] += 1
    if total == 0:
        print(f"\n✅ Converged after {iteration} removals!")
        break
    worst = max(conflicts, key=conflicts.get)
    active.discard(worst)
    b_e = {e for e in b_e if e[0] in active and e[1] in active}
    f_e = {e for e in f_e if e[0] in active and e[1] in active}
    m_e = {e for e in m_e if e[0] in active and e[1] in active}
    iteration += 1
    if iteration % 200 == 0:
        print(f"  iter {iteration:5d}: {len(active):,} neurons, {total:,} conflicts ({time.time()-t:.0f}s)")

# ─── Save ─────────────────────────────────────────────────────
final = valid.iloc[sorted(active)][['BANC','FAFB','MCNS']].copy()
print(f"\n=== FINAL RESULT ===")
print(f"N = {len(final):,} neurons (vs 4,566 from manual matches)")
print(f"Improvement: +{len(final)-4566:,} neurons")
final.to_csv('submission_NBLAST.csv', index=False)
print(f"Saved to submission_NBLAST.csv")

# Compare with current submission
current = pd.read_csv('submission_FINAL_VALID.csv', dtype=str)
overlap = len(set(final['BANC']) & set(current['BANC']))
print(f"\nOverlap with current manual submission: {overlap:,} neurons in common")
print(f"New neurons found by NBLAST: {len(final) - overlap:,}")
