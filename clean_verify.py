"""
CLEAN FINAL VERIFIER + PRUNER
==============================
Fixes the type mismatch bug. Everything is consistently STRING.
Verifies the 10,797 sex-invariant triplets for true isomorphism.
"""
import pandas as pd
import pyarrow.feather as feather
import time
from collections import defaultdict

print("=" * 60)
print("  CLEAN VERIFIER + PRUNER")
print("=" * 60)

# ─── Load data — all IDs forced to STRING ────────────────────
print("\nLoading BANC metadata...")
meta = feather.read_feather('banc_888_meta.feather')

# Get sex-invariant (isomorphic) neurons
meta_isomorphic = meta[meta['sexually_dimorphic'] == 'isomorphic'].copy()
print(f"Sex-invariant BANC neurons: {len(meta_isomorphic):,}")

# Build triplets from metadata (string IDs)
meta_isomorphic = meta_isomorphic[
    meta_isomorphic['fafb_match'].notna() &
    meta_isomorphic['malecns_match'].notna()
].copy()

meta_isomorphic['BANC'] = meta_isomorphic['root_626'].astype(str).str.strip()
meta_isomorphic['FAFB'] = meta_isomorphic['fafb_match'].astype(str).str.strip()
meta_isomorphic['MCNS'] = meta_isomorphic['malecns_match'].astype(str).str.split('.').str[0]  # remove .0

print(f"Sex-invariant triplets with FAFB+MCNS matches: {len(meta_isomorphic):,}")

# ─── Load challenge edge lists (all STRING) ───────────────────
print("\nLoading edge lists as strings...")
t = time.time()

def load_edges_as_str(filename):
    df = pd.read_csv(filename, dtype=str)
    df.columns = ['src', 'tgt']
    df['src'] = df['src'].str.strip()
    df['tgt'] = df['tgt'].str.strip()
    return df

fafb_df = load_edges_as_str('fafb_783_edge_list.csv')
banc_df = load_edges_as_str('banc_626_edge_list.csv')
mcns_df  = load_edges_as_str('mcns_0.9_edge_list.csv')
print(f"Loaded in {time.time()-t:.1f}s")

# Build global ID sets (all strings)
fafb_ids = set(fafb_df['src']) | set(fafb_df['tgt'])
banc_ids = set(banc_df['src']) | set(banc_df['tgt'])
mcns_ids  = set(mcns_df['src'])  | set(mcns_df['tgt'])

# ─── Filter triplets to those present in challenge files ──────
in_banc = meta_isomorphic['BANC'].isin(banc_ids)
in_fafb = meta_isomorphic['FAFB'].isin(fafb_ids)
in_mcns = meta_isomorphic['MCNS'].isin(mcns_ids)

valid = meta_isomorphic[in_banc & in_fafb & in_mcns].reset_index(drop=True)
print(f"\nSex-invariant triplets in ALL 3 challenge files: {len(valid):,}")
print(f"  (Removed {len(meta_isomorphic)-len(valid):,} not found in challenge CSVs)")

banc_list = valid['BANC'].tolist()
fafb_list = valid['FAFB'].tolist()
mcns_list  = valid['MCNS'].tolist()

# ─── Build index-based edge sets ──────────────────────────────
print("\nBuilding internal edge sets...")
t = time.time()

b2i = {b: i for i, b in enumerate(banc_list)}
f2i = {f: i for i, f in enumerate(fafb_list)}
m2i = {m: i for i, m in enumerate(mcns_list)}

# Only keep edges where BOTH endpoints are in our matched set
banc_int = banc_df[banc_df['src'].isin(b2i) & banc_df['tgt'].isin(b2i)]
fafb_int = fafb_df[fafb_df['src'].isin(f2i) & fafb_df['tgt'].isin(f2i)]
mcns_int  = mcns_df[mcns_df['src'].isin(m2i)  & mcns_df['tgt'].isin(m2i)]

banc_eidx = set((b2i[s], b2i[t]) for s,t in zip(banc_int['src'], banc_int['tgt']))
fafb_eidx = set((f2i[s], f2i[t]) for s,t in zip(fafb_int['src'], fafb_int['tgt']))
mcns_eidx  = set((m2i[s],  m2i[t])  for s,t in zip(mcns_int['src'],  mcns_int['tgt']))

print(f"Internal edges — BANC:{len(banc_eidx):,} FAFB:{len(fafb_eidx):,} MCNS:{len(mcns_eidx):,}")
print(f"Built in {time.time()-t:.1f}s")

if len(banc_eidx) + len(fafb_eidx) + len(mcns_eidx) == 0:
    print("ERROR: All edge sets empty — ID format mismatch. Debugging...")
    print("Sample BANC list:", banc_list[:3])
    print("Sample BANC df src:", list(banc_df['src'])[:3])
    exit(1)

# ─── Count initial conflicts ──────────────────────────────────
all_eidx = banc_eidx | fafb_eidx | mcns_eidx
init_conflicts = 0
for (i,j) in all_eidx:
    b = (i,j) in banc_eidx
    f = (i,j) in fafb_eidx
    m = (i,j) in mcns_eidx
    if not (b == f == m):
        init_conflicts += 1

total_internal_pairs = len(all_eidx)
print(f"\nInitial conflicts: {init_conflicts:,} / {total_internal_pairs:,} edge positions")
print(f"Agreement rate: {100*(1-init_conflicts/max(1,total_internal_pairs)):.2f}%")
print(f"Violation types:")
banc_only = sum(1 for e in all_eidx if (e in banc_eidx) and (e not in fafb_eidx) and (e not in mcns_eidx))
fafb_only = sum(1 for e in all_eidx if (e not in banc_eidx) and (e in fafb_eidx) and (e not in mcns_eidx))
mcns_only = sum(1 for e in all_eidx if (e not in banc_eidx) and (e not in fafb_eidx) and (e in mcns_eidx))
print(f"  BANC-only (3-4 syn): {banc_only:,}")
print(f"  FAFB-only:           {fafb_only:,}")
print(f"  MCNS-only:            {mcns_only:,}")

if init_conflicts == 0:
    print("\n✅ PERFECT ISOMORPHISM with 0 removals!")
    final = valid
else:
    # ─── Greedy pruning ───────────────────────────────────────
    print(f"\n=== GREEDY PRUNING ===")
    t = time.time()
    active = set(range(len(valid)))
    b_edges = set(banc_eidx)
    f_edges = set(fafb_eidx)
    m_edges = set(mcns_eidx)
    iteration = 0

    while True:
        all_e = b_edges | f_edges | m_edges
        conflicts = defaultdict(int)
        total = 0
        for (i,j) in all_e:
            if i not in active or j not in active: continue
            b = (i,j) in b_edges
            f = (i,j) in f_edges
            m = (i,j) in m_edges
            if not (b == f == m):
                total += 1
                conflicts[i] += 1
                conflicts[j] += 1
        if total == 0:
            print(f"\n✅ Converged after {iteration} removals! 0 conflicts.")
            break
        worst = max(conflicts, key=conflicts.get)
        active.discard(worst)
        b_edges = {e for e in b_edges if e[0] in active and e[1] in active}
        f_edges = {e for e in f_edges if e[0] in active and e[1] in active}
        m_edges = {e for e in m_edges if e[0] in active and e[1] in active}
        iteration += 1
        if iteration % 100 == 0:
            print(f"  iter {iteration}: {len(active):,} neurons, {total:,} conflicts ({time.time()-t:.0f}s)")

    final = valid.iloc[sorted(active)].copy()

# ─── Save and report ──────────────────────────────────────────
print(f"\n=== FINAL RESULT ===")
print(f"N = {len(final):,} perfectly isomorphic neurons!")
final[['BANC','FAFB','MCNS']].to_csv('submission_FINAL_VALID.csv', index=False)
print(f"Saved to submission_FINAL_VALID.csv")
print(final[['BANC','FAFB','MCNS']].head(5).to_string())
