"""
GROW FROM NBLAST BASE — Different starting point, might reach different peak
============================================================================
Strategy: Start from NBLAST-verified 3,822 neurons, grow with ALL candidates.
Then compare with manual-grown 5,739. Take the best.
"""
import pandas as pd
import pyarrow.feather as feather
import time

print("=" * 60)
print("  GROWING FROM NBLAST BASE (different seed)")
print("=" * 60)

# ─── Load NBLAST base set ─────────────────────────────────────
print("\nLoading NBLAST verified base set (3,822 neurons)...")
base = pd.read_csv('submission_NBLAST.csv', dtype=str)
print(f"Base set: {len(base):,} neurons")

# ─── Load ALL candidate triplets (manual + NBLAST) ────────────
print("Loading ALL candidates from metadata...")
meta = feather.read_feather('banc_888_meta.feather')

pool1 = meta[(meta['sexually_dimorphic']=='isomorphic') &
              meta['fafb_match'].notna() & meta['malecns_match'].notna()].copy()
pool1['BANC'] = pool1['root_626'].astype(str).str.strip()
pool1['FAFB'] = pool1['fafb_match'].astype(str).str.strip()
pool1['MCNS'] = pool1['malecns_match'].astype(str).str.split('.').str[0]

pool2 = meta[(meta['sexually_dimorphic']=='isomorphic') &
              meta['fafb_nblast_match'].notna() & meta['malecns_nblast_match'].notna()].copy()
pool2['BANC'] = pool2['root_626'].astype(str).str.strip()
pool2['FAFB'] = pool2['fafb_nblast_match'].astype(str).str.strip()
pool2['MCNS'] = pool2['malecns_nblast_match'].astype(str).str.split('.').str[0]

all_cands = pd.concat([pool1[['BANC','FAFB','MCNS']],
                        pool2[['BANC','FAFB','MCNS']]]).drop_duplicates(subset=['BANC']).reset_index(drop=True)

base_banc = set(base['BANC'])
new_cands = all_cands[~all_cands['BANC'].isin(base_banc)].reset_index(drop=True)
print(f"New candidates to try: {len(new_cands):,}")

# ─── Load edge lists ──────────────────────────────────────────
print("Loading edge lists...")
t0 = time.time()
def load_str(f):
    df = pd.read_csv(f, dtype=str, header=None)
    df.columns = ['src','tgt']
    df['src'] = df['src'].str.strip(); df['tgt'] = df['tgt'].str.strip()
    return df

fafb_df = load_str('fafb_783_edge_list.csv')
banc_df = load_str('banc_626_edge_list.csv')
mcns_df  = load_str('mcns_0.9_edge_list.csv')
print(f"Loaded in {time.time()-t0:.1f}s")

fafb_ids = set(fafb_df['src'])|set(fafb_df['tgt'])
banc_ids = set(banc_df['src'])|set(banc_df['tgt'])
mcns_ids  = set(mcns_df['src'])|set(mcns_df['tgt'])

new_cands = new_cands[
    new_cands['BANC'].isin(banc_ids) &
    new_cands['FAFB'].isin(fafb_ids) &
    new_cands['MCNS'].isin(mcns_ids)
].drop_duplicates(subset=['FAFB']).drop_duplicates(subset=['MCNS']).reset_index(drop=True)
print(f"Candidates in challenge files: {len(new_cands):,}")

# ─── Build full edge lookup sets ─────────────────────────────
print("Building edge sets...")
t = time.time()
fafb_es = set(zip(fafb_df['src'], fafb_df['tgt']))
banc_es = set(zip(banc_df['src'], banc_df['tgt']))
mcns_es  = set(zip(mcns_df['src'],  mcns_df['tgt']))
print(f"Built in {time.time()-t:.1f}s")

# ─── Grow ────────────────────────────────────────────────────
cur_banc = base['BANC'].tolist()
cur_fafb = base['FAFB'].tolist()
cur_mcns  = base['MCNS'].tolist()

print(f"\nGrowing from {len(base):,}...")
t = time.time()
added = 0

for idx, row in new_cands.iterrows():
    cb, cf, cm = row['BANC'], row['FAFB'], row['MCNS']
    ok = True
    for j in range(len(cur_banc)):
        bb, bf, bm = cur_banc[j], cur_fafb[j], cur_mcns[j]
        for src_b,src_f,src_m, tgt_b,tgt_f,tgt_m in [
            (cb,cf,cm,bb,bf,bm), (bb,bf,bm,cb,cf,cm)
        ]:
            if not (((src_b,tgt_b) in banc_es) == ((src_f,tgt_f) in fafb_es) == ((src_m,tgt_m) in mcns_es)):
                ok = False; break
        if not ok: break
    if ok:
        cur_banc.append(cb); cur_fafb.append(cf); cur_mcns.append(cm)
        added += 1
        if added % 50 == 0:
            print(f"  Added {added} ({idx}/{len(new_cands)} checked, {time.time()-t:.0f}s)")

print(f"\nDone: added {added:,} → N = {len(cur_banc):,}")

# ─── Compare with manual-grown and take best ─────────────────
nblast_grown = pd.DataFrame({'BANC':cur_banc,'FAFB':cur_fafb,'MCNS':cur_mcns})
nblast_grown.to_csv('submission_NBLAST_GROWN.csv', index=False)

manual_grown = pd.read_csv('submission_GROWN.csv', dtype=str)
print(f"\n=== COMPARISON ===")
print(f"Manual-grown:  N = {len(manual_grown):,}")
print(f"NBLAST-grown:  N = {len(nblast_grown):,}")

# Overlap
overlap = len(set(nblast_grown['BANC']) & set(manual_grown['BANC']))
print(f"Overlap:       {overlap:,} neurons in common")
print(f"Unique to manual: {len(set(manual_grown['BANC'])-set(nblast_grown['BANC'])):,}")
print(f"Unique to NBLAST: {len(set(nblast_grown['BANC'])-set(manual_grown['BANC'])):,}")

best = manual_grown if len(manual_grown) >= len(nblast_grown) else nblast_grown
best_n = max(len(manual_grown), len(nblast_grown))
print(f"\nBest submission: N = {best_n:,}")
best.to_csv('submission_BEST.csv', index=False)
print("Saved to submission_BEST.csv")
