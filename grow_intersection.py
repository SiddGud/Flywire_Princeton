"""
GROW FROM INTERSECTION SEED
============================
The manual-grown (5739) and NBLAST-grown (4659) share 2,396 neurons.
This overlap is definitely valid. If we grow from JUST these 2,396
(without the blockers from either set), we might reach a THIRD local
maximum that exceeds 5,739 by including unique neurons from BOTH sets.
"""
import pandas as pd
import pyarrow.feather as feather
import time

print("=" * 60)
print("  GROW FROM INTERSECTION (neutral seed)")
print("=" * 60)

# ─── Find intersection of the two grown sets ──────────────────
manual = pd.read_csv('submission_GROWN.csv', dtype=str)
nblast = pd.read_csv('submission_NBLAST_GROWN.csv', dtype=str)

# Intersection = BANC IDs in both
common_banc = set(manual['BANC']) & set(nblast['BANC'])
base = manual[manual['BANC'].isin(common_banc)].reset_index(drop=True)
print(f"Intersection (neutral seed): {len(base):,} neurons")
print(f"Manual-grown total:          {len(manual):,}")
print(f"NBLAST-grown total:          {len(nblast):,}")
print(f"Neurons to potentially add:  {len(manual)-len(base):,} from manual + {len(nblast)-len(base):,} from NBLAST")

# ─── Load ALL candidates ──────────────────────────────────────
print("\nLoading all candidates...")
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

base_set = set(base['BANC'])
new_cands = all_cands[~all_cands['BANC'].isin(base_set)].reset_index(drop=True)

# ─── Load edge lists ──────────────────────────────────────────
print("Loading edge lists...")
def load_str(f):
    df = pd.read_csv(f, dtype=str, header=None)
    df.columns = ['src','tgt']
    df['src'] = df['src'].str.strip(); df['tgt'] = df['tgt'].str.strip()
    return df

fafb_df = load_str('fafb_783_edge_list.csv')
banc_df = load_str('banc_626_edge_list.csv')
mcns_df  = load_str('mcns_0.9_edge_list.csv')

fafb_ids = set(fafb_df['src'])|set(fafb_df['tgt'])
banc_ids = set(banc_df['src'])|set(banc_df['tgt'])
mcns_ids  = set(mcns_df['src'])|set(mcns_df['tgt'])

new_cands = new_cands[
    new_cands['BANC'].isin(banc_ids) &
    new_cands['FAFB'].isin(fafb_ids) &
    new_cands['MCNS'].isin(mcns_ids)
].drop_duplicates(subset=['FAFB']).drop_duplicates(subset=['MCNS']).reset_index(drop=True)
print(f"Candidates to try: {len(new_cands):,}")

fafb_es = set(zip(fafb_df['src'], fafb_df['tgt']))
banc_es = set(zip(banc_df['src'], banc_df['tgt']))
mcns_es  = set(zip(mcns_df['src'],  mcns_df['tgt']))

# ─── PRIORITIZE: Try manual-unique and NBLAST-unique first ───
manual_unique = set(manual['BANC']) - common_banc
nblast_unique = set(nblast['BANC']) - common_banc
priority_cands = new_cands[new_cands['BANC'].isin(manual_unique | nblast_unique)]
other_cands = new_cands[~new_cands['BANC'].isin(manual_unique | nblast_unique)]
ordered_cands = pd.concat([priority_cands, other_cands]).reset_index(drop=True)
print(f"Priority candidates (from both grown sets): {len(priority_cands):,}")

# ─── Grow ────────────────────────────────────────────────────
cur_b = base['BANC'].tolist()
cur_f = base['FAFB'].tolist()
cur_m  = base['MCNS'].tolist()

print(f"\nGrowing from intersection of {len(base):,}...")
t = time.time()
added = 0

for idx, row in ordered_cands.iterrows():
    cb, cf, cm = row['BANC'], row['FAFB'], row['MCNS']
    ok = True
    for j in range(len(cur_b)):
        bb, bf, bm = cur_b[j], cur_f[j], cur_m[j]
        for s_b,s_f,s_m,t_b,t_f,t_m in [(cb,cf,cm,bb,bf,bm),(bb,bf,bm,cb,cf,cm)]:
            if not (((s_b,t_b) in banc_es)==((s_f,t_f) in fafb_es)==((s_m,t_m) in mcns_es)):
                ok = False; break
        if not ok: break
    if ok:
        cur_b.append(cb); cur_f.append(cf); cur_m.append(cm)
        added += 1
        if added % 100 == 0:
            print(f"  +{added} neurons ({idx}/{len(ordered_cands)} checked, {time.time()-t:.0f}s)")

final_n = len(cur_b)
print(f"\nDone: added {added:,} → N = {final_n:,}")

# ─── Save and compare ─────────────────────────────────────────
result = pd.DataFrame({'BANC':cur_b,'FAFB':cur_f,'MCNS':cur_m})
result.to_csv('submission_INTERSECTION_GROWN.csv', index=False)

print(f"\n=== FINAL SCOREBOARD ===")
print(f"Manual-grown:       N = {len(manual):,}")
print(f"NBLAST-grown:       N = {len(nblast):,}")
print(f"Intersection-grown: N = {final_n:,}")
print(f"Current best:       N = {max(len(manual), len(nblast), final_n):,}")

# Save best overall
best_n = max(len(manual), len(nblast), final_n)
if final_n == best_n:
    result.to_csv('submission_BEST.csv', index=False)
    print("→ Intersection-grown is the new best! submission_BEST.csv updated.")
else:
    print(f"→ Manual-grown remains best (N=5,739). submission_BEST.csv unchanged.")
