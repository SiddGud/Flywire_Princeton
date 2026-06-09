"""
EXPANDED APPROACH 1: Use NBLAST matches (32k candidates instead of 12k)
EXPANDED APPROACH 2: Pure degree fingerprinting on FAFB vs MCNS
"""
import pandas as pd
import pyarrow.feather as feather
import time

print("=" * 60)
print("  APPROACH 1: NBLAST-based triplets (32k candidates)")
print("=" * 60)

meta = feather.read_feather('banc_888_meta.feather')

# NBLAST matches: computed by morphological similarity algorithm
# Less reliable than manual, but covers ~3x more neurons
nblast_triplets = meta[
    meta['fafb_nblast_match'].notna() &
    meta['malecns_nblast_match'].notna()
].copy()

nblast_triplets['BANC'] = nblast_triplets['root_626'].astype(str).str.strip()
nblast_triplets['FAFB'] = nblast_triplets['fafb_nblast_match'].astype(str).str.strip()
nblast_triplets['MCNS'] = nblast_triplets['malecns_nblast_match'].astype(str).str.split('.').str[0]

print(f"NBLAST triplets (FAFB+MCNS both present): {len(nblast_triplets):,}")

# How many are ALSO manually verified?
both_verified_nblast = meta[
    meta['fafb_match'].notna() &
    meta['malecns_match'].notna() &
    meta['fafb_nblast_match'].notna() &
    meta['malecns_nblast_match'].notna()
]
print(f"Both manually verified AND nblast: {len(both_verified_nblast):,}")
print(f"NBLAST-only (no manual match):     {len(nblast_triplets) - len(both_verified_nblast):,}")

# Filter to challenge files
print("\nLoading challenge IDs...")
t = time.time()
fafb_ids = set(pd.read_csv('fafb_783_edge_list.csv', dtype=str, header=None)[0]) | \
           set(pd.read_csv('fafb_783_edge_list.csv', dtype=str, header=None)[1])
banc_ids = set(pd.read_csv('banc_626_edge_list.csv', dtype=str, header=None)[0]) | \
           set(pd.read_csv('banc_626_edge_list.csv', dtype=str, header=None)[1])
mcns_ids = set(pd.read_csv('mcns_0.9_edge_list.csv', dtype=str, header=None)[0]) | \
           set(pd.read_csv('mcns_0.9_edge_list.csv', dtype=str, header=None)[1])
print(f"Loaded in {time.time()-t:.1f}s")

# Also filter sex-invariant only
nblast_iso = nblast_triplets[nblast_triplets['sexually_dimorphic'] == 'isomorphic']
valid_nblast = nblast_iso[
    nblast_iso['BANC'].isin(banc_ids) &
    nblast_iso['FAFB'].isin(fafb_ids) &
    nblast_iso['MCNS'].isin(mcns_ids)
].reset_index(drop=True)
print(f"Sex-invariant NBLAST triplets in challenge files: {len(valid_nblast):,}")

# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  APPROACH 2: Pure degree fingerprinting FAFB vs MCNS")
print("=" * 60)
print("Finds neurons with UNIQUE (in_deg, out_deg) in BOTH datasets")
print("No biology required — pure structural match")

fafb_df = pd.read_csv('fafb_783_edge_list.csv', dtype=str, header=None)
mcns_df  = pd.read_csv('mcns_0.9_edge_list.csv', dtype=str, header=None)
fafb_df.columns = ['src','tgt']
mcns_df.columns  = ['src','tgt']

# Compute degrees
fafb_in  = fafb_df.groupby('tgt').size().rename('in_deg')
fafb_out = fafb_df.groupby('src').size().rename('out_deg')
fafb_deg = pd.concat([fafb_in, fafb_out], axis=1).fillna(0).astype(int)
fafb_deg['sig'] = list(zip(fafb_deg['in_deg'], fafb_deg['out_deg']))

mcns_in  = mcns_df.groupby('tgt').size().rename('in_deg')
mcns_out = mcns_df.groupby('src').size().rename('out_deg')
mcns_deg = pd.concat([mcns_in, mcns_out], axis=1).fillna(0).astype(int)
mcns_deg['sig'] = list(zip(mcns_deg['in_deg'], mcns_deg['out_deg']))

# Find signatures unique in BOTH datasets (auto-matchable!)
fafb_sig_count = fafb_deg['sig'].value_counts()
mcns_sig_count  = mcns_deg['sig'].value_counts()

unique_sigs = set(fafb_sig_count[fafb_sig_count == 1].index) & \
              set(mcns_sig_count[mcns_sig_count == 1].index)

print(f"\nFAFB unique degree signatures: {(fafb_sig_count==1).sum():,} / {len(fafb_sig_count):,}")
print(f"MCNS unique degree signatures:  {(mcns_sig_count==1).sum():,} / {len(mcns_sig_count):,}")
print(f"Auto-matchable pairs (unique in BOTH): {len(unique_sigs):,}")

# Build auto-matched pairs
fafb_auto = fafb_deg[fafb_deg['sig'].isin(unique_sigs)].reset_index()
fafb_auto.columns = ['FAFB', 'in_deg', 'out_deg', 'sig']
mcns_auto  = mcns_deg[mcns_deg['sig'].isin(unique_sigs)].reset_index()
mcns_auto.columns  = ['MCNS',  'in_deg', 'out_deg', 'sig']

auto_pairs = fafb_auto.merge(mcns_auto, on='sig')[['FAFB','MCNS','sig']]
print(f"Auto-matched FAFB↔MCNS pairs: {len(auto_pairs):,}")
print("\nSample auto-matched pairs:")
print(auto_pairs.head(10).to_string())

# How many have a BANC bridge via our verified triplets?
verified = pd.read_csv('banc_fafb_mcns_triplets.csv')
verified['FAFB'] = verified['fafb_match'].astype(str)
verified['MCNS'] = verified['malecns_match'].astype(str).str.split('.').str[0]

auto_with_banc = auto_pairs.merge(verified[['FAFB','root_626']], on='FAFB', how='inner')
print(f"\nAuto-matched pairs that also have a BANC bridge: {len(auto_with_banc):,}")

print("\n=== SUMMARY OF ALL APPROACHES ===")
print(f"Approach 0 (our current):  N=4,566  (manual matches, pruned)")
print(f"Approach 1 (NBLAST):       {len(valid_nblast):,} starting candidates")
print(f"Approach 2 (fingerprint):  {len(auto_pairs):,} auto-matched FAFB↔MCNS pairs")
print(f"\nBest bet to improve N: run pruner on NBLAST candidates")
print(f"or combine fingerprint matches + BANC bridge")
