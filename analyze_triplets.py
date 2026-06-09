"""
Analyze BANC metadata: find triplets with FAFB + MCNS verified matches.
This is the core of our submission strategy.
"""
import pyarrow.feather as feather
import pandas as pd

print("Loading banc_888_meta.feather...")
df = feather.read_feather('banc_888_meta.feather')
print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")

# ─── Cross-dataset match columns ─────────────────────────────
print("\n=== CROSS-DATASET MATCH COLUMNS ===")
match_cols = ['fafb_match', 'malecns_match', 'manc_match',
              'fafb_nblast_match', 'malecns_nblast_match']
for col in match_cols:
    n = df[col].notna().sum()
    print(f"  {col:<35} {n:>8,} / {len(df):,}")

# ─── Check BANC ID columns ───────────────────────────────────
print("\n=== BANC ID COLUMNS ===")
print("root_626 sample:", df['root_626'].head(3).tolist())
print("root_888 sample:", df['root_888'].head(3).tolist())

# ─── Find verified triplets ───────────────────────────────────
triplets_verified = df[df['fafb_match'].notna() & df['malecns_match'].notna()].copy()
print(f"\n=== VERIFIED TRIPLETS (BANC + FAFB + MCNS) ===")
print(f"Neurons with BOTH fafb_match AND malecns_match: {len(triplets_verified):,}")

triplets_nblast = df[df['fafb_nblast_match'].notna() & df['malecns_nblast_match'].notna()]
print(f"Neurons with BOTH nblast matches:               {len(triplets_nblast):,}")

triplets_any = df[
    (df['fafb_match'].notna() | df['fafb_nblast_match'].notna()) &
    (df['malecns_match'].notna() | df['malecns_nblast_match'].notna())
]
print(f"Neurons with ANY fafb match AND ANY mcns match: {len(triplets_any):,}")

# ─── Sample triplets ─────────────────────────────────────────
print("\n=== SAMPLE VERIFIED TRIPLETS (first 10) ===")
cols = ['root_626', 'fafb_match', 'malecns_match', 'cell_type', 'neurotransmitter_predicted']
print(triplets_verified[cols].head(10).to_string())

# ─── What cell types are in the triplets? ────────────────────
print("\n=== CELL TYPES IN VERIFIED TRIPLETS ===")
print(triplets_verified['cell_type'].value_counts().head(20).to_string())

print("\n=== SUPER CLASSES IN VERIFIED TRIPLETS ===")
print(triplets_verified['super_class'].value_counts().head(10).to_string())

# ─── Cross-check with challenge file ─────────────────────────
print("\n=== CROSS-CHECK WITH CHALLENGE BANC FILE ===")
banc_df = pd.read_csv('banc_626_edge_list.csv')
banc_df.columns = ['src', 'tgt']
banc_ids = set(banc_df['src']) | set(banc_df['tgt'])
print(f"BANC challenge file unique neuron IDs: {len(banc_ids):,}")

# Try matching root_626
try:
    meta_root626 = set(df['root_626'].dropna().astype(int))
    overlap = banc_ids & meta_root626
    print(f"Overlap with root_626: {len(overlap):,} ({100*len(overlap)/len(banc_ids):.1f}%)")
except Exception as e:
    print(f"root_626 overlap error: {e}")
    print("root_626 dtype:", df['root_626'].dtype)
    print("banc_ids sample:", list(banc_ids)[:3])

# Save the triplets to CSV for the submission pipeline
out_cols = ['root_626', 'root_888', 'fafb_match', 'malecns_match',
            'cell_type', 'super_class', 'neurotransmitter_predicted',
            'sexually_dimorphic', 'region']
triplets_verified[out_cols].to_csv('banc_fafb_mcns_triplets.csv', index=False)
print(f"\nSaved {len(triplets_verified):,} triplets to banc_fafb_mcns_triplets.csv")

print("\n=== FAFB match sample values ===")
print(triplets_verified['fafb_match'].dropna().head(5).tolist())
print("\n=== MCNS match sample values ===")
print(triplets_verified['malecns_match'].dropna().head(5).tolist())
