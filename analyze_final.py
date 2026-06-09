"""
Analyze the final 4,566 verified triplets — understand composition
and diagnose why so many neurons were pruned.
"""
import pandas as pd
import pyarrow.feather as feather

print("=" * 60)
print("  ANALYZING FINAL SUBMISSION")
print("=" * 60)

# Load final submission
final = pd.read_csv('submission_FINAL_VALID.csv', dtype=str)
print(f"\nFinal N: {len(final):,} neurons")

# Load metadata
meta = feather.read_feather('banc_888_meta.feather')
meta['BANC'] = meta['root_626'].astype(str).str.strip()

# Merge metadata into final
final_meta = final.merge(meta[['BANC','cell_type','super_class','cell_class',
                                'sexually_dimorphic','region','neurotransmitter_predicted',
                                'hemilineage']], on='BANC', how='left')

print("\n=== SUPER CLASS (broad category) ===")
print(final_meta['super_class'].value_counts().to_string())

print("\n=== TOP 30 CELL TYPES ===")
print(final_meta['cell_type'].value_counts().head(30).to_string())

print("\n=== BRAIN REGIONS ===")
print(final_meta['region'].value_counts().head(15).to_string())

print("\n=== NEUROTRANSMITTER ===")
print(final_meta['neurotransmitter_predicted'].value_counts().to_string())

# Compare: what's in final 4,566 vs what was pruned
# Load the original 10,808 set
meta_full = meta[meta['sexually_dimorphic'] == 'isomorphic'].copy()
meta_full = meta_full[meta_full['fafb_match'].notna() & meta_full['malecns_match'].notna()].copy()
meta_full['FAFB'] = meta_full['fafb_match'].astype(str)
meta_full['MCNS'] = meta_full['malecns_match'].astype(str)

# Load challenge IDs to filter
import pandas as pd
fafb_ids = set(pd.read_csv('fafb_783_edge_list.csv', dtype=str, header=None)[0]) | \
           set(pd.read_csv('fafb_783_edge_list.csv', dtype=str, header=None)[1])
banc_ids = set(pd.read_csv('banc_626_edge_list.csv', dtype=str, header=None)[0]) | \
           set(pd.read_csv('banc_626_edge_list.csv', dtype=str, header=None)[1])
mcns_ids = set(pd.read_csv('mcns_0.9_edge_list.csv', dtype=str, header=None)[0]) | \
           set(pd.read_csv('mcns_0.9_edge_list.csv', dtype=str, header=None)[1])

meta_full = meta_full[
    meta_full['BANC'].isin(banc_ids) &
    meta_full['FAFB'].isin(fafb_ids) &
    meta_full['MCNS'].isin(mcns_ids)
].copy()

# What was pruned?
pruned_banc = set(meta_full['BANC']) - set(final['BANC'])
print(f"\n=== PRUNING ANALYSIS ===")
print(f"Started with: {len(meta_full):,} sex-invariant triplets in challenge files")
print(f"Final kept:   {len(final):,}")
print(f"Pruned:       {len(pruned_banc):,} neurons")

pruned_meta = meta_full[meta_full['BANC'].isin(pruned_banc)]
print(f"\nWhat was PRUNED (super_class distribution):")
print(pruned_meta['super_class'].value_counts().to_string())

print(f"\nWhat was KEPT (super_class distribution):")
print(final_meta['super_class'].value_counts().to_string())

# What fraction of each cell type survived?
print("\n=== SURVIVAL RATE BY SUPER CLASS ===")
for sc in meta_full['super_class'].value_counts().index[:8]:
    total = (meta_full['super_class'] == sc).sum()
    kept = (final_meta['super_class'] == sc).sum()
    print(f"  {sc:<35} kept {kept:>5,} / {total:>5,} = {100*kept/total:.0f}%")

# Key question: are there specific cell types with 100% survival?
print("\n=== CELL TYPES WITH 100% SURVIVAL (fully conserved circuits!) ===")
cell_type_total = meta_full.groupby('cell_type').size()
cell_type_kept = final_meta.groupby('cell_type').size()
survival = (cell_type_kept / cell_type_total * 100).dropna()
perfect = survival[survival == 100].sort_values(ascending=False)
print(f"Cell types with 100% survival: {len(perfect)}")
print(perfect.head(30).to_string())
