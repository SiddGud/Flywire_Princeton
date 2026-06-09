import pandas as pd
import pyarrow.feather as feather

meta = feather.read_feather('banc_888_meta.feather')

f_all = pd.concat([
    meta[['root_626','fafb_match']].dropna().rename(columns={'fafb_match':'FAFB'}),
    meta[['root_626','fafb_nblast_match']].dropna().rename(columns={'fafb_nblast_match':'FAFB'})
]).drop_duplicates()

m_all = pd.concat([
    meta[['root_626','malecns_match']].dropna().rename(columns={'malecns_match':'MCNS'}),
    meta[['root_626','malecns_nblast_match']].dropna().rename(columns={'malecns_nblast_match':'MCNS'})
]).drop_duplicates()

merged = f_all.merge(m_all, on='root_626')

fafb_counts = f_all.groupby('FAFB')['root_626'].count()
single_fafb = fafb_counts[fafb_counts == 1].index
multi_fafb = fafb_counts[fafb_counts > 1].index

single_triplets = merged[merged['FAFB'].isin(single_fafb)]
multi_triplets = merged[merged['FAFB'].isin(multi_fafb)]
print(f'Total raw triplets: {len(merged):,}')
print(f'From UNIQUE FAFB mappings: {len(single_triplets):,}')
print(f'From AMBIGUOUS FAFB mappings: {len(multi_triplets):,}')

print(f'Unique BANC in ambiguous set: {multi_triplets["root_626"].nunique():,}')
print(f'Unique FAFB in ambiguous set: {multi_triplets["FAFB"].nunique():,}')
print(f'Unique MCNS in ambiguous set: {multi_triplets["MCNS"].nunique():,}')

# Also check MCNS ambiguity
mcns_counts = m_all.groupby('MCNS')['root_626'].count()
single_mcns = mcns_counts[mcns_counts == 1].index
both_single = merged[merged['FAFB'].isin(single_fafb) & merged['MCNS'].isin(single_mcns)]
print()
print(f'Triplets where BOTH FAFB and MCNS are unique (gold standard): {len(both_single):,}')
print(f'Unique BANC in gold: {both_single["root_626"].nunique():,}')
