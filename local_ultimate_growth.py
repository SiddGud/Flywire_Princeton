import pandas as pd
import numpy as np
import pyarrow.feather as feather
import time
import os

print("=" * 65)
print("  ULTIMATE SMART GROWTH (Hungarian Injection)")
print("=" * 65)

# 1. Load the 17k core
CORE_FILE = 'submission_S2_FINAL_17174_CLEAN.csv'
print(f"\n1. Loading existing core: {CORE_FILE}")
core_df = pd.read_csv(CORE_FILE, dtype=str)
core_banc = set(core_df['BANC'])
core_fafb = set(core_df['FAFB'])
core_mcns = set(core_df['MCNS'])
print(f"   Core size: {len(core_df):,}")

# 2. Load biological metadata
print("\n2. Loading BANC metadata...")
t0 = time.time()
meta = feather.read_feather('banc_888_meta.feather')
print(f"   Loaded in {time.time()-t0:.1f}s")

def score_row(row):
    s = 0
    if row.get('proofread') == True or str(row.get('proofread')).lower() == 'true': s += 2
    if str(row.get('roughly_proofread')).lower() == 'true': s += 1
    if str(row.get('sexually_dimorphic')).lower() == 'isomorphic': s += 3
    return s

print("   Computing biological quality scores...")
meta['_base_score'] = meta.apply(score_row, axis=1)

# Extract UNUSED nodes
unused_meta = meta[~meta['root_626'].astype(str).isin(core_banc)].copy()
print(f"   Unused BANC nodes available: {len(unused_meta):,}")

# 3. Build FAFB candidates
f_manual = unused_meta[['root_626','fafb_match','_base_score']].dropna(subset=['fafb_match']).copy()
f_manual.rename(columns={'fafb_match': 'FAFB'}, inplace=True)
f_manual['match_score'] = f_manual['_base_score'] + 10

f_nblast = unused_meta[['root_626','fafb_nblast_match','_base_score']].dropna(subset=['fafb_nblast_match']).copy()
f_nblast.rename(columns={'fafb_nblast_match': 'FAFB'}, inplace=True)
f_nblast['match_score'] = f_nblast['_base_score'] + 7

f_all = pd.concat([f_manual, f_nblast]).copy()
f_all['BANC'] = f_all['root_626'].astype(str)
f_all['FAFB'] = f_all['FAFB'].astype(str)
# Filter out FAFB nodes already in core
f_all = f_all[~f_all['FAFB'].isin(core_fafb)]
f_all = f_all.groupby(['BANC','FAFB'])['match_score'].max().reset_index()

# 4. Build MCNS candidates
m_manual = unused_meta[['root_626','malecns_match','_base_score']].dropna(subset=['malecns_match']).copy()
m_manual.rename(columns={'malecns_match': 'MCNS'}, inplace=True)
m_manual['match_score'] = m_manual['_base_score'] + 10

m_nblast = unused_meta[['root_626','malecns_nblast_match','_base_score']].dropna(subset=['malecns_nblast_match']).copy()
m_nblast.rename(columns={'malecns_nblast_match': 'MCNS'}, inplace=True)
m_nblast['match_score'] = m_nblast['_base_score'] + 7

m_all = pd.concat([m_manual, m_nblast]).copy()
m_all['BANC'] = m_all['root_626'].astype(str)
m_all['MCNS'] = m_all['MCNS'].astype(str)
# Filter out MCNS nodes already in core
m_all = m_all[~m_all['MCNS'].isin(core_mcns)]
m_all = m_all.groupby(['BANC','MCNS'])['match_score'].max().reset_index()

print(f"\n3. New Candidates Extracted:")
print(f"   BANC-FAFB candidates: {len(f_all):,}")
print(f"   BANC-MCNS candidates: {len(m_all):,}")

# 5. Greedy / Hungarian Proxy Assignment
# To maximize weight efficiently, we sort by score and greedily assign
print("\n4. Resolving bijections by optimal biological score...")

f_sorted = f_all.sort_values('match_score', ascending=False)
assigned_banc = set()
assigned_fafb = set()
f_resolved = []

for _, row in f_sorted.iterrows():
    if row['BANC'] in assigned_banc or row['FAFB'] in assigned_fafb:
        continue
    assigned_banc.add(row['BANC'])
    assigned_fafb.add(row['FAFB'])
    f_resolved.append(row)
f_resolved = pd.DataFrame(f_resolved)

surviving_banc = set(f_resolved['BANC'])
m_filtered = m_all[m_all['BANC'].isin(surviving_banc)].copy()
m_sorted = m_filtered.sort_values('match_score', ascending=False)
assigned_banc_m = set()
assigned_mcns = set()
m_resolved = []

for _, row in m_sorted.iterrows():
    if row['BANC'] in assigned_banc_m or row['MCNS'] in assigned_mcns:
        continue
    assigned_banc_m.add(row['BANC'])
    assigned_mcns.add(row['MCNS'])
    m_resolved.append(row)
m_resolved = pd.DataFrame(m_resolved)

print(f"   Resolved FAFB pairs: {len(f_resolved):,}")
print(f"   Resolved MCNS pairs: {len(m_resolved):,}")

# 6. Merge into triplets
f_scores = f_resolved[['BANC','FAFB','match_score']].rename(columns={'match_score':'f_score'})
m_scores = m_resolved[['BANC','MCNS','match_score']].rename(columns={'match_score':'m_score'})
new_triplets = f_scores.merge(m_scores, on='BANC')
new_triplets['total_score'] = new_triplets['f_score'] + new_triplets['m_score']
new_triplets = new_triplets.sort_values('total_score', ascending=False)[['BANC','FAFB','MCNS']]

print(f"\n5. Extracted {len(new_triplets):,} high-quality biological triplets!")

# 7. Inject into Core
print("\n6. Injecting new triplets into 17k core...")
expanded_pool = pd.concat([core_df[['BANC','FAFB','MCNS']], new_triplets], ignore_index=True)
print(f"   New massive candidate pool: {len(expanded_pool):,} nodes")

# Save to expanded_triplets.csv for SA Pruner
out_file = 'expanded_triplets.csv'
expanded_pool.to_csv(out_file, index=False)
print(f"   Saved pool to '{out_file}'")

print("\n7. Next Step:")
print(f"   Run 'python ultimate_sa_pruner.py' to ruthlessly prune this expanded {len(expanded_pool):,} node pool!")
print("=" * 65)
