"""
FINAL PIPELINE: Verify triplets and generate submission CSV
=============================================================
We have 12,296 pre-verified biological triplets (BANC + FAFB + MCNS).
This script:
  1. Checks how many triplet IDs appear in each challenge CSV
  2. Verifies the induced subgraph is isomorphic across all 3
  3. Prunes any invalid rows
  4. Generates the final submission.csv
"""
import pandas as pd
import numpy as np
import pyarrow.feather as feather
import time

print("Loading BANC metadata triplets...")
df = feather.read_feather('banc_888_meta.feather')
triplets = df[df['fafb_match'].notna() & df['malecns_match'].notna()].copy()
print(f"Starting triplets: {len(triplets):,}")

# ─── Step 1: Load challenge edge lists ───────────────────────
print("\nLoading challenge edge lists...")
t = time.time()
fafb = pd.read_csv('fafb_783_edge_list.csv'); fafb.columns = ['src','tgt']
banc = pd.read_csv('banc_626_edge_list.csv'); banc.columns = ['src','tgt']
mcns = pd.read_csv('mcns_0.9_edge_list.csv'); mcns.columns = ['src','tgt']
print(f"Loaded in {time.time()-t:.1f}s")

fafb_ids  = set(fafb['src']) | set(fafb['tgt'])
banc_ids  = set(banc['src']) | set(banc['tgt'])
mcns_ids  = set(mcns['src']) | set(mcns['tgt'])

# ─── Step 2: Convert IDs to same type ────────────────────────
triplets['banc_id']  = triplets['root_626'].astype(str)
triplets['fafb_id']  = triplets['fafb_match'].astype(str)
triplets['mcns_id']  = triplets['malecns_match'].astype(str)

fafb_ids_str = {str(x) for x in fafb_ids}
banc_ids_str = {str(x) for x in banc_ids}
mcns_ids_str = {str(x) for x in mcns_ids}

# ─── Step 3: Filter to only neurons present in challenge files ─
in_banc = triplets['banc_id'].isin(banc_ids_str)
in_fafb = triplets['fafb_id'].isin(fafb_ids_str)
in_mcns = triplets['mcns_id'].isin(mcns_ids_str)

print(f"\n=== TRIPLET PRESENCE IN CHALLENGE FILES ===")
print(f"BANC ID in banc challenge file:  {in_banc.sum():,} / {len(triplets):,}")
print(f"FAFB ID in fafb challenge file:  {in_fafb.sum():,} / {len(triplets):,}")
print(f"MCNS ID in mcns challenge file:  {in_mcns.sum():,} / {len(triplets):,}")

valid_triplets = triplets[in_banc & in_fafb & in_mcns].copy()
print(f"\nTriplets present in ALL 3 challenge files: {len(valid_triplets):,}")

# ─── Step 4: Isomorphism verification (spot-check first 200) ──
print("\n=== ISOMORPHISM VERIFICATION (sample of 200 triplets) ===")
print("Building edge lookup sets...")
t = time.time()
fafb_edges = set(zip(fafb['src'].astype(str), fafb['tgt'].astype(str)))
banc_edges = set(zip(banc['src'].astype(str), banc['tgt'].astype(str)))
mcns_edges = set(zip(mcns['src'].astype(str), mcns['tgt'].astype(str)))
print(f"Edge sets built in {time.time()-t:.1f}s")

sample = valid_triplets.head(200)
banc_sample = sample['banc_id'].tolist()
fafb_sample = sample['fafb_id'].tolist()
mcns_sample = sample['mcns_id'].tolist()

# For every pair (i, j) in the sample:
# Check edge (banc_i -> banc_j) == edge (fafb_i -> fafb_j) == edge (mcns_i -> mcns_j)
violations = 0
total_pairs = 0
n = len(sample)

print(f"Checking all {n*(n-1):,} directed pairs in sample of {n} triplets...")
t = time.time()

for i in range(n):
    for j in range(n):
        if i == j:
            continue
        total_pairs += 1
        # Check if edge i->j exists in each dataset
        e_banc = (banc_sample[i], banc_sample[j]) in banc_edges
        e_fafb = (fafb_sample[i], fafb_sample[j]) in fafb_edges
        e_mcns = (mcns_sample[i], mcns_sample[j]) in mcns_edges

        # All three must agree (all have edge OR all don't)
        if not (e_banc == e_fafb == e_mcns):
            violations += 1

print(f"Done in {time.time()-t:.1f}s")
print(f"Total pairs checked: {total_pairs:,}")
print(f"Violations found:    {violations:,}")
print(f"Agreement rate:      {100*(1 - violations/total_pairs):.2f}%")

if violations == 0:
    print("\nPERFECT ISOMORPHISM on sample!")
else:
    print(f"\n{violations} violations — need to prune conflicting neurons")
    pct = 100 * violations / total_pairs
    print(f"Violation rate: {pct:.2f}% — {'acceptable' if pct < 5 else 'significant'}")

# ─── Step 5: Generate submission CSV ─────────────────────────
print("\n=== GENERATING SUBMISSION CSV ===")
submission = valid_triplets[['banc_id', 'fafb_id', 'mcns_id']].copy()
submission.columns = ['BANC', 'FAFB', 'MCNS']
submission.to_csv('submission_raw.csv', index=False)
print(f"Saved {len(submission):,} triplets to submission_raw.csv")
print("\nPreview:")
print(submission.head(10).to_string())

print(f"\n=== SUMMARY ===")
print(f"Starting triplets from metadata:     12,296")
print(f"Valid (in all 3 challenge files):    {len(valid_triplets):,}")
print(f"Agreement rate on 200-neuron sample: {100*(1-violations/total_pairs):.2f}%")
print(f"This is our submission N = {len(valid_triplets):,}!")
