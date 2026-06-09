"""
next_strategies.py
==================
Runs three strategies back-to-back to try to push past N=4,831:

1. Isomorphic-only pre-filter: SA on only sexually_dimorphic=='isomorphic' neurons
2. Cell-type weighted bijection: +5 bonus for cell_type match in bijection resolver
3. Iterative SA->Grow->SA->Grow loop on best result so far

All use the corrected DIRECTED grow phase.
"""

import pandas as pd
import numpy as np
import pyarrow.feather as feather
import time
from collections import defaultdict
import multiprocessing

# ── Shared edge loading ─────────────────────────────────────────
print("Loading edge lists...")
t0 = time.time()
fafb_raw = pd.read_csv('fafb_783_edge_list.csv'); fafb_raw.columns = ['src','tgt']
banc_raw = pd.read_csv('banc_626_edge_list.csv'); banc_raw.columns = ['src','tgt']
mcns_raw = pd.read_csv('mcns_0.9_edge_list.csv'); mcns_raw.columns = ['src','tgt']
fafb_raw = fafb_raw.astype(str); banc_raw = banc_raw.astype(str); mcns_raw = mcns_raw.astype(str)
print(f"Loaded in {time.time()-t0:.1f}s")

# Directed string edge sets for grow
banc_str_edges = set(zip(banc_raw['src'], banc_raw['tgt']))
fafb_str_edges = set(zip(fafb_raw['src'], fafb_raw['tgt']))
mcns_str_edges = set(zip(mcns_raw['src'], mcns_raw['tgt']))

def build_index_edges(df, to_idx):
    df_s = df.astype(str)
    mask = df_s['src'].isin(to_idx) & df_s['tgt'].isin(to_idx)
    return [(to_idx[s], to_idx[t]) for s,t in zip(df_s[mask]['src'], df_s[mask]['tgt'])]

def sa_pruner(filtered, alpha=1.5, seed=24):
    np.random.seed(seed)
    b2i = {b:i for i,b in enumerate(filtered['BANC'].tolist())}
    f2i = {f:i for i,f in enumerate(filtered['FAFB'].tolist())}
    m2i = {m:i for i,m in enumerate(filtered['MCNS'].tolist())}
    be = set(map(tuple, build_index_edges(banc_raw, b2i)))
    fe = set(map(tuple, build_index_edges(fafb_raw, f2i)))
    me = set(map(tuple, build_index_edges(mcns_raw, m2i)))
    all_e = be | fe | me
    conflicts = defaultdict(int); adj = defaultdict(list); total = 0
    for (i,j) in all_e:
        adj[i].append((i,j)); adj[j].append((i,j))
        ib,iff,im = (i,j) in be,(i,j) in fe,(i,j) in me
        if not (ib==iff==im): total+=1; conflicts[i]+=1; conflicts[j]+=1
    active = set(range(len(filtered)))
    while total > 0:
        top_k = min(50, len(conflicts))
        items = sorted(conflicts.items(), key=lambda x:x[1], reverse=True)[:top_k]
        nodes=[x[0] for x in items]; counts=np.array([x[1] for x in items],dtype=np.float64)
        weights=counts**alpha; probs=weights/weights.sum()
        worst=int(np.random.choice(nodes,p=probs))
        active.discard(worst)
        for e in adj[worst]:
            if e in all_e:
                ib,iff,im = e in be,e in fe,e in me
                if not (ib==iff==im): total-=1; conflicts[e[0]]-=1; conflicts[e[1]]-=1
                all_e.discard(e); be.discard(e); fe.discard(e); me.discard(e)
        if worst in conflicts: del conflicts[worst]
    return filtered.iloc[sorted(active)].copy().reset_index(drop=True)

def directed_grow(sa_df, all_df):
    in_core = set(sa_df['BANC'].astype(str))
    candidates = all_df[~all_df['BANC'].astype(str).isin(in_core)].copy()
    core = {str(b):(str(f),str(m)) for b,f,m in zip(sa_df['BANC'],sa_df['FAFB'],sa_df['MCNS'])}
    added = 0
    for _,row in candidates.iterrows():
        b,f,m = str(row['BANC']),str(row['FAFB']),str(row['MCNS'])
        ok = True
        for cb,(cf,cm) in core.items():
            if ((b,cb) in banc_str_edges) != ((f,cf) in fafb_str_edges) or \
               ((b,cb) in banc_str_edges) != ((m,cm) in mcns_str_edges): ok=False; break
            if ((cb,b) in banc_str_edges) != ((cf,f) in fafb_str_edges) or \
               ((cb,b) in banc_str_edges) != ((cm,m) in mcns_str_edges): ok=False; break
        if ok: core[b]=(f,m); added+=1
    rows=[{'BANC':b,'FAFB':f,'MCNS':m} for b,(f,m) in core.items()]
    return pd.DataFrame(rows), added

GLOBAL_BEST = 4831
GLOBAL_BEST_DF = pd.read_csv('submission_FIXED_4831.csv', dtype=str)

# ════════════════════════════════════════════════════════════════
# STRATEGY 1: Isomorphic-only pre-filter
# ════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  STRATEGY 1: Isomorphic-only pre-filter SA")
print("="*60)

meta = feather.read_feather('banc_888_meta.feather')
all_triplets = pd.read_csv('hungarian_triplets.csv', dtype=str)

iso_banc = set(meta[meta['sexually_dimorphic']=='isomorphic']['root_626'].astype(str))
iso_triplets = all_triplets[all_triplets['BANC'].isin(iso_banc)].copy().reset_index(drop=True)
print(f"Isomorphic-only pool: {len(iso_triplets):,} (from {len(all_triplets):,})")

# Run SA grid on isomorphic pool
def sa_worker_iso(params):
    a, s, filtered = params
    return sa_pruner(filtered.copy(), alpha=a, seed=s)

best_iso_sa = 0; best_iso_df = None
alphas = [1.2, 1.5, 2, 3, 5, 8, float('inf')]
tasks = [(a,s,iso_triplets) for a in alphas for s in range(10) if not (a==float('inf') and s>0)]

print(f"Running {len(tasks)} SA configs on isomorphic pool...")
for a,s,filt in tasks:
    result = sa_pruner(filt, alpha=float('inf') if a==float('inf') else a, seed=s)
    n = len(result)
    if n > best_iso_sa: best_iso_sa = n; best_iso_df = result; print(f"  New best: α={a} seed={s} N={n}")

final_iso, added_iso = directed_grow(best_iso_df, iso_triplets)
print(f"Isomorphic strategy: SA={best_iso_sa} + Grow={added_iso} = {len(final_iso)}")

if len(final_iso) > GLOBAL_BEST:
    GLOBAL_BEST = len(final_iso); GLOBAL_BEST_DF = final_iso
    final_iso.to_csv(f'submission_ISO_{len(final_iso)}.csv', index=False)
    print(f"*** NEW OVERALL BEST: {GLOBAL_BEST} ***")

# ════════════════════════════════════════════════════════════════
# STRATEGY 2: Cell-type weighted bijection
# ════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  STRATEGY 2: Cell-type weighted bijection")
print("="*60)

# Build quality scores WITH cell type bonus
def score_row(row):
    s = 0
    if str(row.get('proofread','')).lower()=='true': s+=2
    if str(row.get('roughly_proofread','')).lower()=='true': s+=1
    if str(row.get('sexually_dimorphic','')).lower()=='isomorphic': s+=3
    return s

meta['_base_score'] = meta.apply(score_row, axis=1)
meta['_ct_bonus_f'] = (meta['cell_type'].notna() & meta['fafb_cell_type'].notna() &
                       (meta['cell_type'] == meta['fafb_cell_type'])).astype(int) * 5
meta['_ct_bonus_m'] = (meta['cell_type'].notna() & meta['malecns_cell_type'].notna() &
                       (meta['cell_type'] == meta['malecns_cell_type'])).astype(int) * 5

f_manual = meta[['root_626','fafb_match','_base_score','_ct_bonus_f']].dropna(subset=['fafb_match']).copy()
f_manual.rename(columns={'fafb_match':'FAFB'}, inplace=True)
f_manual['match_score'] = f_manual['_base_score'] + f_manual['_ct_bonus_f'] + 10

f_nblast = meta[['root_626','fafb_nblast_match','_base_score','_ct_bonus_f']].dropna(subset=['fafb_nblast_match']).copy()
f_nblast.rename(columns={'fafb_nblast_match':'FAFB'}, inplace=True)
f_nblast['match_score'] = f_nblast['_base_score'] + f_nblast['_ct_bonus_f'] + 7

m_manual = meta[['root_626','malecns_match','_base_score','_ct_bonus_m']].dropna(subset=['malecns_match']).copy()
m_manual.rename(columns={'malecns_match':'MCNS'}, inplace=True)
m_manual['match_score'] = m_manual['_base_score'] + m_manual['_ct_bonus_m'] + 10

m_nblast = meta[['root_626','malecns_nblast_match','_base_score','_ct_bonus_m']].dropna(subset=['malecns_nblast_match']).copy()
m_nblast.rename(columns={'malecns_nblast_match':'MCNS'}, inplace=True)
m_nblast['match_score'] = m_nblast['_base_score'] + m_nblast['_ct_bonus_m'] + 7

f_all = pd.concat([f_manual[['root_626','FAFB','match_score']], f_nblast[['root_626','FAFB','match_score']]]).copy()
f_all['BANC'] = f_all['root_626'].astype(str); f_all['FAFB'] = f_all['FAFB'].astype(str)
f_all = f_all.groupby(['BANC','FAFB'])['match_score'].max().reset_index()

m_all = pd.concat([m_manual[['root_626','MCNS','match_score']], m_nblast[['root_626','MCNS','match_score']]]).copy()
m_all['BANC'] = m_all['root_626'].astype(str); m_all['MCNS'] = m_all['MCNS'].astype(str)
m_all = m_all.groupby(['BANC','MCNS'])['match_score'].max().reset_index()

# Stage 1: BANC->FAFB
banc_fafb_count = f_all.groupby('BANC')['FAFB'].count()
f_unique = f_all[f_all['BANC'].isin(banc_fafb_count[banc_fafb_count==1].index)].copy()
f_ambig = f_all[f_all['BANC'].isin(banc_fafb_count[banc_fafb_count>1].index)].sort_values('match_score',ascending=False)
assigned_f = set(f_unique['FAFB']); assigned_b = set(f_unique['BANC']); resolved_f=[]
for _,row in f_ambig.iterrows():
    if row['BANC'] in assigned_b or row['FAFB'] in assigned_f: continue
    assigned_b.add(row['BANC']); assigned_f.add(row['FAFB']); resolved_f.append(row)
f_resolved = pd.concat([f_unique, pd.DataFrame(resolved_f)], ignore_index=True)

# Stage 2: BANC->MCNS
surviving = set(f_resolved['BANC'])
m_filt = m_all[m_all['BANC'].isin(surviving)].copy()
banc_mcns_count = m_filt.groupby('BANC')['MCNS'].count()
m_unique = m_filt[m_filt['BANC'].isin(banc_mcns_count[banc_mcns_count==1].index)].copy()
m_ambig = m_filt[m_filt['BANC'].isin(banc_mcns_count[banc_mcns_count>1].index)].sort_values('match_score',ascending=False)
assigned_m = set(m_unique['MCNS']); assigned_bm = set(m_unique['BANC']); resolved_m=[]
for _,row in m_ambig.iterrows():
    if row['BANC'] in assigned_bm or row['MCNS'] in assigned_m: continue
    assigned_bm.add(row['BANC']); assigned_m.add(row['MCNS']); resolved_m.append(row)
m_resolved = pd.concat([m_unique, pd.DataFrame(resolved_m)], ignore_index=True)

# Stage 3: merge + global dedup
f_s = f_resolved[['BANC','FAFB','match_score']].rename(columns={'match_score':'f_score'})
m_s = m_resolved[['BANC','MCNS','match_score']].rename(columns={'match_score':'m_score'})
merged = f_s.merge(m_s, on='BANC')
merged['total_score'] = merged['f_score'] + merged['m_score']
merged = merged.sort_values('total_score', ascending=False)
sb,sf,sm,rows=[],[],[],[]
for _,row in merged.iterrows():
    b,f,m=str(row['BANC']),str(row['FAFB']),str(row['MCNS'])
    if b in sb or f in sf or m in sm: continue
    sb.append(b); sf.append(f); sm.append(m); rows.append({'BANC':b,'FAFB':f,'MCNS':m})
ct_triplets = pd.DataFrame(rows)
print(f"Cell-type weighted bijection: {len(ct_triplets):,} triplets (was 14,551 random, 14,551 hungarian)")

ct_triplets.to_csv('ct_weighted_triplets.csv', index=False)

# Run SA on CT-weighted triplets
best_ct_sa=0; best_ct_df=None
for a,s,_ in tasks:
    result = sa_pruner(ct_triplets, alpha=float('inf') if a==float('inf') else a, seed=s)
    n=len(result)
    if n>best_ct_sa: best_ct_sa=n; best_ct_df=result; print(f"  CT new best: α={a} seed={s} N={n}")

final_ct, added_ct = directed_grow(best_ct_df, ct_triplets)
print(f"Cell-type strategy: SA={best_ct_sa} + Grow={added_ct} = {len(final_ct)}")
if len(final_ct) > GLOBAL_BEST:
    GLOBAL_BEST=len(final_ct); GLOBAL_BEST_DF=final_ct
    final_ct.to_csv(f'submission_CT_{len(final_ct)}.csv', index=False)
    print(f"*** NEW OVERALL BEST: {GLOBAL_BEST} ***")

# ════════════════════════════════════════════════════════════════
# STRATEGY 3: Iterative SA→Grow→SA→Grow loop
# ════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  STRATEGY 3: Iterative SA→Grow loop")
print("="*60)

current_df = pd.read_csv('submission_FIXED_4831.csv', dtype=str)
all_triplets = pd.read_csv('hungarian_triplets.csv', dtype=str)
prev_n = len(current_df)

for cycle in range(1, 6):
    print(f"\n--- Cycle {cycle}: Starting from N={len(current_df)} ---")
    # Grow first (might add zero-violation nodes)
    grown_df, added = directed_grow(current_df, all_triplets)
    print(f"  Grow added: {added}")
    # Now run SA on the grown result — treat it as a starting set
    # SA needs to re-prune if the grown set has any violations (it shouldn't, but SA explores new local optima)
    # Actually since grow is zero-violation, we just run SA again on the full pool
    # using the grown result as a reference to potentially find a different optimum
    best_cycle_n=0; best_cycle_df=None
    for a in [1.2, 1.5, 2, 8, float('inf')]:
        for s in range(5):
            if a==float('inf') and s>0: continue
            result = sa_pruner(all_triplets, alpha=a if a!=float('inf') else float('inf'), seed=s)
            if len(result) > best_cycle_n: best_cycle_n=len(result); best_cycle_df=result
    print(f"  SA best this cycle: {best_cycle_n}")
    final_cycle, added_cycle = directed_grow(best_cycle_df, all_triplets)
    print(f"  After grow: {len(final_cycle)}")
    if len(final_cycle) > GLOBAL_BEST:
        GLOBAL_BEST=len(final_cycle); GLOBAL_BEST_DF=final_cycle
        final_cycle.to_csv(f'submission_ITER_{GLOBAL_BEST}.csv', index=False)
        print(f"  *** NEW OVERALL BEST: {GLOBAL_BEST} ***")
        current_df = final_cycle
    elif len(final_cycle) >= prev_n:
        current_df = final_cycle
    else:
        print("  No improvement, stopping iteration")
        break
    prev_n = len(current_df)

print(f"\n{'='*60}")
print(f"  ALL STRATEGIES COMPLETE")
print(f"  Global best: N = {GLOBAL_BEST}")
print(f"{'='*60}")
GLOBAL_BEST_DF.to_csv(f'submission_STRATEGIES_BEST_{GLOBAL_BEST}.csv', index=False)
