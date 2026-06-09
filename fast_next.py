"""
fast_next.py
============
Parallelized version of next_strategies.
Skips isomorphic-only filter (it gave worse results: 4,658 vs 4,784).
Runs:
  1. Cell-type weighted bijection → SA sweep (271 configs, 12 cores) → directed grow
  2. Iterative grow on best known result (submission_FIXED_4831.csv)
"""

import pandas as pd
import numpy as np
import pyarrow.feather as feather
import time
from collections import defaultdict
import multiprocessing

CURRENT_BEST = 4831
CURRENT_BEST_FILE = 'submission_FIXED_4831.csv'

print("Loading edge lists...")
t0 = time.time()
fafb_raw = pd.read_csv('fafb_783_edge_list.csv'); fafb_raw.columns = ['src','tgt']
banc_raw = pd.read_csv('banc_626_edge_list.csv'); banc_raw.columns = ['src','tgt']
mcns_raw = pd.read_csv('mcns_0.9_edge_list.csv'); mcns_raw.columns = ['src','tgt']
fafb_raw = fafb_raw.astype(str); banc_raw = banc_raw.astype(str); mcns_raw = mcns_raw.astype(str)
print(f"Loaded in {time.time()-t0:.1f}s")

banc_str_edges = set(zip(banc_raw['src'], banc_raw['tgt']))
fafb_str_edges = set(zip(fafb_raw['src'], fafb_raw['tgt']))
mcns_str_edges = set(zip(mcns_raw['src'], mcns_raw['tgt']))


def build_index_edges(df, to_idx):
    mask = df['src'].isin(to_idx) & df['tgt'].isin(to_idx)
    return [(to_idx[s], to_idx[t]) for s, t in zip(df[mask]['src'], df[mask]['tgt'])]


def sa_worker(params):
    alpha, seed, edges_data = params
    np.random.seed(seed)
    banc_raw_e, fafb_raw_e, mcns_raw_e, num_nodes = edges_data
    be = set(map(tuple, banc_raw_e))
    fe = set(map(tuple, fafb_raw_e))
    me = set(map(tuple, mcns_raw_e))
    all_e = be | fe | me
    conflicts = defaultdict(int); adj = defaultdict(list); total = 0
    for (i, j) in all_e:
        adj[i].append((i,j)); adj[j].append((i,j))
        ib, iff, im = (i,j) in be, (i,j) in fe, (i,j) in me
        if not (ib == iff == im): total += 1; conflicts[i] += 1; conflicts[j] += 1
    active = set(range(num_nodes))
    while total > 0:
        if alpha == float('inf') or not conflicts:
            worst = max(conflicts, key=conflicts.get)
        else:
            top = sorted(conflicts.items(), key=lambda x: x[1], reverse=True)[:50]
            ns = [x[0] for x in top]; cs = np.array([x[1] for x in top], dtype=np.float64)
            w = cs**alpha; worst = int(np.random.choice(ns, p=w/w.sum()))
        active.discard(worst)
        for e in adj[worst]:
            if e in all_e:
                ib, iff, im = e in be, e in fe, e in me
                if not (ib == iff == im): total -= 1; conflicts[e[0]] -= 1; conflicts[e[1]] -= 1
                all_e.discard(e); be.discard(e); fe.discard(e); me.discard(e)
        if worst in conflicts: del conflicts[worst]
    return (alpha, seed, len(active), sorted(list(active)))


def run_sa_sweep(filtered, label=""):
    b2i = {b:i for i,b in enumerate(filtered['BANC'].tolist())}
    f2i = {f:i for i,f in enumerate(filtered['FAFB'].tolist())}
    m2i = {m:i for i,m in enumerate(filtered['MCNS'].tolist())}
    be = build_index_edges(banc_raw, b2i)
    fe = build_index_edges(fafb_raw, f2i)
    me = build_index_edges(mcns_raw, m2i)
    edges_data = (be, fe, me, len(filtered))

    alphas = [1.2, 1.5, 2, 3, 5, 8, 10, 15, 20, float('inf')]
    seeds = list(range(30))
    tasks = [(a, s, edges_data) for a in alphas for s in seeds
             if not (a == float('inf') and s > 0)]

    ncores = multiprocessing.cpu_count()
    print(f"  {label}: {len(tasks)} SA runs across {ncores} cores...")
    best_n = 0; best_active = None; best_params = None
    t0 = time.time()
    with multiprocessing.Pool(processes=ncores) as pool:
        for res in pool.imap_unordered(sa_worker, tasks):
            a, s, n, active = res
            a_str = "Greedy" if a == float('inf') else f"α={a}"
            if n > best_n:
                best_n = n; best_active = active; best_params = (a_str, s)
                print(f"    *** NEW BEST: {n} ({a_str}, seed={s}) ***")
    print(f"  SA done in {time.time()-t0:.0f}s | Best: {best_n} ({best_params})")
    return filtered.iloc[best_active].copy().reset_index(drop=True), best_n


def directed_grow(sa_df, all_df):
    in_core = set(sa_df['BANC'].astype(str))
    candidates = all_df[~all_df['BANC'].astype(str).isin(in_core)].copy()
    core = {str(b): (str(f), str(m))
            for b, f, m in zip(sa_df['BANC'], sa_df['FAFB'], sa_df['MCNS'])}
    added = 0
    t0 = time.time()
    for _, row in candidates.iterrows():
        b, f, m = str(row['BANC']), str(row['FAFB']), str(row['MCNS'])
        ok = True
        for cb, (cf, cm) in core.items():
            if ((b,cb) in banc_str_edges) != ((f,cf) in fafb_str_edges) or \
               ((b,cb) in banc_str_edges) != ((m,cm) in mcns_str_edges): ok=False; break
            if ((cb,b) in banc_str_edges) != ((cf,f) in fafb_str_edges) or \
               ((cb,b) in banc_str_edges) != ((cm,m) in mcns_str_edges): ok=False; break
        if ok:
            core[b] = (f, m); added += 1
            if added % 10 == 0:
                print(f"    Grow: +{added}, core = {len(sa_df)+added:,} ({time.time()-t0:.0f}s)")
    rows = [{'BANC': b, 'FAFB': f, 'MCNS': m} for b, (f, m) in core.items()]
    return pd.DataFrame(rows), added


# ════════════════════════════════════════════════════════════════
# STRATEGY 1: Cell-type weighted bijection
# ════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  STRATEGY 1: Cell-type weighted bijection")
print("="*60)

meta = feather.read_feather('banc_888_meta.feather')

def base_score(row):
    s = 0
    if str(row.get('sexually_dimorphic','')).lower() == 'isomorphic': s += 3
    if str(row.get('roughly_proofread','')).lower() == 'true': s += 1
    return s

meta['_base'] = meta.apply(base_score, axis=1)
meta['_ct_f'] = ((meta['cell_type'].notna()) & (meta['fafb_cell_type'].notna()) &
                 (meta['cell_type'] == meta['fafb_cell_type'])).astype(int) * 5
meta['_ct_m'] = ((meta['cell_type'].notna()) & (meta['malecns_cell_type'].notna()) &
                 (meta['cell_type'] == meta['malecns_cell_type'])).astype(int) * 5

# Build all FAFB candidates
f_all = []
for col, bonus in [('fafb_match', 10), ('fafb_nblast_match', 7)]:
    tmp = meta[['root_626', col, '_base', '_ct_f']].dropna(subset=[col]).copy()
    tmp.rename(columns={col: 'FAFB'}, inplace=True)
    tmp['f_score'] = tmp['_base'] + tmp['_ct_f'] + bonus
    f_all.append(tmp[['root_626', 'FAFB', 'f_score']])
f_all = pd.concat(f_all).groupby(['root_626', 'FAFB'])['f_score'].max().reset_index()
f_all['BANC'] = f_all['root_626'].astype(str); f_all['FAFB'] = f_all['FAFB'].astype(str)

# Build all MCNS candidates
m_all = []
for col, bonus in [('malecns_match', 10), ('malecns_nblast_match', 7)]:
    tmp = meta[['root_626', col, '_base', '_ct_m']].dropna(subset=[col]).copy()
    tmp.rename(columns={col: 'MCNS'}, inplace=True)
    tmp['m_score'] = tmp['_base'] + tmp['_ct_m'] + bonus
    m_all.append(tmp[['root_626', 'MCNS', 'm_score']])
m_all = pd.concat(m_all).groupby(['root_626', 'MCNS'])['m_score'].max().reset_index()
m_all['BANC'] = m_all['root_626'].astype(str); m_all['MCNS'] = m_all['MCNS'].astype(str)

# Greedy bijection: sort by score, assign first-come
def greedy_bijection(left, right, key):
    merged = left.merge(right, on='BANC')
    merged['total'] = merged['f_score'] + merged['m_score']
    merged = merged.sort_values('total', ascending=False)
    sb, sf, sm, rows = set(), set(), set(), []
    for _, row in merged.iterrows():
        b, f, m = str(row['BANC']), str(row['FAFB']), str(row['MCNS'])
        if b in sb or f in sf or m in sm: continue
        sb.add(b); sf.add(f); sm.add(m)
        rows.append({'BANC': b, 'FAFB': f, 'MCNS': m})
    return pd.DataFrame(rows)

ct_triplets = greedy_bijection(f_all, m_all, 'total')
print(f"CT-weighted bijection pool: {len(ct_triplets):,} triplets")
ct_triplets.to_csv('ct_weighted_triplets.csv', index=False)

ct_sa_df, ct_sa_n = run_sa_sweep(ct_triplets, label="CT-weighted bijection")
ct_final, ct_added = directed_grow(ct_sa_df, ct_triplets)
ct_n = len(ct_final)
print(f"\nCT strategy: SA={ct_sa_n} + Grow={ct_added} = {ct_n}")

if ct_n > CURRENT_BEST:
    ct_final.to_csv(f'submission_CT_{ct_n}.csv', index=False)
    print(f"*** NEW OVERALL BEST: {ct_n} ***")
    CURRENT_BEST = ct_n
else:
    print(f"(No improvement over {CURRENT_BEST})")

print(f"\n{'='*60}\nFINAL BEST: {CURRENT_BEST}\n{'='*60}")
