"""
ITERATIVE PERTURBATION MAXIMIZER
=================================
Keeps perturbing + regrowing until convergence.
Each round: remove random 8% → regrow → if better, update base.
Runs until no improvement found across 10 consecutive attempts.
"""
import pandas as pd
import numpy as np
import time, os
from collections import defaultdict

DATA1 = '/kaggle/input/datasets/siddhantgudwani/dataset'

# Auto-find the best submission
CORE_FILE = None
for root, dirs, files in os.walk('/kaggle/input'):
    for f in files:
        if 'MULTI_CLEAN' in f or 'MULTI_11' in f or 'CONNECTED_9788' in f:
            candidate = os.path.join(root, f)
            if CORE_FILE is None:
                CORE_FILE = candidate
            else:
                # Pick whichever has more rows
                try:
                    n1 = len(pd.read_csv(CORE_FILE, dtype=str, nrows=0).columns)
                    CORE_FILE = candidate  # just take latest
                except:
                    pass

if CORE_FILE is None:
    print("ERROR: Could not find a submission file!")
    exit()

print("=" * 65)
print("  ITERATIVE PERTURBATION MAXIMIZER")
print("=" * 65)

print(f"\n1. Loading Edge Lists...")
t0 = time.time()
fafb_df = pd.read_csv(f'{DATA1}/fafb_783_edge_list.csv', header=None, names=['src','tgt'], dtype=str)
banc_df = pd.read_csv(f'{DATA1}/banc_626_edge_list.csv', header=None, names=['src','tgt'], dtype=str)
mcns_df = pd.read_csv(f'{DATA1}/mcns_0.9_edge_list.csv', header=None, names=['src','tgt'], dtype=str)
print(f"Loaded in {time.time()-t0:.1f}s")

print("2. Pre-building edge sets...")
t0 = time.time()
banc_set = set(zip(banc_df['src'], banc_df['tgt']))
fafb_set = set(zip(fafb_df['src'], fafb_df['tgt']))
mcns_set = set(zip(mcns_df['src'], mcns_df['tgt']))
print(f"Done in {time.time()-t0:.1f}s")

def compute_signatures(df, core_map):
    sig_in  = defaultdict(list)
    sig_out = defaultdict(list)
    mask_out = df['src'].isin(core_map) & (~df['tgt'].isin(core_map))
    for s, t in zip(df[mask_out]['src'], df[mask_out]['tgt']):
        sig_in[t].append(core_map[s])
    mask_in = (~df['src'].isin(core_map)) & df['tgt'].isin(core_map)
    for s, t in zip(df[mask_in]['src'], df[mask_in]['tgt']):
        sig_out[s].append(core_map[t])
    all_neighbors = set(sig_in.keys()) | set(sig_out.keys())
    signatures = defaultdict(list)
    for n in all_neighbors:
        key = (tuple(sorted(sig_in[n])), tuple(sorted(sig_out[n])))
        signatures[key].append(n)
    return signatures

def strict_grow_shuffled(candidates_df, core, rng):
    candidates_df = candidates_df.sample(frac=1, random_state=rng).reset_index(drop=True)
    added = 0
    for _, row in candidates_df.iterrows():
        b, f, m = str(row['BANC']), str(row['FAFB']), str(row['MCNS'])
        if b in core: continue
        ok = True
        for cb, (cf, cm) in list(core.items()):
            b_has = (b, cb) in banc_set;  f_has = (f, cf) in fafb_set;  m_has = (m, cm) in mcns_set
            if b_has != f_has or b_has != m_has: ok = False; break
            b_has2 = (cb, b) in banc_set; f_has2 = (cf, f) in fafb_set; m_has2 = (cm, m) in mcns_set
            if b_has2 != f_has2 or b_has2 != m_has2: ok = False; break
        if ok:
            core[b] = (f, m)
            added += 1
    return added

def remove_zero_edge(core):
    b_nodes = set(core.keys())
    connected = set()
    for s, t in zip(banc_df['src'], banc_df['tgt']):
        if s in b_nodes and t in b_nodes:
            connected.add(s); connected.add(t)
    return {b: v for b, v in core.items() if b in connected}

def run_iterative_grow(starting_core, seed_num):
    core = dict(starting_core)
    rng = np.random.RandomState(seed_num)
    round_num = 0
    while True:
        round_num += 1
        b2i = {b: i for i, b in enumerate(core.keys())}
        f2i = {f: i for i, (f, m) in enumerate(core.values())}
        m2i = {m: i for i, (f, m) in enumerate(core.values())}
        b_sigs = compute_signatures(banc_df, b2i)
        f_sigs = compute_signatures(fafb_df, f2i)
        m_sigs = compute_signatures(mcns_df, m2i)
        shared_sigs = set(b_sigs.keys()) & set(f_sigs.keys()) & set(m_sigs.keys())
        candidates = []
        for sig in shared_sigs:
            bn = b_sigs[sig]; fn = f_sigs[sig]; mn = m_sigs[sig]
            rng.shuffle(bn); rng.shuffle(fn); rng.shuffle(mn)
            for i in range(min(len(bn), len(fn), len(mn))):
                if bn[i] not in core:
                    candidates.append({'BANC': bn[i], 'FAFB': fn[i], 'MCNS': mn[i]})
        if not candidates: break
        added = strict_grow_shuffled(pd.DataFrame(candidates), core, rng)
        if added == 0: break
        if round_num > 20: break
    return remove_zero_edge(core)

# ── Load starting core ─────────────────────────────────────────────
print(f"\n3. Loading core from: {CORE_FILE}")
core_df = pd.read_csv(CORE_FILE, dtype=str)
best_core = {str(b): (str(f), str(m)) for b, f, m in zip(core_df['BANC'], core_df['FAFB'], core_df['MCNS'])}
best_core = remove_zero_edge(best_core)
best_n = len(best_core)
print(f"   Starting N = {best_n:,}")

# ── Iterative Perturbation Loop ────────────────────────────────────
print("\n4. Starting Iterative Perturbation...")
MAX_NO_IMPROVE = 10  # stop after 10 consecutive failures
FRACTIONS = [0.05, 0.08, 0.10, 0.15, 0.20]  # try different removal fractions

no_improve = 0
attempt = 0
global_best_n = best_n
global_best_core = dict(best_core)

while no_improve < MAX_NO_IMPROVE:
    attempt += 1
    frac = FRACTIONS[attempt % len(FRACTIONS)]
    rng = np.random.RandomState(attempt * 173 + 31)
    t_start = time.time()
    
    # Perturb
    perturbed = dict(best_core)
    keys = list(perturbed.keys())
    n_remove = int(len(keys) * frac)
    to_remove = rng.choice(keys, size=n_remove, replace=False)
    for k in to_remove:
        del perturbed[k]
    
    # Regrow with random seed
    core = run_iterative_grow(perturbed, seed_num=attempt * 277 + 13)
    n = len(core)
    elapsed = time.time() - t_start
    
    improved = n > global_best_n
    if improved:
        global_best_n = n
        global_best_core = core
        best_core = core
        best_n = n
        no_improve = 0
        print(f"  Attempt {attempt:3d} (frac={frac:.0%}, removed {n_remove:,}): N = {n:,}  ({elapsed:.0f}s) *** NEW BEST ***")
        # Auto-save after every improvement (crash protection)
        tmp_df = pd.DataFrame([{'BANC': b, 'FAFB': f, 'MCNS': m} for b, (f, m) in global_best_core.items()])
        tmp_df.to_csv(f'/kaggle/working/submission_MAX_{n}.csv', index=False)
        print(f"    Auto-saved to /kaggle/working/submission_MAX_{n}.csv")
    else:
        no_improve += 1
        if n > best_n:
            # Not global best but better than current base - update base
            best_core = core
            best_n = n
        print(f"  Attempt {attempt:3d} (frac={frac:.0%}, removed {n_remove:,}): N = {n:,}  ({elapsed:.0f}s)  [no improve {no_improve}/{MAX_NO_IMPROVE}]")

# ── Save ───────────────────────────────────────────────────────────
final_n = len(global_best_core)
print("\n" + "=" * 65)
print(f"  CONVERGED MAXIMUM N: {final_n:,}")
print("=" * 65)

final_df = pd.DataFrame([{'BANC': b, 'FAFB': f, 'MCNS': m} for b, (f, m) in global_best_core.items()])
out_file = f'/kaggle/working/submission_MAX_{final_n}.csv'
final_df.to_csv(out_file, index=False)

from IPython.display import HTML
import base64
with open(out_file, 'rb') as fh:
    encoded = base64.b64encode(fh.read()).decode('utf-8')
display(HTML(f'<a download="submission_MAX_{final_n}.csv" href="data:text/csv;base64,{encoded}" target="_blank" style="font-size:24px; font-weight:bold; color:green;">⬇️ DOWNLOAD MAX N={final_n} ⬇️</a>'))
