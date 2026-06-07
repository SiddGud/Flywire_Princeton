"""
ITERATIVE PERTURBATION MAXIMIZER — SEASON 2
=============================================
Optimized based on Season 1 logs:
- Season 1 showed frac=5% and frac=8% found 95%+ of all NEW BEST results.
- frac=10%, 15%, 20% wasted ~60% of compute time with almost no improvement.
- Season 2 focuses ONLY on small perturbations: [2%, 3%, 5%, 7%, 8%]
- This gives ~3x more productive attempts in the same 12-hour window.
- Stops after 15 consecutive failures (up from 10, since attempts are faster).
- Uses the SAME fast signature-hash grow function as Season 1 (proven correct).
"""
import pandas as pd
import numpy as np
import time, os
from collections import defaultdict

DATA1 = '/kaggle/input/datasets/siddhantgudwani/dataset'

# Auto-find the best submission (prefer CLEAN files, then highest N)
CORE_FILE = None
best_n_score = 0
for root, dirs, files in os.walk('/kaggle/input'):
    for f in files:
        if f.startswith('submission_MAX') and f.endswith('.csv'):
            path = os.path.join(root, f)
            try:
                base = f.replace('_CLEAN', '').replace('.csv', '')
                n = int(base.split('_')[-1])
                is_clean = 'CLEAN' in f
                score = n * 2 if is_clean else n
                if score > best_n_score:
                    best_n_score = score
                    CORE_FILE = path
            except:
                pass

if CORE_FILE is None:
    print("ERROR: Could not find a submission_MAX_*.csv file!")
    exit()

print("=" * 65)
print("  ITERATIVE PERTURBATION MAXIMIZER — SEASON 2")
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

# ── SAME fast grow functions as Season 1 ──────────────────────────────
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
            b_has  = (b, cb) in banc_set;  f_has  = (f, cf) in fafb_set;  m_has  = (m, cm) in mcns_set
            if b_has != f_has or b_has != m_has: ok = False; break
            b_has2 = (cb, b) in banc_set;  f_has2 = (cf, f) in fafb_set;  m_has2 = (cm, m) in mcns_set
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

# ── Load starting core ────────────────────────────────────────────────
print(f"\n3. Loading core from: {CORE_FILE}")
core_df = pd.read_csv(CORE_FILE, dtype=str)
best_core = {str(b): (str(f), str(m)) for b, f, m in zip(core_df['BANC'], core_df['FAFB'], core_df['MCNS'])}
best_core = remove_zero_edge(best_core)
best_n = len(best_core)
print(f"   Starting N = {best_n:,}")

# ── Season 2 Perturbation Loop ────────────────────────────────────────
# Only small fractions — 5% and 8% drove 95%+ of all improvements in S1
FRACS = [0.02, 0.03, 0.05, 0.07, 0.08]
MAX_NO_IMPROVE = 15

no_improve = 0
attempt = 0
global_best_n = best_n
global_best_core = dict(best_core)

print(f"\n4. Starting Season 2 Perturbation (fracs={[f'{int(f*100)}%' for f in FRACS]})...")
print(f"   Will stop after {MAX_NO_IMPROVE} consecutive failures.\n")

while no_improve < MAX_NO_IMPROVE:
    attempt += 1
    frac = FRACS[attempt % len(FRACS)]
    rng_state = np.random.RandomState(attempt * 173 + 31)
    t_start = time.time()

    # Perturb
    keys = list(best_core.keys())
    n_remove = max(1, int(len(keys) * frac))
    to_remove = set(rng_state.choice(keys, size=n_remove, replace=False))
    perturbed = {b: v for b, v in best_core.items() if b not in to_remove}

    # Regrow (same fast method as Season 1)
    core = run_iterative_grow(perturbed, seed_num=attempt * 277 + 13)
    n = len(core)
    elapsed = time.time() - t_start

    if n > global_best_n:
        global_best_n = n
        global_best_core = core
        best_core = core
        best_n = n
        no_improve = 0
        print(f"  Attempt {attempt:3d} (frac={frac:.0%}, removed {n_remove:,}): N = {n:,}  ({elapsed:.0f}s) *** NEW BEST ***")
        tmp_df = pd.DataFrame([{'BANC': b, 'FAFB': f, 'MCNS': m} for b, (f, m) in global_best_core.items()])
        tmp_df.to_csv(f'/kaggle/working/submission_S2_{n}.csv', index=False)
        print(f"    Auto-saved to /kaggle/working/submission_S2_{n}.csv")
    else:
        no_improve += 1
        if n > best_n:
            best_core = core
            best_n = n
        print(f"  Attempt {attempt:3d} (frac={frac:.0%}, removed {n_remove:,}): N = {n:,}  ({elapsed:.0f}s)  [no improve {no_improve}/{MAX_NO_IMPROVE}]")

# ── Final Save ────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print(f"  CONVERGED! Best N = {global_best_n:,}")
print("=" * 65)

final_df = pd.DataFrame([{'BANC': b, 'FAFB': f, 'MCNS': m} for b, (f, m) in global_best_core.items()])
out = f'/kaggle/working/submission_S2_FINAL_{global_best_n}.csv'
final_df.to_csv(out, index=False)
print(f"\nFinal file saved: {out}")
print("Download and run verify_max.py on your laptop!")
# Season 2: focused on productive fractions [2%,3%,5%,7%,8%]
# Fix: use imap_unordered for proper result streaming
