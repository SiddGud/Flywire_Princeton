"""
MULTI-TRIAL ITERATIVE SIGNATURE GROW
=====================================
The original grow was GREEDY — it accepted candidates in a fixed order.
This script runs multiple trials with SHUFFLED candidate ordering.
Different orderings → different nodes accepted → different growth trajectories
→ potentially MUCH larger final N.

Also includes PERTURBATION: start by removing random nodes from the
converged 9,788 core to escape the local optimum, then regrow.
"""
import pandas as pd
import numpy as np
import time, os
from collections import defaultdict

# ── Paths ──────────────────────────────────────────────────────────
DATA1 = '/kaggle/input/datasets/siddhantgudwani/dataset'

# Auto-find the 5092 seed
SEED_FILE = None
CONV_FILE = None  # The converged 9788 file (for perturbation)
for root, dirs, files in os.walk('/kaggle/input'):
    for f in files:
        if 'submission_CT_FINE_5092' in f and SEED_FILE is None:
            SEED_FILE = os.path.join(root, f)
        if ('CONNECTED_9788' in f or 'SIG_GROW_7822' in f or 'ITER_GROW' in f) and CONV_FILE is None:
            CONV_FILE = os.path.join(root, f)

if SEED_FILE is None:
    print("ERROR: Could not find submission_CT_FINE_5092.csv!")
    exit()
print(f"Seed file: {SEED_FILE}")
if CONV_FILE:
    print(f"Converged file for perturbation: {CONV_FILE}")

print("=" * 65)
print("  MULTI-TRIAL ITERATIVE SIGNATURE GROW")
print("=" * 65)

print("\n1. Loading Edge Lists...")
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

# ── Signature function ─────────────────────────────────────────────
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

# ── Strict Grow (with shuffle) ─────────────────────────────────────
def strict_grow_shuffled(candidates_df, core, rng):
    """Add candidates that don't conflict, but in RANDOM order."""
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

# ── Remove 0-edge nodes ───────────────────────────────────────────
def remove_zero_edge(core):
    """Remove nodes with 0 internal edges."""
    b_nodes = set(core.keys())
    connected = set()
    for s, t in zip(banc_df['src'], banc_df['tgt']):
        if s in b_nodes and t in b_nodes:
            connected.add(s); connected.add(t)
    return {b: v for b, v in core.items() if b in connected}

# ── Iterative grow (one full run) ─────────────────────────────────
def run_iterative_grow(starting_core, seed_num):
    core = dict(starting_core)  # copy
    rng = np.random.RandomState(seed_num)
    
    round_num = 0
    while True:
        round_num += 1
        N_before = len(core)
        
        b2i = {b: i for i, b in enumerate(core.keys())}
        f2i = {f: i for i, (f, m) in enumerate(core.values())}
        m2i = {m: i for i, (f, m) in enumerate(core.values())}
        
        b_sigs = compute_signatures(banc_df, b2i)
        f_sigs = compute_signatures(fafb_df, f2i)
        m_sigs = compute_signatures(mcns_df, m2i)
        
        shared_sigs = set(b_sigs.keys()) & set(f_sigs.keys()) & set(m_sigs.keys())
        
        candidates = []
        for sig in shared_sigs:
            b_nodes = b_sigs[sig]
            f_nodes = f_sigs[sig]
            m_nodes = m_sigs[sig]
            # Shuffle within each signature group too!
            rng.shuffle(b_nodes)
            rng.shuffle(f_nodes)
            rng.shuffle(m_nodes)
            max_pairs = min(len(b_nodes), len(f_nodes), len(m_nodes))
            for i in range(max_pairs):
                if b_nodes[i] not in core:
                    candidates.append({'BANC': b_nodes[i], 'FAFB': f_nodes[i], 'MCNS': m_nodes[i]})
        
        if len(candidates) == 0:
            break
        
        candidates_df = pd.DataFrame(candidates)
        added = strict_grow_shuffled(candidates_df, core, rng)
        
        if added == 0:
            break
        
        if round_num > 20:  # safety cap
            break
    
    # Remove 0-edge nodes and self-loop violators
    core = remove_zero_edge(core)
    return core

# ── Load seed ──────────────────────────────────────────────────────
print(f"\n3. Loading seed from: {SEED_FILE}")
seed_df = pd.read_csv(SEED_FILE, dtype=str)
seed_core = {str(b): (str(f), str(m)) for b, f, m in zip(seed_df['BANC'], seed_df['FAFB'], seed_df['MCNS'])}
print(f"   Seed size: {len(seed_core)}")

# ══════════════════════════════════════════════════════════════════
#  APPROACH 1: Multi-trial from biological seed (different orderings)
# ══════════════════════════════════════════════════════════════════
NUM_TRIALS = 8
best_n = 0
best_core = None
best_trial = -1

print(f"\n4. Running {NUM_TRIALS} trials with different random orderings...")
for trial in range(NUM_TRIALS):
    t_trial = time.time()
    core = run_iterative_grow(seed_core, seed_num=trial * 137 + 42)
    n = len(core)
    elapsed = time.time() - t_trial
    is_best = n > best_n
    if is_best:
        best_n = n
        best_core = core
        best_trial = trial
    marker = " *** NEW BEST ***" if is_best else ""
    print(f"  Trial {trial}: N = {n:,}  ({elapsed:.0f}s){marker}")

# ══════════════════════════════════════════════════════════════════
#  APPROACH 2: Perturbation + Regrowth
#  Start from the CONVERGED core (9788 or 7822) if available,
#  otherwise from the best multi-trial result
# ══════════════════════════════════════════════════════════════════
perturb_core = dict(best_core)
if CONV_FILE:
    print(f"\n5. Loading converged core for perturbation: {CONV_FILE}")
    conv_df = pd.read_csv(CONV_FILE, dtype=str)
    conv_core = {str(b): (str(f), str(m)) for b, f, m in zip(conv_df['BANC'], conv_df['FAFB'], conv_df['MCNS'])}
    conv_clean = remove_zero_edge(conv_core)
    print(f"   Converged core (after removing 0-edge): {len(conv_clean):,}")
    if len(conv_clean) > best_n:
        perturb_core = conv_clean
        best_n = len(conv_clean)
        best_core = conv_clean
        print(f"   Using converged core as new best: {best_n:,}")
    else:
        print(f"   Multi-trial already beat it ({best_n:,} > {len(conv_clean):,}), perturbing multi-trial result")

print(f"\n   Perturbation + Regrowth from N={len(perturb_core):,}...")
NUM_PERTURB = 5
PERTURB_FRACTION = 0.08  # remove 8% of nodes

for p in range(NUM_PERTURB):
    t_perturb = time.time()
    rng = np.random.RandomState(p * 271 + 99)
    
    # Remove random fraction of nodes
    perturbed = dict(perturb_core)
    keys = list(perturbed.keys())
    n_remove = int(len(keys) * PERTURB_FRACTION)
    to_remove = rng.choice(keys, size=n_remove, replace=False)
    for k in to_remove:
        del perturbed[k]
    
    # Regrow
    core = run_iterative_grow(perturbed, seed_num=p * 313 + 7)
    n = len(core)
    elapsed = time.time() - t_perturb
    is_best = n > best_n
    if is_best:
        best_n = n
        best_core = core
    marker = " *** NEW BEST ***" if is_best else ""
    print(f"  Perturb {p} (removed {n_remove}): N = {n:,}  ({elapsed:.0f}s){marker}")

# ── Save Final Result ──────────────────────────────────────────────
final_n = len(best_core)
print("\n" + "=" * 65)
print(f"  FINAL BEST N: {final_n:,}  (from trial {best_trial})")
print("=" * 65)

final_df = pd.DataFrame([{'BANC': b, 'FAFB': f, 'MCNS': m} for b, (f, m) in best_core.items()])
out_file = f'/kaggle/working/submission_MULTI_{final_n}.csv'
final_df.to_csv(out_file, index=False)

from IPython.display import HTML
import base64
with open(out_file, 'rb') as fh:
    encoded = base64.b64encode(fh.read()).decode('utf-8')
display(HTML(f'<a download="submission_MULTI_{final_n}.csv" href="data:text/csv;base64,{encoded}" target="_blank" style="font-size:24px; font-weight:bold; color:green;">⬇️ DOWNLOAD N={final_n} ⬇️</a>'))
