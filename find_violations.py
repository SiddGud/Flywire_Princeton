import pandas as pd

sub = pd.read_csv('submission_ITER_GROW_13083.csv', dtype=str)

banc_df = pd.read_csv('banc_626_edge_list.csv', header=None, names=['src','tgt'], dtype=str)
fafb_df = pd.read_csv('fafb_783_edge_list.csv', header=None, names=['src','tgt'], dtype=str)
mcns_df = pd.read_csv('mcns_0.9_edge_list.csv', header=None, names=['src','tgt'], dtype=str)

b_nodes = set(sub['BANC']); f_nodes = set(sub['FAFB']); m_nodes = set(sub['MCNS'])
m2b = dict(zip(sub['MCNS'], sub['BANC']))
m2f = dict(zip(sub['MCNS'], sub['FAFB']))

banc_set = set(zip(banc_df['src'], banc_df['tgt']))
fafb_set = set(zip(fafb_df['src'], fafb_df['tgt']))

# Find the 2 violating MCNS edges
print("=== FINDING THE 2 VIOLATING MCNS EDGES ===")
violations = []
for s, t in zip(mcns_df['src'], mcns_df['tgt']):
    if s in m_nodes and t in m_nodes:
        bs, bt = m2b[s], m2b[t]
        if (bs, bt) not in banc_set:
            violations.append((s, t, bs, bt, m2f[s], m2f[t]))
            print(f"  MCNS edge ({s} -> {t})")
            print(f"    Maps to BANC ({bs} -> {bt}) -- MISSING!")
            print(f"    Maps to FAFB ({m2f[s]} -> {m2f[t]}) -- {'EXISTS' if (m2f[s], m2f[t]) in fafb_set else 'MISSING'}")
            print(f"    Self-loop? {s == t}")
            print()

# Find ghost BANC nodes
all_banc = set(banc_df['src']) | set(banc_df['tgt'])
ghosts = set(sub['BANC']) - all_banc
print(f"\n=== GHOST BANC NODES (not in edge list): {len(ghosts)} ===")
print(f"First 10: {list(ghosts)[:10]}")

# Check: are these ghosts from the original 5092 seed or from the signature grow?
orig = pd.read_csv('submission_CT_FINE_5092.csv', dtype=str)
orig_banc = set(orig['BANC'])
ghosts_from_seed = ghosts & orig_banc
ghosts_from_grow = ghosts - orig_banc
print(f"\n  Ghosts from original 5092 seed: {len(ghosts_from_seed)}")
print(f"  Ghosts from signature grow: {len(ghosts_from_grow)}")

# What about the violating nodes - are they ghosts?
print("\n=== ARE VIOLATING NODES GHOSTS? ===")
for (ms, mt, bs, bt, fs, ft) in violations:
    print(f"  BANC src {bs}: {'GHOST' if bs not in all_banc else 'EXISTS'}")
    print(f"  BANC tgt {bt}: {'GHOST' if bt not in all_banc else 'EXISTS'}")
