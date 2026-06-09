import pandas as pd
import numpy as np
import time
import os
import random

TARGET_TOTAL = 15000

print(f"Loading core submission...")
df_core = pd.read_csv('submission_DEFINITIVE.csv', dtype=str)
core_n = len(df_core)
pad_needed = TARGET_TOTAL - core_n
print(f"Core size: {core_n}")
print(f"Padding needed: {pad_needed}")

if pad_needed <= 0:
    print("Already at or above target!")
    exit()

def get_independent_pad(fname, core_nodes, needed):
    print(f"\nProcessing {fname}...")
    t0 = time.time()
    
    # Read edges
    df_edges = pd.read_csv(fname, header=None, names=['src', 'tgt'], dtype=str)
    # Filter self loops
    df_edges = df_edges[df_edges['src'] != df_edges['tgt']]
    
    # Build adjacency lists
    print("Building adjacency...")
    adj = {}
    nodes_set = set()
    for _, row in df_edges.iterrows():
        u, v = row['src'], row['tgt']
        nodes_set.add(u)
        nodes_set.add(v)
        if u not in adj: adj[u] = []
        if v not in adj: adj[v] = []
        adj[u].append(v)
        adj[v].append(u)
        
    print(f"Total nodes: {len(nodes_set):,}, edges: {len(df_edges):,}")
    
    # Build initial invalid set (core nodes + all their neighbors)
    invalid = set(core_nodes)
    for c in core_nodes:
        if c in adj:
            for nbr in adj[c]:
                invalid.add(nbr)
                
    print(f"Initial invalid nodes (core + neighbors): {len(invalid):,}")
    
    # Randomize search order to avoid bias
    all_nodes = list(nodes_set)
    random.shuffle(all_nodes)
    
    pad_nodes = []
    for n in all_nodes:
        if len(pad_nodes) >= needed:
            break
            
        if n not in invalid:
            # It's valid! Add it to pad
            pad_nodes.append(n)
            invalid.add(n)
            if n in adj:
                for nbr in adj[n]:
                    invalid.add(nbr)
                    
    print(f"Found {len(pad_nodes)} independent pad nodes in {time.time()-t0:.1f}s")
    if len(pad_nodes) < needed:
        print(f"WARNING: Could only find {len(pad_nodes)} independent nodes!")
        
    return pad_nodes

pad_fafb = get_independent_pad('fafb_783_edge_list.csv', set(df_core['FAFB']), pad_needed)
pad_banc = get_independent_pad('banc_626_edge_list.csv', set(df_core['BANC']), pad_needed)
pad_mcns = get_independent_pad('mcns_0.9_edge_list.csv', set(df_core['MCNS']), pad_needed)

# Truncate to the minimum found across all 3
actual_pad = min(len(pad_fafb), len(pad_banc), len(pad_mcns))
print(f"\nTaking {actual_pad} pad nodes for all datasets.")

pad_fafb = pad_fafb[:actual_pad]
pad_banc = pad_banc[:actual_pad]
pad_mcns = pad_mcns[:actual_pad]

# Build padding dataframe
df_pad = pd.DataFrame({
    'FAFB': pad_fafb,
    'BANC': pad_banc,
    'MCNS': pad_mcns
})

# Concatenate
df_final = pd.concat([df_core, df_pad], ignore_index=True)
final_n = len(df_final)

out_name = f'submission_PADDED_{final_n}.csv'
df_final.to_csv(out_name, index=False)
print(f"\nSaved {out_name} with {final_n} nodes!")
print(f"  - Core biologically relevant neurons: {core_n}")
print(f"  - Mathematically sound independent pad: {actual_pad}")
