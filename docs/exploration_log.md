# Exploration Log: Full Chronological Record of All Strategies Attempted

This document catalogues every algorithm and strategy attempted during development, including failed and intermediate approaches. The `src/` folder contains only the final pipeline; this document preserves the full research record.

---

## Phase 1: Initial Seeding Strategies (June 4)

### `hungarian_bijection.py` *(in src/)*
**Strategy:** Cell-type bipartite matching using the Hungarian algorithm.

Fetched cell-type classification metadata from the Codex API for all neurons in BANC, FAFB, and MCNS. Built a bipartite cost matrix where entry (i,j) is the negative overlap score between cell-type distribution of BANC neuron i and FAFB neuron j. Ran `scipy.optimize.linear_sum_assignment` (Hungarian algorithm) to find the optimal one-to-one assignment. The matched triplets were then fed into the signature-grow loop.

**Result:** ~4,780 connected nodes. Best initial seed found; used as starting point for subsequent SA sweeps.

---

### `kaggle_ct_sweep.py` *(exploratory, not in src/)*
**Strategy:** Cell-type weighted Simulated Annealing sweep over CT-seeded triplets.

Pre-built a pool of `ct_weighted_triplets.csv` - candidate triplets scored by cross-connectome cell-type overlap. Ran a parallel SA sweep over 10 temperature schedules (alpha ∈ {1.2, 1.5, 2, 3, 5, 8, 10, 15, 20, ∞}) × 30 seeds = 271 SA configurations simultaneously across all CPU cores using `multiprocessing.Pool`. Each SA worker removed the highest-conflict node at each step, weighted by alpha-powered degree. Best result was then grown via strict grow.

**Result:** ~5,083 → 5,092 nodes after fine-sweep. Limited by the quality of the CT-triplet pool.

---

### `kaggle_fine_sweep.py` *(exploratory, not in src/)*
**Strategy:** Refined SA sweep over the CT triplet pool with tighter alpha range.

Same architecture as `kaggle_ct_sweep.py` but focused alpha on the range [3, 5, 8] where the best results clustered in the broad sweep. Ran more seeds per alpha value. Produced marginal improvement to 5,092 nodes.

---

## Phase 2: Topology and KMeans Approaches (June 5)

### `kaggle_topo_sweep.py` *(exploratory, not in src/)*
**Strategy:** Topological signature-based triplet generation + SA.

Rather than using cell-type metadata, this approach generated candidate triplets based purely on network topology: neurons with identical in-degree and out-degree distributions within their 2-hop neighborhood were considered candidate matches. Built `topological_triplets.csv` as the seed pool. Ran SA sweep on this pool.

**Result:** Did not outperform the CT-seeded approach. The topological similarity criterion was too broad and the pool contained too many false positives.

---

### `kaggle_generate_kmeans.py` *(exploratory, not in src/)*
**Strategy:** KMeans clustering of neurons by degree-distribution feature vectors for triplet generation.

For each neuron in BANC, FAFB, and MCNS, computed a feature vector: [in-degree, out-degree, clustering coefficient, number of distinct 2-hop neighbors]. Applied KMeans with K=500 clusters per connectome. Neurons in matching cluster IDs across connectomes were treated as candidate triplets. Saved `kmeans_triplets.csv`.

---

### `kaggle_kmeans_sweep.py` *(exploratory, not in src/)*
**Strategy:** SA sweep over the KMeans-generated triplet pool.

Loaded `kmeans_triplets.csv` (a much larger pool than the CT pool). Ran SA sweep with alphas ∈ {2, 5, 8, 10, 15} × 45 seeds = 225 configurations. The larger pool gave the SA more room to find good subsets.

**Result:** ~5,653 nodes - the first result to exceed the CT-seeded approach. This confirmed that degree-distribution clustering produces better initial candidates than cell-type alone.

---

### `kaggle_kmeans_no_zeros.py` *(exploratory, not in src/)*
**Strategy:** KMeans sweep with zero-degree nodes removed from the pool before SA.

Nodes with zero internal edges (isolated in the current pool) were removed before running the SA, since they can never contribute to a connected subgraph. This reduced the pool size and allowed SA to focus on the structurally active subset.

**Result:** Marginal improvement over `kaggle_kmeans_sweep.py`. The insight that zero-edge nodes should be filtered early became standard in all subsequent pipelines.

---

## Phase 3: Signature Grow and Iterative Perturbation (June 5-6)

### `kaggle_signature_grow.py` *(exploratory, not in src/)*
**Strategy:** First implementation of the signature-based grow algorithm.

The key algorithmic breakthrough of the project. Instead of relying on a pre-built triplet pool, this script computed structural fingerprints for all frontier nodes adjacent to the current core and grew directly from the connectome edge lists. Eliminated the quality ceiling imposed by the triplet pool.

**Result:** ~7,822 nodes starting from the KMeans seed - a massive jump. This approach became the backbone of all subsequent work.

---

### `kaggle_iterative_grow.py` *(exploratory, not in src/)*
**Strategy:** Iterative grow with random perturbation loop.

Wrapped `kaggle_signature_grow.py` in an outer perturbation loop: remove 5-10% of nodes randomly, re-grow, keep if better. Single-threaded. The first implementation of the grow-perturb-regrow cycle.

**Result:** 7,822 → 13,083 nodes. Demonstrated that iterative perturbation dramatically outperforms single-pass growth.

---

### `kaggle_multi_trial.py` *(exploratory, not in src/)*
**Strategy:** Multi-trial parallel perturbation with `multiprocessing.Pool`.

Extended `kaggle_iterative_grow.py` to run 4 perturbation trials simultaneously in parallel. Each trial used a different random seed. At the end of each round, the best result across all 4 workers was kept as the new core. This gave approximately 4x the throughput of the single-threaded version.

**Result:** 13,083 → 16,255 nodes. The parallel architecture became standard in all subsequent notebooks.

---

## Phase 4: Season 1 Perturbation Maximizer (June 6)

### `kaggle_maximize.py` *(exploratory, not in src/)*
**Strategy:** Season 1 - full perturbation maximizer with convergence criterion.

Generalized the multi-trial loop with a formal convergence criterion: stop after 10 consecutive failures to improve. Tried perturbation fractions `[0.05, 0.08, 0.10, 0.15, 0.20]` in rotation. Auto-saved whenever a new best was found.

**Result:** 16,255 nodes (plateau). Log analysis after Season 1 revealed that frac=5% and frac=8% produced 95%+ of all improvements, while 10%/15%/20% wasted ~60% of compute.

---

### `kaggle_ultimate_growth.py` *(exploratory, not in src/)*
**Strategy:** Season 1 variant with aggressive multi-start racing.

Launched multiple independent Season 1 runs from different random seeds simultaneously and kept the global best. Designed for the 12-hour Kaggle compute window.

**Result:** ~17,174 nodes. Best single-session result without the Season 2 fraction optimization.

---

## Phase 5: Season 2 and Season 3 (June 7-8)

### `kaggle_maximize_s2.py` *(in src/)*
**Strategy:** Season 2 - refined perturbation fractions based on Season 1 log analysis.

See main README for full description. Focused exclusively on `[2%, 3%, 5%, 7%, 8%]` fractions, achieving ~3x more productive attempts per compute window.

**Result:** 17,174 → 17,676 nodes.

---

### `kaggle_maximize_s3.py` *(in src/)*
**Strategy:** Season 3 - degree-weighted boundary perturbation.

See main README for full description. Introduced smart perturbation preferentially targeting low-degree boundary nodes.

**Result:** 17,676 → **19,827 nodes** - highest count ever achieved.

### `kaggle_maximize_s4.py` *(exploratory, not in src/)*
**Strategy:** Season 4 - attempted continuation after the connectivity crash.

After the 19,827 node result shattered to ~100 nodes upon LWCC extraction, Season 4 attempted to restart from the largest connected fragment of the 19,827 result and grow outward. Starting from ~8,526 connected nodes.

**Result:** Superseded by NB2 high-degree seeding which produced a cleaner starting point.

---

## Phase 6: Connectivity-First Rebuild (June 9)

### `kaggle_nb2_highdeg.py` *(in src/)*
**Strategy:** High-degree seed racing with LWCC enforced from step 1.

See main README for full description. Fresh start with connectivity enforced inside every grow call.

**Result:** 8,526 connected nodes.

---

### `kaggle_nb3_boundary.py` *(exploratory, not in src/)*
**Strategy:** Boundary-expansion seeding.

Rather than seeding from a single high-degree neuron, this approach started from the "boundary" of the LWCC - the set of nodes with exactly one connection to the core - and attempted to expand inward. The hypothesis was that boundary nodes were the most flexible attachment points.

**Result:** Did not outperform the high-degree hub seeding. Boundary nodes tend to be less reliable anchors because they are structurally ambiguous.

---

## Phase 7: Advanced Search (June 10-12)

### `kaggle_nb4_mcts.py` *(in src/)* - **FINAL RESULT**
**Strategy:** MCTS guided search with lookahead rollouts.

See main README for full description. Produced the final verified 14,484-node result.

---

### `kaggle_nb5_genetic.py` *(in src/)*
**Strategy:** Genetic algorithm with crossover.

See main README for full description. Reached 14,955 nodes.

---

### `kaggle_nb6_spectral_faq.py` *(in src/)*
**Strategy:** Spectral FAQ continuous relaxation.

See main README for full description. Reached 15,083 nodes (soft matches; strict connected result converged to 14,484).

---

### `kaggle_nb7_spectral.py` *(exploratory, not in src/)*
**Strategy:** Spectral embedding alignment - Laplacian eigenvector matching.

Computed the top-k eigenvectors of the normalized graph Laplacian for each connectome's boundary halo and aligned them using Procrustes rotation. The aligned eigenvector coordinates were used to generate candidate triplets. More mathematically principled than FAQ but slower and less effective at this scale.

**Result:** Did not improve upon NB6.

---

### `kaggle_nb8_pruning.py` *(exploratory, not in src/)*
**Strategy:** Aggressive edge-pruning and re-verification.

Starting from the 15,083-node spectral result, attempted to identify and prune the minimum set of nodes whose removal would resolve all remaining soft-match inconsistencies, then re-grow. Used a greedy violation-removal loop.

**Result:** Converged to the same 14,484-node connected core as the MCTS result, confirming that 14,484 is a robust attractor.

---

### `kaggle_nb9_global_match.py` *(exploratory, not in src/)*
**Strategy:** Global graph alignment using graph edit distance approximation.

Attempted to frame the problem as minimizing graph edit distance between the three connectome adjacency matrices restricted to the matched set. Used a beam-search approximation. Computationally prohibitive at this scale; did not complete within the 12-hour Kaggle window.

---

## Summary Table

| Date | Script | Strategy | Peak Nodes | In src/? |
|---|---|---|---|---|
| Jun 4 | `hungarian_bijection.py` | Cell-type Hungarian assignment | 4,780 | Yes |
| Jun 4 | `kaggle_ct_sweep.py` | CT-weighted SA sweep (271 configs) | 5,092 | No |
| Jun 4 | `kaggle_fine_sweep.py` | Refined CT SA sweep | 5,092 | No |
| Jun 5 | `kaggle_topo_sweep.py` | Topological degree-signature triplets | <5,000 | No |
| Jun 5 | `kaggle_generate_kmeans.py` | KMeans degree-feature clustering | (seed gen) | No |
| Jun 5 | `kaggle_kmeans_sweep.py` | SA sweep over KMeans pool (225 configs) | 5,653 | No |
| Jun 5 | `kaggle_kmeans_no_zeros.py` | KMeans sweep with zero-degree filter | 5,653 | No |
| Jun 5 | `kaggle_signature_grow.py` | First signature-based grow | 7,822 | No |
| Jun 6 | `kaggle_iterative_grow.py` | Iterative grow + random perturbation | 13,083 | No |
| Jun 6 | `kaggle_multi_trial.py` | Parallel multi-trial perturbation | 16,255 | No |
| Jun 6 | `kaggle_maximize.py` | Season 1 maximizer | 16,255 | No |
| Jun 7 | `kaggle_ultimate_growth.py` | Season 1 multi-start racing | 17,174 | No |
| Jun 7 | `kaggle_maximize_s2.py` | Season 2 refined fractions | 17,676 | **Yes** |
| Jun 8 | `kaggle_maximize_s3.py` | Season 3 degree-weighted perturbation | **19,827** | **Yes** |
| Jun 8 | *(connectivity crash)* | LWCC extraction of 19,827 result | ~100 | - |
| Jun 9 | `kaggle_maximize_s4.py` | Season 4 restart attempt | 8,526 | No |
| Jun 9 | `kaggle_nb2_highdeg.py` | High-degree seed racing (LWCC-first) | 8,526 | **Yes** |
| Jun 9 | `kaggle_nb3_boundary.py` | Boundary-expansion seeding | <8,526 | No |
| Jun 10 | `kaggle_nb5_genetic.py` | Genetic algorithm with crossover | 14,955 | **Yes** |
| Jun 11 | `kaggle_nb6_spectral_faq.py` | Spectral FAQ QAP relaxation | 15,083 | **Yes** |
| Jun 11 | `kaggle_nb7_spectral.py` | Laplacian eigenvector alignment | <14,484 | No |
| Jun 11 | `kaggle_nb8_pruning.py` | Aggressive pruning + re-grow | 14,484 | No |
| Jun 11 | `kaggle_nb9_global_match.py` | Global GED approximation | timeout | No |
| **Jun 12** | **`kaggle_nb4_mcts.py`** | **MCTS guided search** | **14,484** | **Yes** |
<!-- checkpoint: 17174 Jun 7 14:20 -->
<!-- season2 final: 17676 -->
<!-- checkpoint: 19583 Jun 8 13:20 -->
<!-- season3 peak: 19827 -->
<!-- CRASH: 19827 -> ~100 under LWCC Jun 8 22:15 -->
<!-- Phase 2 begin: LWCC inside grow loop -->
<!-- nb2 result: 8526 connected Jun 9 12:00 -->
<!-- nb3: boundary seeding did not outperform nb2 -->
<!-- nb5 run1: 15079 Jun 10 13:30 -->
<!-- nb5 final: 14955 connected -->
<!-- nb6: 15083 soft, needs discrete verification -->
<!-- nb4 run1: 13427 connected Jun 11 17:30 -->
<!-- nb4 final run started Jun 12 10:00 -->
<!-- FINAL: 14484 zero violations Jun 12 15:30 -->
<!-- exploration log finalized -->
