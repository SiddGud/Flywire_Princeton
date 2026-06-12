# Technical Approach: Three-Way Common Induced Subgraph Isomorphism Across *Drosophila* Connectomes

**Siddhant Gudwani | Result: 14,484 strictly homologous neurons | Zero edge violations | Single weakly-connected component**

---

## Problem Formulation

Given three independently reconstructed *Drosophila* connectomes - FAFB (783k neurons), BANC (626k neurons), and MCNS (900k neurons) - I sought the largest set of neurons forming a common induced subgraph that is simultaneously isomorphic across all three, where the result must constitute a single weakly connected component.

Formally, I searched for the largest bijection f: V_BANC -> V_FAFB and g: V_BANC -> V_MCNS such that for every pair of matched neurons (b_i, b_j):

```
edge(b_i, b_j) in BANC  <==>  edge(f(b_i), f(b_j)) in FAFB  <==>  edge(g(b_i), g(b_j)) in MCNS
```

This is a three-way maximum common induced subgraph problem, which is NP-hard. The three edge lists total over 330 MB; exact solvers are computationally intractable at this scale.

---

## Mathematical Formulation

Let $G_B = (V_B, E_B)$, $G_F = (V_F, E_F)$, and $G_M = (V_M, E_M)$ denote the directed graphs for the BANC, FAFB, and MCNS connectomes respectively. The goal is to find the maximum cardinality set $S \subseteq V_B$ and injective mappings $\phi: S \to V_F$, $\psi: S \to V_M$ such that the **strict isomorphism constraint** holds for all pairs $(u, v) \in S \times S$:

$$
(u, v) \in E_B \iff (\phi(u), \phi(v)) \in E_F \iff (\psi(u), \psi(v)) \in E_M
$$

subject to the **connectivity constraint** that the induced subgraph $G_B[S]$ forms a single weakly connected component.

**Signature Function.** For a node $v \notin S$ (a frontier candidate) and the current matching $\phi, \psi$ restricted to $S$, define the structural signature:

$$
\sigma(v) = \left( \;\text{sort}\!\left(\{i \mid (b_i, v) \in E_B,\; b_i \in S\}\right),\; \text{sort}\!\left(\{i \mid (v, b_j) \in E_B,\; b_j \in S\}\right) \right)
$$

where $i$ is the positional index of $b_i$ in the current ordering of $S$. Two candidates $v \in V_B$, $f \in V_F$, $m \in V_M$ are a valid match if and only if $\sigma_B(v) = \sigma_F(f) = \sigma_M(m)$ - i.e., they connect identically to the already-matched core.

**Growth Operator.** Let $\mathcal{C}(S)$ denote the set of all valid frontier candidate triplets at state $S$. The growth operator $\mathcal{G}$ is:

$$
\mathcal{G}(S) = \text{LWCC}\!\left( S \cup \{(b, f, m) \in \mathcal{C}(S) : \text{no violation introduced}\} \right)
$$

where LWCC extracts the largest weakly connected component. The algorithm iterates $S \leftarrow \mathcal{G}(S)$ until $\mathcal{G}(S) = S$.

**Perturbation Operator.** Let $\delta_S(v)$ denote the internal degree of node $v$ in $G_B[S]$. The degree-weighted perturbation samples a removal set $R$ from the lowest-degree nodes:

$$
R \sim \text{Uniform}\!\left(\{v \in S : \delta_S(v) \leq Q_\alpha(\delta_S)\}^{(k)}\right)
$$

where $Q_\alpha$ is the $\alpha$-quantile of internal degrees, $k = \lfloor \epsilon |S| \rfloor$ for fraction $\epsilon \in [0.02, 0.08]$, and the superscript $(k)$ denotes a size-$k$ subset.

**MCTS Value Function.** For a frontier candidate $c = (b, f, m)$, the MCTS rollout estimate of its value is:

$$
V(c \mid S) = \left|\mathcal{G}^T(S \cup \{c\})\right|
$$

where $\mathcal{G}^T$ denotes $T$ applications of the growth operator (a finite-horizon rollout). The committed candidate at each MCTS round is:

$$
c^* = \arg\max_{c \in \mathcal{C}(S')} V(c \mid S'), \quad S' = S \setminus R
$$

**Genetic Crossover Operator.** Given two parent solutions $S_1, S_2$ with matchings $(\phi_1, \psi_1)$ and $(\phi_2, \psi_2)$, the crossover produces offspring:

$$
S_{\text{cross}} = \mathcal{G}\!\left(\text{LWCC}\!\left(\{b \in S_1 \cap S_2 : \phi_1(b) = \phi_2(b) \text{ and } \psi_1(b) = \psi_2(b)\}\right)\right)
$$

Conflicting nodes (same $b$ but different $\phi$ or $\psi$ assignments) are discarded. The offspring is then mutated by applying $\mathcal{G}(\mathcal{P}_\epsilon(S_{\text{cross}}))$ for perturbation fraction $\epsilon$.

**Quadratic Assignment (Spectral FAQ).** The boundary alignment problem is cast as:

$$
\max_{P \in \mathcal{P}_{n}} \text{tr}(A^T P B P^T)
$$

where $A \in \{0,1\}^{n \times n}$ and $B \in \{0,1\}^{n \times n}$ are the adjacency matrices of the BANC and FAFB boundary halos respectively, $\mathcal{P}_n$ is the set of $n \times n$ permutation matrices, and $n$ is the halo size (up to 1,500). The continuous relaxation (FAQ) optimizes over doubly-stochastic matrices, then the result is rounded and verified discretely against MCNS.



---

## Foundational Algorithm: Signature-Based Iterative Growth

The core algorithmic insight that underpins every subsequent notebook is **edge signature matching**. If neuron `b` in BANC is adjacent to already-matched core neurons `{b_1, b_2}` (inbound) and `{b_3}` (outbound), then its valid FAFB match `f` must connect to exactly `{f(b_1), f(b_2)}` and `{f(b_3)}`. The sorted neighborhood pattern acts as a unique structural fingerprint:

```python
signature(candidate) = (
    tuple(sorted(inbound_core_neighbor_indices)),
    tuple(sorted(outbound_core_neighbor_indices))
)
```

Two candidates from different connectomes are potentially valid matches if and only if their signatures are identical. This reduces the search from O(N^3) enumeration to near-linear per growth iteration.

The **iterative grow loop** proceeds as follows:
1. Compute signatures of all frontier nodes adjacent to the current core in BANC, FAFB, and MCNS.
2. Identify signature keys shared across all three connectomes.
3. For each shared signature, form candidate triplets and attempt strict addition.
4. A triplet `(b, f, m)` is accepted only if it introduces zero edge violations against every existing member.
5. Repeat until convergence; extract the largest weakly connected component.

This procedure guarantees zero violations by construction at every step.

---

## Phase 1: Initial Seeding Strategies

The growth algorithm is sensitive to the quality of the initial seed - starting from a random singleton typically yields very small subgraphs.

I evaluated three seeding approaches in parallel:

**Cell-type bipartite matching (Hungarian algorithm).** I fetched cell-type classifications from the Codex metadata API and computed the optimal one-to-one assignment across connectomes using the Hungarian algorithm. This produced an initial seed of approximately 4,780 triplets.

**NBLAST morphology matching.** Precomputed NBLAST morphological similarity scores were used to generate candidate triplets across connectomes, yielding approximately 4,500 initial nodes.

**KMeans degree-distribution clustering.** Neurons were grouped by degree and cell-type distribution, then matched cluster-by-cluster. This reached approximately 5,600 initial nodes.

The Hungarian cell-type seed provided the best starting point and was used as the foundation for subsequent optimization.

---

## Phase 2: Iterative Perturbation and Parallel Grow

The signature-grow algorithm is a greedy procedure that saturates at local maxima. To escape these, I introduced an outer **perturbation loop**:

1. Take the current best core.
2. Remove a random fraction (2-8%) of nodes from the core.
3. Re-run the signature-grow algorithm from the perturbed state.
4. If the result is larger, replace the current best.

I parallelized this using `multiprocessing.Pool` across 4 workers, running different perturbation fractions `[0.02, 0.03, 0.05, 0.07, 0.08]` simultaneously and retaining the best result at each round.

Analysis of early-run logs showed that fractions of 5% and 8% produced over 95% of all improvements. Subsequent runs ("Season 2") focused exclusively on the productive fraction range, achieving approximately 3x more useful attempts per compute window. The result climbed from ~5,600 to 16,255 and then to 17,676 nodes.

---

## Phase 3: Degree-Weighted Smart Perturbation

A key refinement was introduced in the third optimization pass: **degree-weighted node removal**. Rather than removing nodes uniformly at random, I preferentially targeted low-degree boundary nodes - those with the fewest internal edges - while preserving well-connected hub nodes.

```python
degree = compute_internal_degrees(core)
sorted_keys = sorted(keys, key=lambda b: degree[b])  # ascending by degree
pool = sorted_keys[:n_remove * 3]                     # pool of boundary candidates
to_remove = rng.choice(pool, size=n_remove)           # sample from boundary
```

The intuition is that high-degree hub neurons are structurally central and almost certainly correct matches, while low-degree peripheral neurons are most likely to be causing topological conflicts that prevent further growth. This strategy preserved the stable core skeleton while allowing the outer shell to reorganize, reaching **19,827 nodes** - the highest raw count achieved.

---

## Phase 4: Connectivity Constraint and Pipeline Redesign

Upon applying the weakly-connected component requirement - extracting the single largest BFS-reachable component from the 19,827-node result - the subgraph reduced to approximately 100 nodes.

The cause was that the greedy growth procedure had been simultaneously expanding multiple disconnected clusters across the connectome, each individually isomorphic but sharing no edges. The optimization objective (maximize total node count) and the evaluation constraint (single connected component) were misaligned.

This required a fundamental redesign: `extract_lwcc()` was moved inside the grow loop itself, applied at every iteration, so that connectivity was enforced throughout optimization rather than as a post-processing step.

```python
def run_grow(starting_core, seed):
    core = dict(starting_core)
    for _ in range(max_iters):
        # ... signature grow step ...
    return extract_lwcc(core)  # enforced at every call
```

All subsequent phases operated under this constraint from the first iteration.

---

## Phase 5: High-Degree Seed Racing

With connectivity enforced, a fresh seeding strategy was needed. I computed the degree of every node in all three connectomes and selected the top 2,000 highest-degree BANC nodes as candidate seeds. These were paired rank-by-rank with the top 2,000 in FAFB and MCNS, forming 2,000 seed triplets that were raced in parallel across 4 workers. The largest resulting connected subgraph was retained.

The rationale is that hub neurons - being highly connected - are most likely to lie at the center of a large connected isomorphic region rather than scattered across disconnected islands.

This produced an initial connected core of approximately 8,526 nodes, which served as the seed for all subsequent phases.

---

## Phase 6: Monte Carlo Tree Search

The largest single improvement came from applying **Monte Carlo Tree Search (MCTS)** as the outer optimization strategy, replacing the myopic perturbation loop.

The fundamental limitation of greedy growth is that it cannot account for how each addition affects future growth opportunities. Adding a given neuron may be immediately valid while simultaneously closing off dozens of other candidates due to the strict edge constraint.

MCTS addresses this by evaluating the **future value** of each candidate addition through rollout simulations:

1. Perturb the current best core by removing 2-6% of low-degree nodes.
2. Identify all valid frontier candidates adjacent to the perturbed core.
3. For each candidate, temporarily force-add it to the core and run a fast greedy rollout for 15 iterations to estimate future growth potential.
4. Permanently commit to the candidate that produced the largest rollout - the one that opens the most downstream growth.
5. Run up to 16 candidate rollouts simultaneously across 4 parallel workers.

```python
# Evaluate the future value of each frontier candidate
tasks = [(perturbed_core, candidate, seed) for candidate in frontier]
rollout_results = pool.map(mcts_rollout_worker, tasks)

# Commit to the branch with highest long-term potential
best_n, best_core = max(rollout_results, key=lambda r: r[0])
```

The first MCTS run produced 13,427 connected nodes. After tuning the perturbation schedule and rollout depth, the final run reached **14,484 connected nodes** with zero violations - the submitted result.

---

## Phase 7: Genetic Algorithm and Spectral Relaxation

Two additional strategies were pursued in parallel to attempt to push beyond 14,484.

**Genetic Algorithm with Crossover (NB5).** Multiple independent runs of the growth algorithm converge to different local maxima. A genetic algorithm was implemented to combine these: crossover merges all non-conflicting nodes from two parent cores, discards conflicts, re-verifies strict isomorphism, and extracts the LWCC. Mutation removes 2-6% of low-degree nodes, followed by regrowth. A population of 10 genomes was maintained with tournament selection biased toward larger cores. This reached 14,955 nodes.

**Spectral FAQ Continuous Relaxation (NB6).** This approach used `scipy.optimize.quadratic_assignment` with the FAQ algorithm to perform soft probabilistic alignment of the boundary "halo" (up to 1,500 nodes) of the current core, augmented with ghost edges encoding core affinities. The continuous soft matches were then snapped to strict discrete triplets by pairwise verification against MCNS. This reached 15,083 nodes. Both results required additional connectivity pruning and the cleanest strictly-verified connected answer remained 14,484 from the MCTS run.

---

## Final Verification

The 14,484-node result was verified by exhaustive pairwise edge checking across all three connectomes:

```python
violations = 0
for (b_i, f_i, m_i) in triplets:
    for (b_j, f_j, m_j) in triplets:
        banc_has = (b_i, b_j) in banc_set
        fafb_has = (f_i, f_j) in fafb_set
        mcns_has = (m_i, m_j) in mcns_set
        if banc_has != fafb_has or banc_has != mcns_has:
            violations += 1
```

**Result: 0 violations across 209,784,256 directed edge pair checks (14,484^2).**
Runtime: 397 seconds across all three edge lists. Single weakly-connected component confirmed via BFS.

---

## Reproduction Instructions

### Prerequisites
```bash
pip install pandas numpy scipy
```

### Data
The three competition edge lists are required:
- `fafb_783_edge_list.csv` (~145 MB)
- `banc_626_edge_list.csv` (~104 MB)
- `mcns_0.9_edge_list.csv` (~86 MB)

### Steps

**1. Generate the connected seed**
```bash
python kaggle_nb2_highdeg.py
# Output: ~8,526 connected nodes
```

**2. Run MCTS search** (upload NB2 output as a Kaggle dataset, then run)
```bash
python kaggle_nb4_mcts.py
# Output: submission_NB4_MCTS_14484.csv
```

**3. Verify**
```bash
python verify_14484.py
# Confirms: 0 violations, 1 connected component, 14,484 nodes
```

### Key Assumptions
- **Strict isomorphism**: Every directed edge in the subgraph must exist simultaneously in all three connectomes.
- **Bijection**: Each BANC neuron maps to exactly one FAFB and one MCNS neuron.
- **Connectivity**: Single weakly-connected component, enforced throughout optimization.
- **Structure-only growth**: Cell-type labels were used only for initial seeding. The growth algorithm is purely edge-driven.

---

## Progression Summary

| Phase | Strategy | Peak Nodes | Connected |
|---|---|---|---|
| 1 | Cell-type Hungarian + grow | 4,780 | Post-hoc only |
| 2 | Iterative perturbation (Seasons 1-2) | 17,676 | Post-hoc only |
| 3 | Degree-weighted smart perturbation (Season 3) | 19,827 | Post-hoc only |
| - | LWCC enforcement applied to 19,827-node result | ~100 | Enforced |
| 4 | High-degree seed racing (NB2, connectivity-first) | 8,526 | Enforced |
| 5 | Genetic algorithm crossover (NB5) | 14,955 | Enforced |
| 6 | Spectral FAQ relaxation (NB6) | 15,083 | Enforced |
| **Final** | **MCTS guided search (NB4)** | **14,484** | **Enforced** |

---

## Repository Structure

| File | Description |
|---|---|
| `kaggle_nb4_mcts.py` | MCTS search - produced the final 14,484 result |
| `kaggle_nb5_genetic.py` | Genetic algorithm with crossover |
| `kaggle_nb6_spectral_faq.py` | Spectral FAQ continuous relaxation |
| `kaggle_nb2_highdeg.py` | High-degree seed racing |
| `kaggle_maximize_s3.py` | Season 3 degree-weighted perturbation |
| `kaggle_maximize_s2.py` | Season 2 refined perturbation |
| `hungarian_bijection.py` | Cell-type bipartite matching seeder |
| `verify_14484.py` | Final pairwise verification script |
| `submission_NB4_MCTS_14484.csv` | **Final answer (14,484 triplets)** |
| `science.md` | Biological significance report |
| [Interactive 3D Visualization](https://siddgud.github.io/14k_interactive_3d_network/14k_interactive_3d_network.html) | Force-directed 3D graph of the 14,484-node subgraph |
