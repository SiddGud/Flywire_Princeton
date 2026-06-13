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

> **Not interested in the math?** [Click here to skip to the algorithmic approach.](#foundational-algorithm-signature-based-iterative-growth)

The three connectomes are directed graphs $G_B$, $G_F$, and $G_M$. We want to find the largest set of neurons $S$ with mappings $\phi$ (BANC→FAFB) and $\psi$ (BANC→MCNS) such that:

$$
(u, v) \in E_B \iff (\phi(u), \phi(v)) \in E_F \iff (\psi(u), \psi(v)) \in E_M
$$

*In plain terms: a synapse from neuron u to neuron v must exist in BANC if and only if the exact same connection exists between their matched partners in FAFB and MCNS. The arrows must agree across all three brains simultaneously - no exceptions. The result must also form a single connected subgraph.*

**Signature Function** — used in [`kaggle_nb2_highdeg.py`](src/kaggle_nb2_highdeg.py), [`kaggle_nb4_mcts.py`](src/kaggle_nb4_mcts.py), and all grow scripts.

For a frontier neuron $v$ not yet matched, its structural fingerprint is the sorted list of which core neurons it connects to:

$$
\sigma(v) = \bigl(\,\text{sort}(\text{in-neighbors in core}),\; \text{sort}(\text{out-neighbors in core})\bigr)
$$

*Two neurons from different connectomes can be matched only if their fingerprints are identical. This turns a massive exhaustive search into a fast hash-table lookup.*

**Growth Operator** — [`kaggle_nb4_mcts.py`](src/kaggle_nb4_mcts.py), [`kaggle_maximize_s2.py`](src/kaggle_maximize_s2.py), [`kaggle_maximize_s3.py`](src/kaggle_maximize_s3.py)

$$
\mathcal{G}(S) = \text{LWCC}\!\left( S \cup \{\text{valid frontier triplets}\} \right)
$$

*This finds every neuron on the boundary that can be added safely, adds them all, and keeps only the largest connected component (LWCC). Repeated application drives the subgraph toward its local maximum.*

**Perturbation Operator** — [`kaggle_maximize_s2.py`](src/kaggle_maximize_s2.py), [`kaggle_maximize_s3.py`](src/kaggle_maximize_s3.py)

To escape local maxima, we remove a fraction $\epsilon \in [0.02, 0.08]$ of nodes. Season 3 removes from the **lowest-degree** nodes first (boundary nodes), preserving the stable hub core:

$$
R \;\sim\; \text{sample}\!\left(\{v \in S : \text{degree}(v) \leq \text{bottom quantile}\},\; \text{size} = \lfloor \epsilon \lvert S \rvert \rfloor \right)
$$

*Low-degree peripheral neurons are the most likely source of topological blockages. Targeting them - rather than random removal - produced measurably faster convergence.*

**MCTS Value Function** — [`kaggle_nb4_mcts.py`](src/kaggle_nb4_mcts.py)

For each candidate $c$, estimate how much the subgraph will grow if we commit to $c$ now:

$$
V(c \mid S) = \lvert \mathcal{G}^{15}(S \cup \{c\}) \rvert
$$

We permanently commit to the candidate with the highest future value: $c^* = \arg\max_{c} \; V(c \mid S)$.

*Greedy growth adds any valid node immediately, even if it blocks better additions later. MCTS looks 15 grow-steps ahead before committing. This lookahead is what pushed the result from 13,427 to 14,484.*

**Genetic Crossover Operator** — [`kaggle_nb5_genetic.py`](src/kaggle_nb5_genetic.py)

Given two parent solutions $S_1$ and $S_2$, keep only the neurons both agreed on, discard conflicts, and mutate/regrow:

$$
S_{\text{cross}} = \mathcal{G}\!\left(\text{LWCC}\!\left(\{b \in S_1 \cap S_2 : \phi_1(b) = \phi_2(b),\; \psi_1(b) = \psi_2(b)\}\right)\right)
$$

*Different random seeds find different local maxima. The crossover merges the parts they agree on - producing a higher-quality consensus seed than either parent alone.*

**Quadratic Assignment (Spectral FAQ)** — [`kaggle_nb6_spectral_faq.py`](src/kaggle_nb6_spectral_faq.py)

Align the boundary region of the subgraph by finding the permutation $P$ of FAFB neurons that maximizes edge overlap with BANC:

$$
\max_{P}\; \text{tr}(A^\top P B P^\top), \quad P \in \text{permutation matrices}
$$

*The exact solution is NP-hard. The FAQ algorithm relaxes this to a continuous optimization, solves it via gradient ascent, then rounds back to a valid discrete matching and verifies against MCNS.*



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

## Phases 1-3: Initial Seeding and Perturbation

The growth algorithm requires a strong initial seed to prevent early stagnation.
1. **Seeding:** I evaluated multiple approaches (NBLAST morphology, KMeans degree clustering), but found that cell-type bipartite matching (via the Hungarian algorithm on Codex metadata) provided the best foundation.
2. **Greedy Growth:** The signature-matching loop was run to grow the subgraph from the initial seed. However, greedy addition naturally saturates at local maxima.
3. **Smart Perturbation:** To escape these local traps, I wrapped the growth algorithm in a perturbation loop. By repeatedly removing 5-8% of the lowest-degree boundary nodes and re-growing from the stable core, the algorithm successfully pushed the raw node count near 20,000.

---

## Phases 4-5: Enforcing the Connectivity Constraint

A critical issue emerged when enforcing the competition's rule that the final subgraph must form a single weakly-connected component.
1. **The Disconnected Islands Problem:** Extracting the largest connected component from the 20,000-node result reduced it to merely ~100 nodes, revealing that the algorithm had been growing disconnected isomorphic "islands."
2. **Inside-Loop Constraint:** To fix this, I fundamentally redesigned the pipeline to enforce connectivity *inside* the growth loop at every single step, guaranteeing all added nodes touch the existing core.
3. **Hub Seeding:** Because the Hungarian seed was disconnected, I implemented a new "racing" strategy: identifying the top 2,000 highest-degree hub neurons across all three connectomes and racing them in parallel to find the densest, most connected starting core.

---

## Phases 6-7: Monte Carlo Tree Search and Advanced Search

The final breakthroughs pushed the fully-connected subgraph to its maximum size.
1. **The Lookahead Problem:** The limitation of greedy growth is that adding one valid neuron might permanently block dozens of better future additions due to edge constraints.
2. **Monte Carlo Tree Search (MCTS):** To solve this, I simulated 15 steps ahead (rollouts) before committing to any boundary addition. This evaluated the true "future value" of each candidate rather than just immediate validity.
3. **Genetic Crossover:** In parallel, a Genetic Algorithm with crossover was used to merge non-conflicting components from different randomized runs.
4. **Result:** The MCTS approach ultimately produced the cleanest, fully-connected 14,484-node result submitted.

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
# Output: network.csv
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

```
Flywire_Princeton/
├── README.md                        # Technical approach and reproduction guide
├── science.md                       # Biological significance report (1-page summary)
├── network.csv                      # Final answer: 14,484 verified matched neuron triplets
├── src/
│   ├── kaggle_nb4_mcts.py           # MCTS guided search - produced the final 14,484 result
│   ├── kaggle_nb5_genetic.py        # Genetic algorithm with crossover
│   ├── kaggle_nb6_spectral_faq.py   # Spectral FAQ continuous relaxation
│   ├── kaggle_nb2_highdeg.py        # High-degree seed racing
│   ├── kaggle_maximize_s3.py        # Season 3 degree-weighted smart perturbation
│   ├── kaggle_maximize_s2.py        # Season 2 refined perturbation fractions
│   ├── hungarian_bijection.py       # Cell-type bipartite matching seeder
│   └── verify_14484.py              # Final pairwise verification script (0 violations)
├── docs/
│   └── exploration_log.md           # Full chronological record of all strategies attempted
└── figures/
    └── README.md                    # Embedded figures and link to interactive visualization
```

**Interactive 3D Visualization:** [https://siddgud.github.io/14k_interactive_3d_network/14k_interactive_3d_network.html](https://siddgud.github.io/14k_interactive_3d_network/14k_interactive_3d_network.html)

<!-- network.csv: final 14484 triplet answer file -->
