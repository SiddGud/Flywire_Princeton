# Structural Homology and Functional Invariance: A 14,484-Neuron Conserved Visual Circuit Across Three *Drosophila* Connectomes

**Abstract:** Using a multi-stage heuristic pipeline combining Monte Carlo Tree Search, Simulated Annealing, aggressive network pruning, and Genetic Algorithms seeded by cell-type bipartite matching, I identified a weakly connected directed subgraph of **14,484 strictly homologous neurons** - verified with zero edge violations - shared identically across the FAFB, BANC, and MCNS connectomes. The result points to a sex-invariant visual-to-navigation backbone that achieves behavioral plasticity not by rewiring itself, but by letting specialized modulatory neurons hijack its output.

---

## Results

The subgraph I found spans **10.4% of the FAFB connectome**, containing 14,652 directed edges and 4,062,696 synapses. When I ran the verification script across all three 300 MB edge lists (which took 397 seconds to complete), it confirmed **zero isomorphism violations** and a single weakly connected component - no isolated clusters, every node reachable from every other.

One thing I noticed immediately was how *sparse* the network is - an average degree of only ~2.03 per node. For 14,484 neurons I initially expected something much denser. But looking at the structure, it makes sense: the optic lobe is ~800 parallel retinotopic columns, each independently processing one point in the visual field like pixels in a camera, not an all-to-all web.

The circuit traces the complete **Elementary Motion Detection (EMD) pathway**: R1-6 photoreceptors (*N*=340) → lamina ON/OFF separation (L1/L2/L3, *N*=692) → medullary delay-and-correlate (Mi1, Mi4, Mi9; Tm9 *N*=415, which I found to be the single most common cell type in the entire subgraph) → direction-selective T4a-d (*N*=669) and T5a-d (*N*=827) neurons → Central Complex navigation neurons (vDelta, *N*=23). When I ran an Independent Cascade Model seeded at the R1-6 layer, I found the Fan-Shaped Body activated at **96%** frequency and the Ellipsoid Body at **88%** - which told me this visual circuit is directly driving spatial navigation, not just feeding signals upward abstractly.

As an independent check, I cross-referenced against the Janelia Hemibrain connectome (which I didn't use in the matching). It recovered 2,169 confirmed homologs - 15% of the subgraph validated in a fourth dataset I never touched.

The neurotransmitter breakdown I pulled from Codex showed **67% cholinergic** (excitatory), **19% glutamatergic**, and **13% GABAergic** - a ~2:1 E/I ratio that matches the theoretical optimum for null-direction suppression [1]. Topologically, I found that 97.4% of neurons participate in feedforward loops and 71.9% in reciprocal connections [5], both signatures of a circuit built for sustained, noise-resistant signal integration.

<table width="100%">
<tr>
<th width="50%">Figure 1: Codex 3D Mesh View</th>
<th width="50%">Figure 2: 14k Connectivity Graph</th>
</tr>
<tr>
<td valign="top">
<img width="100%" src="https://gist.github.com/user-attachments/assets/407086e0-186c-4dcd-ac14-4c3cc2e1e00f" />
<br><br>
Neuroglancer rendering of the visual circuit, <b>exactly matching with</b> the identified 14,484 strictly homologous neurons. The layout highlights the crystalline T4/T5 columnar arrays and massive wide-field integrators.
</td>
<td valign="top">
<img width="100%" src="https://gist.github.com/user-attachments/assets/0ab73fc3-cac1-497c-8a68-62260dc497f4" />
<br><br>
Force-directed layout of the complete 14,484-node FAFB induced subgraph. Since 14,484 neurons are densely clustered together, it is difficult to discern individual pathways in a static image. Therefore, I have provided an interactive 3D HTML visualization. <b><a href="https://siddgud.github.io/14k_interactive_3d_network/14k_interactive_3d_network.html">Click here to view the live interactive 3D HTML visualization</a></b> for detailed insights, interactive exploration, and synapse edge analysis.
</td>
</tr>
</table>

---

## Observations and Biological Hypotheses

**1 - Hardwired Backbone:** The finding that 14,484 neurons are wired identically across three independently imaged brains strongly suggests a genetically fixed architecture. Molecular guidance cues like N-cadherin and Dscam deterministically route axons during development [2], and what I found here suggests that determinism scales to at least 10% of the whole brain - the circuit is essentially printed from a template, set before the fly ecloses.

**2 - Sensory Hijacking (the observation I found most compelling):** I found that a male and female fly share an *exactly identical* visual circuit. This raised an obvious question: how does a male see a female and choose to court her, while a female doesn't? Instead of building sex-specific visual hardware - costly and redundant - evolution appears to solve this by adding Fru+/Dsx+ modulatory neurons that inject into the LC10a hub in the AVLP and redirect the invariant motion signal toward sex-specific motor programs. Crucially, when I analyzed the location of the ~89,046 sexually dimorphic synapses, I found they sit *adjacent to* the isomorphic backbone, not inside it. The core visual computation is unchanged; only the downstream routing is sexually reassigned. I think this is an elegant evolutionary strategy.

**3 - Visual-Navigation Pipeline:** Tracing connectivity from R1-6 → T4/T5 → VS/HS → vDelta neurons in the Central Complex, I found an unbroken structural pathway from photon capture to heading computation. The cascade simulation confirmed this: a signal seeded at photoreceptors reliably reaches navigation centers 96% of the time. A fly's ability to fly straight and navigate by the sun appears to be a direct output of this one conserved circuit.

**4 - Multi-Modal Convergence:** Two things I found genuinely unexpected: the Gnathal Ganglion (jaw-motor region) receives 77,832 visual input synapses from this subgraph - the 5th highest of any region - and the Lateral Horn (innate olfactory center) receives 33,467. I had not expected the visual backbone to be structurally wired into feeding and olfactory circuits at this scale. It suggests the brain integrates sensory modalities much earlier in processing than commonly assumed.

*Limitation: the multi-stage heuristic pipeline (MCTS, Simulated Annealing, Genetic Algorithms) I used cannot guarantee a globally optimal solution - a larger isomorphic subgraph may exist. Exact solvers applied to smaller seed regions remain a direction worth exploring.*

---

## References
1. Borst & Euler (2011). *Neuron*, 71, 974-994.
2. Clandinin & Zipursky (2002). *Neuron*, 35, 827-841.
3. Dorkenwald et al. (2024). *Nature*, 634, 124-138.
4. Maisak et al. (2013). *Nature*, 500, 212-216.
5. Milo et al. (2002). *Science*, 298, 824-827.
6. Matsliah, Yu et al. (2024). *Nature*, 634, 166-180.
7. Scheffer et al. (2020). *eLife*, 9, e57443.
8. Schlegel et al. (2024). *Nature*, 634, 139-152.
9. Shinomiya et al. (2019). *eLife*, 8, e42344.
