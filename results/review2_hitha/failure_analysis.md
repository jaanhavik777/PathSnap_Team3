# HMM Map-Matching Failure Case Analysis & Trade-offs (Review 2)

## 1. Verified Results Across 10 Student-Test Trips

| Metric | Classifier-Only (Greedy Argmax) | Classifier + HMM (Viterbi Global Decoding) | Net Improvement |
| :--- | :---: | :---: | :---: |
| **Route Continuity** | **92.69%** | **97.84%** | **+5.15%** |
| **Invalid Transition Rate** | **7.31%** | **2.16%** | **-5.15% (70.5% reduction)** |
| **Observation Top-1 Accuracy** | **75.68%** | **76.32%** | **+0.64%** |
| **Candidate Miss Rate** | **5.23%** | **5.23%** | **0.00%** |

---

## 2. The Pointwise Accuracy vs Topological Continuity Trade-off

A crucial algorithmic property discovered during evaluation is why HMM occasionally exhibits minor pointwise accuracy regressions on specific trips (e.g. Trip `00000007` -4.2%, Trip `00000000` -0.4%, Trip `00000065` -0.7%, Trip `00000091` -0.5%):

1. **Greedy Pointwise Selection vs Graph Realism**:
   - The standalone classifier treats each GPS ping independently ($P(c_t \mid z_t)$). If a vehicle is driving along an urban arterial and a noisy GPS ping drifts 15 meters to the right towards a parallel service road or alleyway, the classifier may assign slightly higher confidence to the parallel segment due to proximity.
   - However, in reality the vehicle never teleported laterally onto the parallel service road. Selecting the parallel arc creates a physically impossible disconnected jump (violating road graph connectivity).
2. **HMM Transition Penalty ($d_{\text{net}}$ vs $d_{\text{gps}}$)**:
   - The HMM transition term $\log P(c_t \mid c_{t-1}) \propto -|d_{\text{net}} - d_{\text{gps}}|/\beta$ heavily penalizes transitions between disconnected or unreachable road segments.
   - When the parallel alley is disconnected from the mainline road, the directed network travel distance $d_{\text{net}} \rightarrow \infty$, triggering the floor penalty ($-30.0$).
   - Viterbi decoding therefore forces the path to remain on the mainline arterial road, successfully preserving **97.84% route continuity** and eliminating zigzagging.
3. **Where Regressions Occur**:
   - On trips where vehicle motion includes sharp turns, complex elevated interchanges, or stop-and-go maneuvers near intersections with high GPS scatter, Viterbi's sequence inertia can occasionally commit to an adjacent connected branch for 2–3 consecutive observations before correcting back.
   - This produces a localized minor pointwise accuracy dip in exchange for a complete, traversable, physically valid vehicle trajectory.
