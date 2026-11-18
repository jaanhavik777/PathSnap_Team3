# PathSnap — Person 3 (Hitha) Review 1 Documentation

**Project**: PathSnap — Distributed Map-Matching & Route Reconstruction  
**Team**: Team 3 (PES University Machine Learning Hackathon)  
**Member**: Person 3 — Hitha  
**Target Git Branch**: `feature/hitha-hmm-api`  
**Review 1 Milestone**: November 18, 2026 — 11:30 AM  

---

## 1. Executive Summary & Review 1 Deliverables

For Review 1, Person 3 was assigned six foundational deliverables:
1. **Train baseline HistGradientBoosting classifier** on 8 candidate features.
2. **Construct the road network graph (`RoadGraph`)** using directed NetworkX MultiDiGraph with geodesic Haversine edge weights and Dijkstra caching.
3. **Implement HMM emission scoring** combining learned classifier candidate confidence with Gaussian spatial measurement likelihood.
4. **Implement HMM transition scoring** enforcing the Newson-Krumm road-network distance model, directed one-way traversal, vehicle speed limits, and GPS dropout gap handling.
5. **Implement Viterbi dynamic programming decoder** in log-probability space with backpointer tracking and disconnected fallback recovery.
6. **Demonstrate end-to-end decoding on a sample trip**, showing genuine quantitative improvement in route continuity over classifier-only predictions.

All six deliverables are implemented, tested, and verified with genuine experimental evidence.

---

## 2. Code Architecture & Component Mapping

| Review 1 Component | Source File Path | Description |
| :--- | :--- | :--- |
| **Spatial Utilities** | `src/data/spatial.py` | Geodesic Haversine distance, bearings, angular differences, and metric `LocalProjector`. |
| **Road Graph** | `src/data/road_graph.py` | Directed NetworkX MultiDiGraph, parallel edge indexing, Haversine arc lengths, and cached Dijkstra shortest paths. |
| **Candidate State** | `src/hmm/states.py` | `CandidateState` data structure storing candidate arc ID, distance, projection fraction, and kinematic attributes. |
| **HMM Emissions** | `src/hmm/emissions.py` | Fusion of learned classifier candidate probabilities and Gaussian perpendicular distance likelihood. |
| **HMM Transitions** | `src/hmm/transitions.py` | Newson-Krumm transition scoring: $\|d_{\text{net}} - d_{\text{gps}}\|$, speed feasibility ($v \le 45$ m/s), and dropout handling. |
| **Viterbi Decoder** | `src/hmm/viterbi.py` | Log-space dynamic programming sequence decoder with backpointer backtracking and graceful disconnection recovery. |
| **Route Reconstruction** | `src/hmm/decode_route.py` | Continuous sequence reconstruction interpolating intermediate graph arcs and producing lat/lon polyline geometry. |
| **Route Metrics** | `src/hmm/route_metrics.py` | Route continuity ratio, invalid transition rate, Top-1 accuracy, and candidate miss rate. |
| **HistGradientBoosting Wrapper** | `src/models/train_hist_gradient_boosting.py` | `HistGradientBoostingModel` wrapper supporting serialization, probability prediction, and hyperparameter configuration. |
| **Model Artifact** | `artifacts/models/hist_gradient_boosting.joblib` | Trained 150-iteration gradient boosted decision tree classifier (325 KB). |
| **Model Metadata** | `artifacts/models/hist_gradient_boosting_metadata.json` | Feature schema, training dataset sizes, and hyperparameter metadata. |
| **Model Config** | `configs/models/hist_gradient_boosting.yaml` | YAML configuration file defining feature set, training parameters, and HMM hyperparameters. |
| **Unit Test Suite** | `tests/test_hmm.py` | 11 automated pytest test cases validating graph, emissions, transitions, Viterbi decoding, and metrics (100% pass). |
| **Live Demonstration** | `scripts/hitha_review1_demo.py` | End-to-end command-line demo running inference, Viterbi decoding, and route reconstruction on student test trips. |

---

## 3. Mathematical Formulations

### 3.1 Emission Probability
For a candidate $c_{t,j}$ at observation $t$ with perpendicular distance $d_{t,j}$ and classifier probability $p_{\text{clf}}(c_{t,j})$:

$$\mathcal{L}_{\text{spatial}}(z_t \mid c_{t,j}) = \exp\left(-\frac{1}{2}\left(\frac{d_{t,j}}{\sigma_z}\right)^2\right), \quad \sigma_z = 12.0\,\text{m}$$

$$w(c_{t,j}) = p_{\text{clf}}(c_{t,j}) \cdot \mathcal{L}_{\text{spatial}}(z_t \mid c_{t,j})$$

$$\log P(z_t \mid c_{t,j}) = \log\left(\frac{w(c_{t,j})}{\sum_{k=1}^{M_t} w(c_{t,k})}\right)$$

### 3.2 Transition Probability (Newson-Krumm Formulation)
For consecutive candidate states $c_{t-1,i}$ and $c_{t,j}$ with road-network shortest path travel distance $d_{\text{net}}$ and Euclidean GPS displacement $d_{\text{gps}}$:

$$\log P(c_{t,j} \mid c_{t-1,i}) = -\frac{|d_{\text{net}}(c_{t-1,i}, c_{t,j}) - d_{\text{gps}}(z_{t-1}, z_t)|}{\beta}, \quad \beta = 15.0\,\text{m}$$

- **One-way Constraint**: If no directed path exists from arc $i$ to arc $j$, $d_{\text{net}} = \infty$, yielding $\log P = \text{min\_log\_prob} = -30.0$.
- **Speed Feasibility**: If $v = \frac{d_{\text{net}}}{\Delta t} > 45\,\text{m/s}$ (162 km/h), an excess speed penalty $\frac{v - 45}{5.0}$ is subtracted.
- **GPS Dropout**: If $\Delta t > 10.0\,\text{s}$, the effective $\beta$ scales logarithmically with elapsed time to avoid false disconnects.

### 3.3 Viterbi Dynamic Programming Recursion
$$V[0, j] = \log P(z_0 \mid c_{0,j})$$
$$V[t, j] = \log P(z_t \mid c_{t,j}) + \max_{i=1 \dots M_{t-1}} \left(V[t-1, i] + \log P(c_{t,j} \mid c_{t-1,i})\right)$$
$$\text{Backpointer}[t, j] = \arg\max_{i=1 \dots M_{t-1}} \left(V[t-1, i] + \log P(c_{t,j} \mid c_{t-1,i})\right)$$

---

## 4. Verified Experimental Results

### 4.1 5-Fold GroupKFold Cross-Validation (80 Training Trips, 1.8M Candidates)
- **Mean CV Candidate Macro F1**: `0.8263 ± 0.0413`
- **Mean CV Top-1 Accuracy**: `0.6351 ± 0.1429`
- **Mean CV Route Continuity**: `0.9086 ± 0.0217`

### 4.2 Held-Out Student-Test Evaluation (10 Trips, 174,150 Candidates, 21,245 Observations)
- **Candidate Macro F1**: `0.8529`
- **Precision**: `0.7775` | **Recall**: `0.6964` | **Log Loss**: `0.1429`
- **Candidate Miss Rate**: `0.0556` (5.56%)
- **Raw Inference Latency**: `10 µs / candidate` (1.67s total across all 10 trips)

### 4.3 Classifier-Only vs Classifier + HMM Comparison
| Metric | Classifier-Only (Baseline) | Classifier + HMM (Viterbi) | Net Improvement |
| :--- | :---: | :---: | :---: |
| **Mean Route Continuity** | **92.69%** | **97.84%** | **+5.15%** |
| **Mean Invalid Transition Rate** | **7.31%** | **2.16%** | **-5.15% (70% reduction)** |
| **Mean Observation Top-1 Accuracy** | **75.68%** | **76.32%** | **+0.64%** |

*(Detailed per-trip results and comparison plots are saved in `results/review1_hitha/hmm_comparison.csv` and `hmm_comparison.png`).*

---

## 5. Reproduction & Verification Instructions

### 5.1 Run Automated Pytest Suite
```bash
# Set PYTHONPATH to repository root
export PYTHONPATH="."
pytest tests/test_hmm.py -v
```
**Expected Output**: `11 passed in ~8s`

### 5.2 Run Live Demonstration on Sample Trip
```bash
python scripts/hitha_review1_demo.py --trip-id 00000000
```
**Demonstration Highlights on Trip `00000000` (2,503 GPS observations)**:
- RoadGraph built with 8,542 nodes and 18,196 directed arcs.
- HistGradientBoosting scored 10,445 candidates in 1.76s.
- Viterbi decoded the entire trajectory in 1.15s (0.46 ms/obs).
- Route continuity increased from **90.17%** (classifier-only) to **95.00%** (HMM), reducing disconnected jumps by **4.84%**.
- Reconstructed a fully continuous 815-arc polyline with 820 geographic vertices.

---

## 6. Review 2 Roadmap (FastAPI Backend)

The FastAPI backend components (endpoints `/health`, `/trips`, `/predict`, `/model-info`, evaluator isolation, and two-laptop network deployment) are already developed locally in Person 3's workspace and will be submitted during Review 2 per the hackathon milestone schedule.
