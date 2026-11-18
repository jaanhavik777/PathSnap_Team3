"""
Viterbi dynamic programming decoding algorithm for PathSnap.

Decodes the most likely sequence of candidate road segments in log-probability space:
  V[t, j] = logEmission[t, j] + max_i (V[t-1, i] + logTransition[i, j])

Supports:
  - Variable candidate counts per observation
  - Numerically stable log-sum calculations
  - Disconnected transition fallback without breaking sequence recovery
  - Backpointer backtracking
  - Warning generation for gaps and broken transitions
"""

from typing import Any, List, Optional, Tuple, Union
import numpy as np

try:
    from src.data.road_graph import RoadGraph
    from src.data.spatial import haversine_distance
    from src.hmm.emissions import calculate_emission_log_probs
    from src.hmm.transitions import calculate_transition_log_prob
except ImportError:
    from ..data.road_graph import RoadGraph
    from ..data.spatial import haversine_distance
    from .emissions import calculate_emission_log_probs
    from .transitions import calculate_transition_log_prob


def _get_arc_id(cand: Any) -> int:
    if hasattr(cand, "arc_id"):
        return int(cand.arc_id)
    if isinstance(cand, dict) and "arc_id" in cand:
        return int(cand["arc_id"])
    return int(cand)


class ViterbiDecoder:
    """
    Decodes the optimal candidate road segment sequence using the Viterbi algorithm.
    """

    def __init__(
        self,
        road_graph: RoadGraph,
        beta_meters: float = 15.0,
        sigma_z_meters: float = 12.0,
        max_speed_mps: float = 45.0,
        gap_threshold_sec: float = 10.0,
        min_log_prob: float = -30.0,
    ):
        self.road_graph = road_graph
        self.beta = float(beta_meters)
        self.sigma_z = float(sigma_z_meters)
        self.max_speed_mps = float(max_speed_mps)
        self.gap_threshold_sec = float(gap_threshold_sec)
        self.min_log_prob = float(min_log_prob)

    def decode(
        self,
        observation_candidates: List[List[Any]],
        classifier_scores: List[Union[np.ndarray, List[float]]],
        lons: np.ndarray,
        lats: np.ndarray,
        timestamps: np.ndarray,
    ) -> Tuple[List[Any], List[int], float, List[str]]:
        """
        Run Viterbi decoding over a trip trajectory.

        Args:
            observation_candidates: List of length N, each containing Candidate objects.
            classifier_scores: List of length N, each containing candidate scores.
            lons: GPS longitudes, shape (N,)
            lats: GPS latitudes, shape (N,)
            timestamps: GPS timestamps in seconds, shape (N,)

        Returns:
            (selected_candidates, observation_arc_ids, final_log_score, warnings)
        """
        N = len(observation_candidates)
        warnings: List[str] = []

        if N == 0:
            return [], [], 0.0, ["Empty trajectory provided to decoder."]

        # Handle trivial case of 1 observation
        if N == 1:
            cands = observation_candidates[0]
            if len(cands) == 0:
                return [], [-1], 0.0, ["No candidates for single observation."]
            scores = classifier_scores[0]
            log_em = calculate_emission_log_probs(
                cands, scores, sigma_z_meters=self.sigma_z, min_log_prob=self.min_log_prob
            )
            best_idx = int(np.argmax(log_em))
            cand = cands[best_idx]
            arc_id = _get_arc_id(cand)
            return [cand], [arc_id], float(log_em[best_idx]), []

        # DP tables:
        V: List[np.ndarray] = []
        backpointers: List[np.ndarray] = []

        # 1. Initialization at t = 0
        cands_0 = observation_candidates[0]
        if len(cands_0) == 0:
            warnings.append("Observation 0 has no candidates.")
            V.append(np.array([self.min_log_prob]))
            backpointers.append(np.array([-1]))
        else:
            log_em_0 = calculate_emission_log_probs(
                cands_0, classifier_scores[0], sigma_z_meters=self.sigma_z, min_log_prob=self.min_log_prob
            )
            V.append(log_em_0)
            backpointers.append(np.full(len(cands_0), -1, dtype=int))

        # 2. Forward recursion for t = 1 to N-1
        for t in range(1, N):
            curr_cands = observation_candidates[t]
            prev_cands = observation_candidates[t - 1]
            M_curr = len(curr_cands)
            M_prev = len(prev_cands)

            if M_curr == 0:
                warnings.append(f"Observation {t} has no candidates.")
                V.append(np.array([self.min_log_prob]))
                backpointers.append(np.array([0]))
                continue

            curr_log_em = calculate_emission_log_probs(
                curr_cands, classifier_scores[t], sigma_z_meters=self.sigma_z, min_log_prob=self.min_log_prob
            )

            # GPS displacement and elapsed time
            gps_dist = haversine_distance(
                float(lons[t - 1]), float(lats[t - 1]), float(lons[t]), float(lats[t])
            )
            dt_sec = max(0.1, float(timestamps[t] - timestamps[t - 1]))

            if dt_sec > self.gap_threshold_sec:
                warnings.append(
                    f"Observation {t}: Large GPS gap of {dt_sec:.1f}s detected."
                )

            curr_V = np.full(M_curr, -np.inf)
            curr_bp = np.zeros(M_curr, dtype=int)
            prev_V = V[t - 1]

            # Compute transition from each predecessor i to successor j
            for j in range(M_curr):
                best_score = -np.inf
                best_pred = 0

                for i in range(M_prev):
                    log_trans = calculate_transition_log_prob(
                        self.road_graph,
                        prev_cands[i],
                        curr_cands[j],
                        gps_dist,
                        dt_sec,
                        beta_meters=self.beta,
                        max_speed_mps=self.max_speed_mps,
                        gap_threshold_sec=self.gap_threshold_sec,
                        min_log_prob=self.min_log_prob,
                    )
                    score = prev_V[i] + log_trans
                    if score > best_score:
                        best_score = score
                        best_pred = i

                # Add emission score
                curr_V[j] = best_score + curr_log_em[j]
                curr_bp[j] = best_pred

            # Check for total disconnection (all incoming transitions hit floor)
            if np.all(curr_V <= self.min_log_prob + 1.0):
                warnings.append(
                    f"Observation {t}: Disconnected transition detected. Resetting chain to emission."
                )
                curr_V = curr_log_em
                curr_bp = np.zeros(M_curr, dtype=int)

            V.append(curr_V)
            backpointers.append(curr_bp)

        # 3. Backtracking
        last_V = V[-1]
        best_last_idx = int(np.argmax(last_V))
        final_log_score = float(last_V[best_last_idx])

        decoded_indices = [best_last_idx]
        for t in range(N - 1, 0, -1):
            pred_idx = int(backpointers[t][decoded_indices[-1]])
            decoded_indices.append(pred_idx)

        decoded_indices.reverse()

        # 4. Assemble resolved candidates and arc IDs
        selected_candidates: List[Any] = []
        observation_arc_ids: List[int] = []

        for t in range(N):
            cands = observation_candidates[t]
            idx = decoded_indices[t]
            if 0 <= idx < len(cands):
                cand = cands[idx]
                selected_candidates.append(cand)
                observation_arc_ids.append(_get_arc_id(cand))
            elif len(cands) > 0:
                cand = cands[0]
                selected_candidates.append(cand)
                observation_arc_ids.append(_get_arc_id(cand))
            else:
                selected_candidates.append(None)
                observation_arc_ids.append(-1)

        return selected_candidates, observation_arc_ids, final_log_score, warnings


def viterbi_decode(
    road_graph: RoadGraph,
    observation_candidates: List[List[Any]],
    classifier_scores: List[Union[np.ndarray, List[float]]],
    lons: np.ndarray,
    lats: np.ndarray,
    timestamps: np.ndarray,
    beta_meters: float = 15.0,
    sigma_z_meters: float = 12.0,
    max_speed_mps: float = 45.0,
    gap_threshold_sec: float = 10.0,
    min_log_prob: float = -30.0,
) -> Tuple[List[Any], List[int], float, List[str]]:
    """Functional convenience wrapper for Viterbi decoding."""
    decoder = ViterbiDecoder(
        road_graph=road_graph,
        beta_meters=beta_meters,
        sigma_z_meters=sigma_z_meters,
        max_speed_mps=max_speed_mps,
        gap_threshold_sec=gap_threshold_sec,
        min_log_prob=min_log_prob,
    )
    return decoder.decode(observation_candidates, classifier_scores, lons, lats, timestamps)
