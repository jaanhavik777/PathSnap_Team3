"""
PathSnap HMM map matching and route reconstruction package.
"""

from .states import CandidateState
from .emissions import calculate_emission_log_probs
from .transitions import calculate_transition_log_prob
from .viterbi import ViterbiDecoder, viterbi_decode
from .decode_route import ReconstructedRoute, reconstruct_continuous_route
from .route_metrics import (
    calculate_route_continuity,
    calculate_invalid_transition_rate,
    calculate_top1_accuracy,
    calculate_candidate_miss_rate,
)

__all__ = [
    "CandidateState",
    "calculate_emission_log_probs",
    "calculate_transition_log_prob",
    "ViterbiDecoder",
    "viterbi_decode",
    "ReconstructedRoute",
    "reconstruct_continuous_route",
    "calculate_route_continuity",
    "calculate_invalid_transition_rate",
    "calculate_top1_accuracy",
    "calculate_candidate_miss_rate",
]
