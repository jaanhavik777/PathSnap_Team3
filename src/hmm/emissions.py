"""
HMM emission scoring for PathSnap map matching.

Combines:
  1. Learned classifier candidate confidence score: P_clf(cand | features)
  2. Geographic measurement likelihood: Gaussian distribution over perpendicular distance
     P_spatial(z_t | cand) ~ (1 / (sqrt(2*pi)*sigma_z)) * exp(-dist^2 / (2*sigma_z^2))
"""

from typing import Any, List, Union
import numpy as np


def calculate_emission_log_probs(
    candidates: List[Any],
    classifier_scores: Union[np.ndarray, List[float]],
    sigma_z_meters: float = 12.0,
    min_log_prob: float = -30.0,
) -> np.ndarray:
    """
    Calculate normalized log emission probabilities for candidates at observation t.

    Args:
        candidates: List of candidate states (objects with .distance_meters or dicts).
        classifier_scores: Array or list of class-1 classifier probabilities for candidates.
        sigma_z_meters: Gaussian standard deviation for GPS measurement error in meters.
        min_log_prob: Numerical floor for log probability values.

    Returns:
        log_emissions: 1D np.ndarray of shape (len(candidates),) in log-space.
    """
    M = len(candidates)
    if M == 0:
        return np.array([], dtype=np.float64)

    scores_arr = np.asarray(classifier_scores, dtype=np.float64)
    if len(scores_arr) != M:
        scores_arr = np.full(M, 1.0 / M, dtype=np.float64)

    # Extract perpendicular distance in meters for each candidate
    dists = np.zeros(M, dtype=np.float64)
    for i, c in enumerate(candidates):
        if hasattr(c, "distance_meters"):
            dists[i] = float(c.distance_meters)
        elif isinstance(c, dict) and "distance_meters" in c:
            dists[i] = float(c["distance_meters"])
        else:
            dists[i] = 10.0  # reasonable default

    # 1. Spatial Gaussian likelihood
    spatial_likelihood = np.exp(-0.5 * (dists / max(1.0, sigma_z_meters)) ** 2)

    # 2. Classifier confidence weight clipped away from zero
    clf_weights = np.clip(scores_arr, 1e-6, 1.0)

    # 3. Composite unnormalized likelihood
    weights = clf_weights * spatial_likelihood
    total_weight = float(np.sum(weights))

    if total_weight < 1e-12:
        # Fallback to normalized inverse distance
        weights = 1.0 / (dists + 1.0)
        total_weight = float(np.sum(weights))

    normalized_probs = weights / total_weight
    log_emissions = np.log(np.clip(normalized_probs, 1e-15, 1.0))
    return np.maximum(log_emissions, float(min_log_prob))
