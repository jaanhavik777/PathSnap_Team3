"""
Candidate state representation for HMM map matching in PathSnap.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class CandidateState:
    """Represents a candidate road segment match for a GPS observation."""
    arc_id: int
    distance_meters: float
    projection_fraction: float
    heading_diff_deg: float = 0.0
    segment_length_m: float = 0.0
    gps_speed_mps: float = 0.0
    proximity_score: float = 0.0
    u_node: Optional[int] = None
    v_node: Optional[int] = None
