"""
Trip discovery and management service for PathSnap API.
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, List, Tuple
from api.dependencies import DATA_ROOT, EVALUATION_PARTITION
from api.schemas import TripSummary


def list_available_trips(data_dir: Path = DATA_ROOT) -> Tuple[List[str], List[TripSummary]]:
    """
    Scan data directory dynamically and return list of valid trip IDs and summaries.
    Does not hardcode any trip IDs.
    """
    if not data_dir.exists():
        return [], []

    trip_ids_set = set()
    trip_summaries = []

    # 1. Direct children with {trip_id}.track
    for item in data_dir.iterdir():
        if item.is_dir():
            track_file = item / f"{item.name}.track"
            if track_file.exists():
                tid = item.name
                trip_ids_set.add(tid)
                has_route = (item / f"{tid}.route").exists()
                trip_summaries.append(TripSummary(trip_id=tid, has_reference=has_route))

    # 2. If nothing found in direct children, check nested subdirectories
    if not trip_ids_set:
        for track_path in data_dir.glob("*/*/*.track"):
            folder = track_path.parent
            tid = folder.name
            if tid not in trip_ids_set:
                trip_ids_set.add(tid)
                has_route = (folder / f"{tid}.route").exists()
                trip_summaries.append(TripSummary(trip_id=tid, has_reference=has_route))

    # 3. Fallback search across workspace
    if not trip_ids_set:
        parent_candidate = data_dir.parent.parent / "hack" / "Student folder" / "Student folder" / "student_test"
        if parent_candidate.exists():
            for item in parent_candidate.iterdir():
                if item.is_dir() and (item / f"{item.name}.track").exists():
                    tid = item.name
                    trip_ids_set.add(tid)
                    has_route = (item / f"{tid}.route").exists()
                    trip_summaries.append(TripSummary(trip_id=tid, has_reference=has_route))

    sorted_ids = sorted(trip_ids_set)
    trip_summaries.sort(key=lambda s: s.trip_id)
    return sorted_ids, trip_summaries
