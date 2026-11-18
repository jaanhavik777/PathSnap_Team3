"""
Geospatial calculations and metric projection utilities for PathSnap.

Never treats angular latitude/longitude degrees as Euclidean meters.
Provides:
  - Haversine geodesic distance in meters
  - Local equidistant metric projection (centered per trip)
  - Point-to-segment projection, perpendicular distance, and projection fraction
  - Compass bearing and heading difference in degrees [0, 180]
"""

from typing import Tuple
import numpy as np

EARTH_RADIUS_METERS = 6371000.0


def haversine_distance(
    lon1: float, lat1: float, lon2: float, lat2: float
) -> float:
    """Calculate great circle distance between two points in meters using Haversine formula."""
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)

    a = (
        np.sin(dphi / 2.0) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    )
    a = np.clip(a, 0.0, 1.0)
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return float(EARTH_RADIUS_METERS * c)


def haversine_distance_vectorized(
    lons1: np.ndarray,
    lats1: np.ndarray,
    lons2: np.ndarray,
    lats2: np.ndarray,
) -> np.ndarray:
    """Vectorized Haversine distance in meters between coordinate arrays."""
    phi1 = np.radians(lats1)
    phi2 = np.radians(lats2)
    dphi = np.radians(lats2 - lats1)
    dlambda = np.radians(lons2 - lons1)

    a = (
        np.sin(dphi / 2.0) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    )
    a = np.clip(a, 0.0, 1.0)
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return EARTH_RADIUS_METERS * c


def calculate_bearing(
    lon1: float, lat1: float, lon2: float, lat2: float
) -> float:
    """Calculate initial compass bearing from (lon1, lat1) to (lon2, lat2) in degrees [0, 360)."""
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dlambda = np.radians(lon2 - lon1)

    y = np.sin(dlambda) * np.cos(phi2)
    x = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(dlambda)
    bearing = np.degrees(np.arctan2(y, x))
    return float((bearing + 360.0) % 360.0)


def angle_difference(angle1: float, angle2: float) -> float:
    """Compute minimal angular difference between two compass bearings in degrees [0, 180]."""
    diff = abs(angle1 - angle2) % 360.0
    return float(360.0 - diff if diff > 180.0 else diff)


class LocalProjector:
    """
    Local metric projection centered at (lon_center, lat_center) of a trip.
    Converts spherical coordinates to 2D Cartesian plane (x_meters, y_meters).
    """

    def __init__(self, lon_center: float, lat_center: float):
        self.lon_center = float(lon_center)
        self.lat_center = float(lat_center)
        self.cos_lat = float(np.cos(np.radians(lat_center)))
        self.m_per_deg_lat = float(EARTH_RADIUS_METERS * (np.pi / 180.0))
        self.m_per_deg_lon = float(self.m_per_deg_lat * self.cos_lat)

    def forward(self, lon: float, lat: float) -> Tuple[float, float]:
        """Convert (lon, lat) to (x_meters, y_meters) relative to center."""
        x = (lon - self.lon_center) * self.m_per_deg_lon
        y = (lat - self.lat_center) * self.m_per_deg_lat
        return float(x), float(y)

    def forward_batch(
        self, lons: np.ndarray, lats: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Vectorized conversion to (x, y) meters."""
        x = (lons - self.lon_center) * self.m_per_deg_lon
        y = (lats - self.lat_center) * self.m_per_deg_lat
        return x, y

    def inverse(self, x: float, y: float) -> Tuple[float, float]:
        """Convert (x_meters, y_meters) back to (lon, lat)."""
        lon = self.lon_center + (x / self.m_per_deg_lon)
        lat = self.lat_center + (y / self.m_per_deg_lat)
        return float(lon), float(lat)


def project_point_to_segment_2d(
    px: float, py: float, ax: float, ay: float, bx: float, by: float
) -> Tuple[float, float, float, float]:
    """
    Projects a 2D point (px, py) onto line segment AB (from (ax, ay) to (bx, by)).
    All coordinates in meters.

    Returns:
      (proj_x, proj_y, distance_meters, fraction)
      fraction in [0.0, 1.0] indicates position along directed segment from A to B.
    """
    dx = bx - ax
    dy = by - ay
    seg_len_sq = dx * dx + dy * dy

    if seg_len_sq < 1e-12:
        dist = np.hypot(px - ax, py - ay)
        return ax, ay, float(dist), 0.0

    t = ((px - ax) * dx + (py - ay) * dy) / seg_len_sq
    t_clamped = max(0.0, min(1.0, float(t)))

    proj_x = ax + t_clamped * dx
    proj_y = ay + t_clamped * dy
    dist = np.hypot(px - proj_x, py - proj_y)

    return float(proj_x), float(proj_y), float(dist), float(t_clamped)
