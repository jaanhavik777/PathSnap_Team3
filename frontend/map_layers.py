"""Folium renderers for geometry returned by FastAPI."""

from __future__ import annotations

import math
from typing import Any

import folium


def _point(value: Any) -> list[float] | None:
    if isinstance(value, dict):
        lat, lon = value.get("lat"), value.get("lon")
    elif isinstance(value, (list, tuple)) and len(value) >= 2:
        lat, lon = value[0], value[1]
    else:
        return None
    try:
        lat, lon = float(lat), float(lon)
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(lat) and math.isfinite(lon)):
        return None
    return [lat, lon] if -90 <= lat <= 90 and -180 <= lon <= 180 else None


def _segment(value: Any) -> list[list[float]]:
    """Read a segment object or GeoJSON feature into Folium [lat, lon] points."""
    segment_id = None
    if isinstance(value, dict):
        segment_id = value.get("arc_id", value.get("id"))
        geometry = value.get("geometry")
        coordinates = value.get("coordinates")
        if coordinates is None and isinstance(geometry, dict):
            coordinates = geometry.get("coordinates")
    else:
        coordinates = value

    if isinstance(coordinates, dict):
        coordinates = coordinates.get("coordinates")
    # A backend can return a GeoJSON Feature or FeatureCollection.
    if isinstance(coordinates, dict) and coordinates.get("type") == "Feature":
        coordinates = coordinates.get("geometry", {}).get("coordinates", [])
    if not isinstance(coordinates, (list, tuple)):
        return []
    # Allow a MultiLineString as well as the more common LineString.
    if coordinates and isinstance(coordinates[0], (list, tuple)) and coordinates[0]:
        if isinstance(coordinates[0][0], (list, tuple)):
            coordinates = [point for line in coordinates for point in line]

    points = []
    for coordinate in coordinates:
        if not isinstance(coordinate, (list, tuple)) or len(coordinate) < 2:
            continue
        point = _point([coordinate[1], coordinate[0]])
        if point:
            points.append(point)
    return points


def build_prediction_map(data: dict[str, Any]) -> folium.Map:
    """Show API-returned GPS, context roads and prediction as toggleable layers."""
    gps_points = [point for point in (_point(p) for p in data.get("gps_points", [])) if point]
    roads = [line for line in (_segment(s) for s in data.get("road_segments", [])) if len(line) > 1]
    route = [line for line in (_segment(s) for s in data.get("predicted_route", [])) if len(line) > 1]
    bounds = gps_points + [point for line in roads + route for point in line]
    if not bounds:
        raise ValueError("API response contains no usable map geometry")

    center = [sum(p[0] for p in bounds) / len(bounds), sum(p[1] for p in bounds) / len(bounds)]
    map_view = folium.Map(center, zoom_start=14, control_scale=True, prefer_canvas=True)

    road_layer = folium.FeatureGroup(name="Relevant road segments", show=True)
    for segment in data.get("road_segments", []):
        line = _segment(segment)
        if len(line) > 1:
            arc_id = segment.get("arc_id", "") if isinstance(segment, dict) else ""
            label = f"Road arc {arc_id}" if arc_id != "" else "Road segment"
            folium.PolyLine(line, color="#9aa4b2", weight=3, tooltip=label).add_to(road_layer)
    road_layer.add_to(map_view)

    gps_layer = folium.FeatureGroup(name="Original GPS observations", show=True)
    for index, point in enumerate(gps_points):
        folium.CircleMarker(
            point, radius=4, color="#2677d8", fill=True, fill_opacity=0.85,
            tooltip=f"GPS observation {index + 1}",
        ).add_to(gps_layer)
    gps_layer.add_to(map_view)

    prediction_layer = folium.FeatureGroup(name="Predicted road route", show=True)
    for segment in data.get("predicted_route", []):
        line = _segment(segment)
        if len(line) > 1:
            arc_id = segment.get("arc_id", "") if isinstance(segment, dict) else ""
            label = f"Predicted arc {arc_id}" if arc_id != "" else "Predicted segment"
            folium.PolyLine(line, color="#159447", weight=6, tooltip=label).add_to(prediction_layer)
    prediction_layer.add_to(map_view)

    if gps_points:
        folium.Marker(gps_points[0], tooltip="GPS start", icon=folium.Icon(color="green")).add_to(map_view)
        folium.Marker(gps_points[-1], tooltip="GPS end", icon=folium.Icon(color="red")).add_to(map_view)
    map_view.fit_bounds(bounds)
    folium.LayerControl(collapsed=False).add_to(map_view)
    return map_view
