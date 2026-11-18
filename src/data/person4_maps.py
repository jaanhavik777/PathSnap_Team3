"""Folium development map from the team's parsed trip representation."""
from __future__ import annotations

import math

import folium
import numpy as np


def _arc_coords(trip, arc_id):
    arcs = trip["arcs"]
    row = arcs.loc[arcs["arc_id"] == int(arc_id)]
    if row.empty:
        return None
    arc = row.iloc[0]
    nodes = trip["nodes"].set_index("node_id")
    try:
        start = nodes.loc[int(arc.from_node_id)]
        end = nodes.loc[int(arc.to_node_id)]
        coords = [[float(start.latitude), float(start.longitude)],
                  [float(end.latitude), float(end.longitude)]]
    except (KeyError, TypeError, ValueError):
        return None
    if not np.isfinite(np.asarray(coords)).all():
        return None
    return coords


def build_map(trip, predicted_arc_ids=(), reference_arc_ids=(), *, include_reference=False,
              max_roads=2500):
    """Render GPS, sampled network, predicted road arcs, and optional dev truth."""
    gps = trip["track"]
    gps = gps[np.isfinite(gps[["latitude", "longitude"]]).all(axis=1)]
    gps = gps[gps.latitude.between(-90, 90) & gps.longitude.between(-180, 180)]
    if gps.empty:
        raise ValueError("No valid GPS coordinates")
    points = gps[["latitude", "longitude"]].to_numpy().tolist()
    center = [float(gps.latitude.mean()), float(gps.longitude.mean())]
    map_view = folium.Map(center, zoom_start=14, control_scale=True, prefer_canvas=True)

    road_layer = folium.FeatureGroup(name="Road network", show=True)
    nodes = trip["nodes"].set_index("node_id")
    arcs = trip["arcs"]
    pad_lat = max(0.003, float(gps.latitude.max() - gps.latitude.min()) * 0.25)
    pad_lon = max(0.003, float(gps.longitude.max() - gps.longitude.min()) * 0.25)
    min_lat, max_lat = gps.latitude.min() - pad_lat, gps.latitude.max() + pad_lat
    min_lon, max_lon = gps.longitude.min() - pad_lon, gps.longitude.max() + pad_lon
    selected = []
    for arc in arcs.itertuples(index=False):
        try:
            start, end = nodes.loc[int(arc.from_node_id)], nodes.loc[int(arc.to_node_id)]
        except KeyError:
            continue
        inside = all(min_lon <= point.longitude <= max_lon and min_lat <= point.latitude <= max_lat
                     for point in (start, end))
        if inside:
            selected.append(int(arc.arc_id))
    if len(selected) > max_roads:
        selected = selected[::math.ceil(len(selected) / max_roads)]
    for arc_id in selected:
        coords = _arc_coords(trip, arc_id)
        if coords:
            folium.PolyLine(coords, color="#9aa4b2", weight=2, opacity=0.5,
                            tooltip=f"Road arc {arc_id}").add_to(road_layer)
    road_layer.add_to(map_view)

    raw = folium.FeatureGroup(name="GPS observations", show=True)
    for index, point in enumerate(points):
        folium.CircleMarker(point, radius=3, color="#2677d8", fill=True,
                            fill_opacity=0.8, tooltip=f"GPS observation {index}").add_to(raw)
    if len(points) > 1:
        folium.PolyLine(points, color="#2677d8", weight=2, opacity=0.35,
                        tooltip="Observed GPS trace").add_to(raw)
    raw.add_to(map_view)

    predicted = folium.FeatureGroup(name="Predicted route", show=True)
    for arc_id in predicted_arc_ids:
        coords = _arc_coords(trip, arc_id)
        if coords:
            folium.PolyLine(coords, color="#159447", weight=5, opacity=0.9,
                            tooltip=f"Predicted arc {arc_id}").add_to(predicted)
    predicted.add_to(map_view)

    if include_reference:
        reference = folium.FeatureGroup(name="Reference route (development only)", show=False)
        for arc_id in reference_arc_ids:
            coords = _arc_coords(trip, arc_id)
            if coords:
                folium.PolyLine(coords, color="#e89022", weight=4, opacity=0.85,
                                tooltip=f"Reference arc {arc_id}").add_to(reference)
        reference.add_to(map_view)

    folium.Marker(points[0], tooltip="Start", icon=folium.Icon(color="green")).add_to(map_view)
    folium.Marker(points[-1], tooltip="End", icon=folium.Icon(color="red")).add_to(map_view)
    map_view.fit_bounds([[float(gps.latitude.min()), float(gps.longitude.min())],
                         [float(gps.latitude.max()), float(gps.longitude.max())]])
    folium.LayerControl(collapsed=False).add_to(map_view)
    return map_view
