"""Folium map layers for raw GPS, road arcs and road-based predicted routes."""
from __future__ import annotations
import math
import numpy as np
import folium
from src.data.person4_candidates import Trip

def _coords(trip,aid):
    if int(aid)<0 or int(aid)>=len(trip.arcs):return None
    a,b=map(int,trip.arcs[int(aid)])
    if min(a,b)<0 or max(a,b)>=len(trip.nodes):return None
    p,q=trip.nodes[a],trip.nodes[b]
    if not np.isfinite([*p,*q]).all() or abs(p[0])>180 or abs(p[1])>90 or abs(q[0])>180 or abs(q[1])>90:return None
    return [[float(p[1]),float(p[0])],[float(q[1]),float(q[0])]]

def build_map(trip:Trip,predicted_arc_ids=(),reference_arc_ids=(),include_reference=False,max_roads=2500):
    gps=trip.gps;good=gps[np.isfinite(gps[["lat","lon"]]).all(axis=1)&gps.lat.between(-90,90)&gps.lon.between(-180,180)]
    if good.empty:raise ValueError("No valid GPS coordinates")
    points=good[["lat","lon"]].to_numpy().tolist();m=folium.Map([float(good.lat.mean()),float(good.lon.mean())],zoom_start=14,control_scale=True,prefer_canvas=True)
    roads=folium.FeatureGroup(name="Road network",show=True); padx=max(.003,float(good.lon.max()-good.lon.min())*.25);pady=max(.003,float(good.lat.max()-good.lat.min())*.25);ids=[]
    for aid,(a,b) in enumerate(trip.arcs):
        if max(a,b)>=len(trip.nodes):continue
        p,q=trip.nodes[a],trip.nodes[b]
        if good.lon.min()-padx<=p[0]<=good.lon.max()+padx and good.lat.min()-pady<=p[1]<=good.lat.max()+pady and good.lon.min()-padx<=q[0]<=good.lon.max()+padx and good.lat.min()-pady<=q[1]<=good.lat.max()+pady:ids.append(aid)
    if len(ids)>max_roads:ids=ids[::math.ceil(len(ids)/max_roads)]
    for aid in ids:
        c=_coords(trip,aid)
        if c:folium.PolyLine(c,color="#9aa4b2",weight=2,opacity=.48,tooltip=f"Road arc {aid}").add_to(roads)
    roads.add_to(m);raw=folium.FeatureGroup(name="GPS observations",show=True)
    for i,p in enumerate(points):folium.CircleMarker(p,radius=3,color="#2677d8",fill=True,fill_opacity=.8,tooltip=f"GPS {i}").add_to(raw)
    if len(points)>1:folium.PolyLine(points,color="#2677d8",weight=2,opacity=.35).add_to(raw)
    raw.add_to(m);pred=folium.FeatureGroup(name="Predicted route",show=True)
    for aid in predicted_arc_ids:
        c=_coords(trip,aid)
        if c:folium.PolyLine(c,color="#159447",weight=5,opacity=.9,tooltip=f"Predicted arc {aid}").add_to(pred)
    pred.add_to(m)
    if include_reference:
        ref=folium.FeatureGroup(name="Reference route (development only)",show=False)
        for aid in reference_arc_ids:
            c=_coords(trip,aid)
            if c:folium.PolyLine(c,color="#e89022",weight=4,opacity=.85,tooltip=f"Reference arc {aid}").add_to(ref)
        ref.add_to(m)
    folium.Marker(points[0],tooltip="Start",icon=folium.Icon(color="green")).add_to(m);folium.Marker(points[-1],tooltip="End",icon=folium.Icon(color="red")).add_to(m)
    m.fit_bounds([[float(good.lat.min()),float(good.lon.min())],[float(good.lat.max()),float(good.lon.max())]]);folium.LayerControl(collapsed=False).add_to(m);return m
