"""Render backend-returned geometry without accessing labels or trip files."""
from __future__ import annotations
import folium

def _point(p):
 if isinstance(p,dict):lat,lon=p.get("lat"),p.get("lon")
 elif isinstance(p,(list,tuple)) and len(p)>=2:lat,lon=p[0],p[1]
 else:return None
 try:lat,lon=float(lat),float(lon)
 except (TypeError,ValueError):return None
 return [lat,lon] if -90<=lat<=90 and -180<=lon<=180 else None

def _segment(s):
 if isinstance(s,dict):coords=s.get("coordinates") or s.get("geometry")
 else:coords=s
 if isinstance(coords,dict):coords=coords.get("coordinates")
 if not isinstance(coords,(list,tuple)):return []
 # API geometry follows conventional [lon,lat] GeoJSON coordinates.
 points=[]
 for p in coords:
  if isinstance(p,(list,tuple)) and len(p)>=2:
   try:lon,lat=float(p[0]),float(p[1])
   except (TypeError,ValueError):continue
   if -180<=lon<=180 and -90<=lat<=90:points.append([lat,lon])
 return points

def build_prediction_map(data):
 points=[p for p in (_point(x) for x in data.get("gps_points",[])) if p]
 segments=[p for p in (_segment(x) for x in data.get("road_segments",[])) if len(p)>1]
 route=[p for p in (_segment(x) for x in data.get("predicted_route",[])) if len(p)>1]
 centers=points+[q for g in segments+route for q in g]
 if not centers:raise ValueError("API response contains no usable map geometry")
 center=[sum(p[0] for p in centers)/len(centers),sum(p[1] for p in centers)/len(centers)]
 m=folium.Map(center,zoom_start=14,control_scale=True,prefer_canvas=True)
 for name,groups,color,weight in [("Road network",segments,"#9aa4b2",2),("Predicted route",route,"#159447",5)]:
  layer=folium.FeatureGroup(name=name)
  for i,g in enumerate(groups):folium.PolyLine(g,color=color,weight=weight,tooltip=f"{name} {i+1}").add_to(layer)
  layer.add_to(m)
 gps=folium.FeatureGroup(name="GPS observations")
 for i,p in enumerate(points):folium.CircleMarker(p,radius=3,color="#2677d8",fill=True,tooltip=f"GPS {i+1}").add_to(gps)
 gps.add_to(m)
 if points:
  folium.Marker(points[0],tooltip="Start",icon=folium.Icon(color="green")).add_to(m);folium.Marker(points[-1],tooltip="End",icon=folium.Icon(color="red")).add_to(m)
 m.fit_bounds(centers);folium.LayerControl(collapsed=False).add_to(m);return m
