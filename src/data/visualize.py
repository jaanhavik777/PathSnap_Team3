from pathlib import Path
import folium
from .parsers import load_trip
def make_trip_map(folder, output, max_arcs=15000):
    t=load_trip(folder); gps=t['track'];nodes=t['nodes'];arcs=t['arcs'];route=t['route']
    center=[gps.latitude.median(),gps.longitude.median()];m=folium.Map(location=center,zoom_start=13)
    gps_layer=folium.FeatureGroup(name='Raw GPS',show=True)
    folium.PolyLine(gps[['latitude','longitude']].values.tolist(),color='blue',weight=3).add_to(gps_layer);gps_layer.add_to(m)
    network=folium.FeatureGroup(name='Road network (sample)',show=False)
    for row in arcs.iloc[:max_arcs].itertuples(index=False):
        a=nodes.iloc[row.from_node_id];b=nodes.iloc[row.to_node_id]
        folium.PolyLine([[a.latitude,a.longitude],[b.latitude,b.longitude]],color='gray',weight=1,opacity=.35).add_to(network)
    network.add_to(m)
    if route is not None:
        truth=folium.FeatureGroup(name='Reference route',show=True)
        for arc_id in route.arc_id:
            a_id,b_id=arcs.iloc[int(arc_id)][['from_node_id','to_node_id']]
            a=nodes.iloc[int(a_id)];b=nodes.iloc[int(b_id)]
            folium.PolyLine([[a.latitude,a.longitude],[b.latitude,b.longitude]],color='green',weight=4,opacity=.75).add_to(truth)
        truth.add_to(m)
    folium.LayerControl().add_to(m);Path(output).parent.mkdir(parents=True,exist_ok=True);m.save(str(output))
