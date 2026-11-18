"""PathSnap evaluator UI. Predictions and permitted trip IDs come from live API."""
import streamlit as st
from streamlit_folium import st_folium
from api_client import PathSnapClient,APIError
from map_layers import build_prediction_map

st.set_page_config(page_title="PathSnap | Map Matching",page_icon="🗺️",layout="wide")
st.markdown("<style>.block-container{max-width:1400px;padding-top:2rem}.hero{padding:1.3rem 1.5rem;border-radius:16px;background:linear-gradient(120deg,#13253d,#245a73);color:white;margin-bottom:1.2rem}.muted{color:#6b7280}</style>",unsafe_allow_html=True)
st.markdown('<div class="hero"><h1>PathSnap — Intelligent GPS Map Matching</h1><p>Reconstructing road routes from noisy GPS observations</p></div>',unsafe_allow_html=True)
client=PathSnapClient()
try:
 health=client.health();st.sidebar.success("Backend connected")
except APIError as exc:
 st.sidebar.error("Backend unavailable");st.error(str(exc));st.caption("Set PATHSNAP_API_URL to Laptop 1's reachable address, for example http://192.168.1.20:8000.");st.stop()
try: trips=client.trips()
except APIError as exc:st.error(f"Could not load trips: {exc}");st.stop()
if not trips:st.warning("The backend currently advertises no trips.");st.stop()
trip_id=st.selectbox("Select evaluation trip",trips,key="trip_select")
if st.button("Predict route",type="primary",use_container_width=False):
 st.session_state.pop("prediction",None)
 with st.spinner(f"Generating route for {trip_id}…"):
  try:st.session_state["prediction"]=client.predict(trip_id)
  except APIError as exc:st.error(f"Prediction failed: {exc}")
data=st.session_state.get("prediction")
if data and str(data.get("trip_id"))==str(trip_id):
 st.success(f"Prediction received for trip {trip_id}")
 metrics=data.get("metrics",{});cols=st.columns(4)
 items=[("Accuracy",metrics.get("accuracy",metrics.get("prediction_accuracy"))), ("GPS observations",metrics.get("gps_observations",len(data.get("gps_points",[])))), ("Predicted segments",metrics.get("predicted_segments",len(data.get("predicted_route",[])))), ("Inference time",data.get("inference_time_ms",metrics.get("inference_time_ms")))]
 for col,(label,value) in zip(cols,items):col.metric(label,"Unavailable" if value is None else (f"{value:.2f} ms" if label=="Inference time" and isinstance(value,(int,float)) else str(value)))
 continuity=metrics.get("route_continuity")
 if continuity is not None:st.caption(f"Route continuity: {continuity}")
 try:st_folium(build_prediction_map(data),height=620,use_container_width=True,returned_objects=[])
 except ValueError as exc:st.warning(f"Prediction received, map geometry unavailable: {exc}")
 with st.expander("Returned metrics and response details"):st.json({"metrics":metrics,"inference_time_ms":data.get("inference_time_ms")})
elif data:st.info("Select Predict Route to generate a result for the newly selected trip.")
else:st.info("Choose a trip and run a prediction. Results are fetched from the backend for each request.")
st.caption("Trip IDs, routes and metrics are supplied by FastAPI. The frontend does not read reference labels or create predictions.")
