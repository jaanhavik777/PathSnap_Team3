"""PathSnap evaluator dashboard; all predictions come from FastAPI."""

from __future__ import annotations

import streamlit as st
from streamlit_folium import st_folium

from frontend.api_client import APIError, PathSnapClient
from frontend.map_layers import build_prediction_map


st.set_page_config(page_title="PathSnap | Map Matching", page_icon="🗺️", layout="wide")
st.markdown(
    """<style>
    .block-container { max-width: 1440px; padding-top: 1.6rem; }
    .hero { padding: 1.4rem 1.7rem; border-radius: 16px;
      background: linear-gradient(120deg, #13253d, #245a73); color: white; }
    .hero p { margin-bottom: 0; color: #e4edf5; }
    </style>""",
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="hero"><h1>PathSnap — Intelligent GPS Map Matching</h1>'
    "<p>Reconstructing road routes from noisy GPS observations</p></div>",
    unsafe_allow_html=True,
)

client = PathSnapClient()
st.sidebar.header("Connection")
st.sidebar.caption(f"API: {client.base_url}")
try:
    health = client.health()
    st.sidebar.success("Backend connected")
    if health.get("ready") is False:
        st.sidebar.warning("Backend is online but reports that it is not ready.")
except APIError as exc:
    st.sidebar.error("Backend unavailable")
    st.error(str(exc))
    st.caption(
        "Set PATHSNAP_API_URL to Laptop 1's reachable address, "
        "for example http://192.168.1.20:8000."
    )
    st.stop()

try:
    trip_ids = client.trips()
except APIError as exc:
    st.error(f"Could not load trip IDs: {exc}")
    st.stop()
if not trip_ids:
    st.warning("The backend currently advertises no permitted trip IDs.")
    st.stop()

trip_id = st.selectbox("Select evaluation trip", trip_ids, key="trip_select")
previous_trip = st.session_state.get("prediction_trip_id")
if previous_trip is not None and previous_trip != trip_id:
    st.session_state.pop("prediction", None)
    st.session_state.pop("prediction_trip_id", None)

if st.button("Predict route", type="primary"):
    st.session_state.pop("prediction", None)
    st.session_state.pop("prediction_trip_id", None)
    with st.spinner(f"Generating a fresh prediction for trip {trip_id}…"):
        try:
            st.session_state["prediction"] = client.predict(trip_id)
            st.session_state["prediction_trip_id"] = trip_id
        except APIError as exc:
            st.error(f"Prediction failed: {exc}")

prediction = st.session_state.get("prediction")
if prediction and st.session_state.get("prediction_trip_id") == trip_id:
    st.success(f"Prediction received for trip {trip_id}")
    metrics = prediction.get("metrics") or {}
    gps_points = prediction.get("gps_points") or []
    predicted_route = prediction.get("predicted_route") or []
    accuracy = metrics.get("accuracy", metrics.get("prediction_accuracy"))
    inference_ms = prediction.get("inference_time_ms", metrics.get("inference_time_ms"))
    try:
        accuracy = f"{float(accuracy):.3f}" if accuracy is not None else "Unavailable"
    except (TypeError, ValueError):
        accuracy = "Unavailable"
    try:
        inference_ms = f"{float(inference_ms):.2f} ms" if inference_ms is not None else "Unavailable"
    except (TypeError, ValueError):
        inference_ms = "Unavailable"
    columns = st.columns(4)
    columns[0].metric("Accuracy", accuracy)
    columns[1].metric("GPS observations", metrics.get("gps_observations", len(gps_points)))
    columns[2].metric("Predicted segments", metrics.get("predicted_segments", len(predicted_route)))
    columns[3].metric("Inference time", inference_ms)
    if metrics.get("route_continuity") is not None:
        st.caption(f"Route continuity: {metrics['route_continuity']}")

    try:
        st_folium(
            build_prediction_map(prediction), height=640,
            use_container_width=True, returned_objects=[],
        )
    except ValueError as exc:
        st.warning(f"Prediction received, but map geometry is unavailable: {exc}")
    with st.expander("Metrics returned by the API"):
        st.json(metrics)
elif prediction:
    st.info("Select Predict Route to generate a fresh result for this trip.")
else:
    st.info("Choose a trip and run a prediction. Results come directly from FastAPI.")

st.caption(
    "Trip IDs, route geometry and metrics are supplied by FastAPI. "
    "The frontend does not read reference labels or generate predictions."
)
