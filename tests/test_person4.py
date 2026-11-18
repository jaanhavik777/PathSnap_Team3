from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import pytest
from src.data.person4_candidates import Trip,candidate_features,FEATURE_COLUMNS
from src.data.person4_maps import build_map
from frontend.map_layers import build_prediction_map
from frontend.api_client import PathSnapClient,APIError
from src.models.train_mlp import make_model

def fixture_trip():
 return Trip("fixture",pd.DataFrame([[0.0004,0.0,0.0],[0.0014,0.0,1.0]],columns=["lon","lat","timestamp"]),np.array([[0.,0.],[.001,0.],[.002,0.]]),np.array([[0,1],[1,2]]),np.array([0,1]))

def test_candidate_features_and_observation_grouping():
 f=candidate_features(fixture_trip(),candidate_count=2)
 assert len(f)==4 and set(FEATURE_COLUMNS).issubset(f.columns)
 assert f.groupby("observation_id").size().eq(2).all()
 assert f.label.sum()>=1
 assert (f.loc[f.reference_arc_id.eq(-1),"label"]==-1).all()

def test_arc_ids_and_coordinate_order_rendered():
 html=build_map(fixture_trip(),predicted_arc_ids=[1]).get_root().render()
 assert "Predicted route" in html and "0.001" in html

def test_api_trip_response_and_payload(monkeypatch):
 class Resp:
  def raise_for_status(self):pass
  def json(self):return {"trip_ids":["a","b"]}
 def get(url,timeout):return Resp()
 monkeypatch.setattr("frontend.api_client.requests.request",lambda method,url,**kwargs:get(url,kwargs.get("timeout")))
 assert PathSnapClient("http://fake").trips()==["a","b"]

def test_prediction_response_validation(monkeypatch):
 class Resp:
  def raise_for_status(self):pass
  def json(self):return {"trip_id":"x","gps_points":[],"predicted_route":[],"metrics":{}}
 monkeypatch.setattr("frontend.api_client.requests.request",lambda *a,**k:Resp())
 assert PathSnapClient("http://fake").predict("x")["trip_id"]=="x"

def test_missing_geometry_is_reported():
 with pytest.raises(ValueError):build_prediction_map({"gps_points":[],"road_segments":[],"predicted_route":[]})

def test_small_mlp_fit_probability_shape_and_reload(tmp_path):
 X=pd.DataFrame({c:np.linspace(0,1,24) for c in FEATURE_COLUMNS})
 y=np.array([0,1]*12);model=make_model(hidden=(4,),max_iter=5,seed=3).fit(X,y)
 p=model.predict_proba(X)
 assert p.shape==(24,2) and np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
 import joblib
 path=tmp_path/'model.joblib';joblib.dump(model,path);loaded=joblib.load(path)
 np.testing.assert_array_equal(model.predict(X),loaded.predict(X))

def test_api_timeout_is_clear(monkeypatch):
 import requests
 def timeout(*args,**kwargs):raise requests.Timeout('slow')
 monkeypatch.setattr('frontend.api_client.requests.request',timeout)
 with pytest.raises(APIError,match='timed out'):PathSnapClient('http://fake',timeout=.01).health()

def test_prediction_trip_id_mismatch_is_rejected(monkeypatch):
 class Resp:
  def raise_for_status(self):pass
  def json(self):return {'trip_id':'different','metrics':{},'gps_points':[],'predicted_route':[]}
 monkeypatch.setattr('frontend.api_client.requests.request',lambda *a,**k:Resp())
 with pytest.raises(APIError,match='mismatch'):PathSnapClient('http://fake').predict('expected')
