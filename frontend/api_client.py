"""Configurable API client; errors remain visible to the UI."""
from __future__ import annotations
import os,requests
class APIError(RuntimeError):pass
class PathSnapClient:
 def __init__(self,base_url=None,timeout=30):self.base_url=(base_url or os.getenv("PATHSNAP_API_URL","http://127.0.0.1:8000")).rstrip("/");self.timeout=timeout
 def _request(self,method,path,**kwargs):
  try:r=requests.request(method,self.base_url+path,timeout=self.timeout,**kwargs);r.raise_for_status();d=r.json()
  except requests.Timeout as e:raise APIError("Backend request timed out") from e
  except requests.RequestException as e:raise APIError(f"Backend unavailable: {e}") from e
  except ValueError as e:raise APIError("Backend returned malformed JSON") from e
  if not isinstance(d,dict):raise APIError("Backend response must be a JSON object")
  return d
 def health(self):return self._request("GET","/api/health")
 def trips(self):
  d=self._request("GET","/api/trips");items=d.get("trip_ids",d.get("trips"))
  if not isinstance(items,list):raise APIError("Trip response must contain trip_ids or trips list")
  return [str(x.get("trip_id") if isinstance(x,dict) else x) for x in items if x is not None]
 def predict(self,trip_id):
  d=self._request("POST","/api/predict",json={"trip_id":str(trip_id)})
  if str(d.get("trip_id"))!=str(trip_id):raise APIError("Prediction response trip_id mismatch")
  if not isinstance(d.get("metrics",{}),dict):raise APIError("metrics must be an object")
  if not isinstance(d.get("gps_points",[]),list) or not isinstance(d.get("predicted_route",[]),list):raise APIError("Geometry fields must be lists")
  return d
