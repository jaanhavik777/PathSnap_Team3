# Frontend API contract — pending Person 3 confirmation

The scaffold's `api/` implementation files and schemas were empty when inspected. The frontend client is implemented against this proposed contract; adjust its parsing when Person 3 publishes the actual schema.

## Health

`GET /api/health` returns a JSON object indicating service status and readiness. The UI treats any valid object with an HTTP success status as connected.

## Available trips

`GET /api/trips` returns either `{"trip_ids":["..."]}` or `{"trips":["..."]}`. Each list item may also be an object with a `trip_id` field. IDs are obtained dynamically; the frontend does not hardcode evaluator IDs.

## Prediction

`POST /api/predict` request:

```json
{"trip_id":"00000000"}
```

Proposed response:

```json
{
  "trip_id":"00000000",
  "gps_points":[{"lat":49.84,"lon":8.59}],
  "road_segments":[{"arc_id":12,"coordinates":[[8.59,49.84],[8.60,49.85]]}],
  "predicted_route":[{"arc_id":12,"coordinates":[[8.59,49.84],[8.60,49.85]]}],
  "metrics":{"accuracy":null,"route_continuity":null},
  "inference_time_ms":null
}
```

The numbers above are illustrative schema examples, not results. GPS points may be `[lat, lon]` arrays or `{lat, lon}` objects. Road geometry uses GeoJSON coordinate order `[longitude, latitude]`, with `coordinates` as a LineString coordinate list. Return only permitted prediction data and server-computed permitted metrics. Evaluator ground truth must not be sent to the UI. The client checks status, timeout, response JSON, trip ID, metrics type, and geometry list shapes; it never substitutes sample predictions.

The real API schemas, response body, and two-device behavior have not yet been verified because backend modules remain placeholders.
