# Two-laptop demo checklist

## Laptop 1 — backend

1. In the project environment, install `python -m pip install -r requirements.txt`.
2. Start the team's API with its documented command, bound to `0.0.0.0:8000` (for example `uvicorn <backend_module>:app --host 0.0.0.0 --port 8000`).
3. Find the private LAN IPv4 address with `ipconfig`; ensure TCP 8000 is allowed on the private network.
4. Confirm `/api/health` reports ready and `/api/trips` returns the allowed evaluator trip IDs. Keep the terminal visible to show genuine inference logs.

## Laptop 2 — frontend

1. Install the same requirements file.
2. Set `$env:PATHSNAP_API_URL="http://<LAPTOP_1_LAN_IP>:8000"` in PowerShell.
3. Verify `Invoke-RestMethod "$env:PATHSNAP_API_URL/api/health"`.
4. Start `python -m streamlit run frontend/app.py --server.address 0.0.0.0`.
5. Select an ID actually returned by the API, click Predict Route, and explain which metrics are absent versus returned.
6. Repeat on multiple allowed IDs and show the visible Laptop 1 server logs.

Troubleshoot wrong LAN IP, different/VPN-isolated networks, host firewall port 8000, API readiness, timeout and actual endpoint/JSON schema. The API was unavailable in the empty repository, and cross-device testing remains pending; do not claim the live integration is verified until performed with Person 3's server.
