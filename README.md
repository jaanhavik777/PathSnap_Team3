# PathSnap — Team 3

## Current status
Real dataset parsers, inventory and Folium exploration map implemented. Model training, candidate labeling, HMM and API are **not yet implemented**. Other source files are scaffolding placeholders.

## Setup
```bash
python -m venv .venv
pip install -r requirements.txt
python -m scripts.inspect_data
python -m scripts.make_map data/raw/train/00000001 --out artifacts/maps/trip_00000001.html
python -m pytest tests/test_parsers.py
```

## Raw data
Extract the provided Student folder ZIP under `data/raw/` so that `data/raw/train/00000001/00000001.track` exists. The bundled development ZIP includes the provided train and student_test files; raw data is gitignored to avoid accidentally publishing the supplied dataset.

## Critical data convention
`.nodes` and `.arcs` use zero-based row indices as IDs. `.route` contains a chronological sequence of arc IDs, **not** GPS-aligned observation labels. Do not train on route row index as if it were GPS observation index. Student-test routes are for development evaluation only; never expose reference routes to prediction inference.
