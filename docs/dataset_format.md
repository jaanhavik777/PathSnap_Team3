# Dataset format (verified from provided official Dataset Exploration PDF and files)
- `.track`: longitude, latitude, timestamp seconds (whitespace separated; no header).
- `.nodes`: longitude, latitude; node_id is zero-based row index.
- `.arcs`: from_node_id, to_node_id; arc_id is zero-based row index.
- `.route`: chronological arc_id list; NOT aligned 1:1 to GPS observations.
- `train`: 80 trips; `student_test`: 10 trips. Evaluator set not included.
- Never treat route membership alone as a definitive per-observation label: repeated passes and parallel roads can make alignment ambiguous.
