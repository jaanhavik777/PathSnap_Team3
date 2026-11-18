# Contributing to PathSnap

See [contributors](CONTRIBUTORS.md) for the owners of each component.

1. Work on a feature branch and keep changes focused. Review `git diff --cached` before committing.
2. Reuse the [shared feature contract](docs/feature_contract.md). When feature defaults or label rules change, coordinate regeneration with all model owners.
3. Fit preprocessing and select model parameters inside trip-grouped training folds. Use the student-test set for development comparison and keep evaluator references outside training and prediction.
4. Report the partition, metric definition, label coverage, and validation split with each result. Specify binary or macro F1 and whether continuity was measured before or after decoding.
5. Run `python -m pytest -q`. GitHub Actions runs the suite on push and pull request. The parser smoke test that uses the private dataset runs when its fixture is available locally.
6. Keep raw data, private credentials, and generated local outputs out of commits. Include reproducible commands and any disclosures required by the event with your handoff.
