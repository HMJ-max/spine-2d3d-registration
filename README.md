# Intervertebral priors for biplanar 2D–3D spine registration

This repository accompanies the manuscript *Intervertebral priors for biplanar 2D–3D registration of five lumbar vertebrae: an exploratory synthetic evaluation*. It contains selected source snapshots, analysis tables, protocol records and scripts.

## Reproduce the reported statistics

With Python and NumPy installed, run:

```bash
python reproduce_statistics.py
```

The script recomputes CT-level summaries and paired bootstrap intervals from `analysis/task_outcomes.csv` and checks the results against `analysis/analysis_summary.json`. It does not need CT images, a GPU or network access. The five corruption seeds for a CT are clustered replicates; patient independence across the cohorts is not established.

`build_validation_figures.py` additionally requires Matplotlib. `numerical_diagnostics/` contains detector-sampling and midpoint-step summaries. `protocol/` and `input_manifests/` retain selected protocol and input records; the two `experiments/*/source_snapshot/` directories preserve the corresponding code snapshots.

## Scope and data

This repository is a selected subset of the retained experiment records and does not contain all historical records. Native CT, segmentations and target arrays are not included. Reproducing GPU experiments requires independently obtaining source data under the providers' terms and recreating the recorded environment; the portable statistics check alone does not establish full registration replay.

The manuscript reports an exploratory synthetic evaluation. Different integration methods do not eliminate the shared simulation assumptions. The analysis does not establish clinical accuracy, cross-dataset patient independence or a sampling-independent ranking.

## Integrity and release

`MANIFEST.json` records SHA-256 checksums for selected files and the source archive's checksum. Third-party datasets and dependencies remain subject to their respective terms.
