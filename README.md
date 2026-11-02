# Intervertebral priors for biplanar spine registration

An exploratory study of **five-vertebra 2D–3D registration** under synthetic biplanar imaging. This repository contains selected code snapshots, protocols, analysis tables and diagnostics accompanying the manuscript *Intervertebral priors for biplanar 2D–3D registration of five lumbar vertebrae: an exploratory synthetic evaluation*.

[Models](#registration-models) · [Results](#selected-results) · [Reproduce statistics](#reproduce-the-statistics) · [Repository guide](#repository-guide) · [Data and limitations](#data-and-limitations)

## Registration models

| Model | Representation | Intervertebral prior |
|:--|:--|:--|
| **A · Global rigid** | One rigid transform for L1–L5 | No relative motion between vertebrae |
| **B · Independent** | Five independently parameterized vertebral transforms | None |
| **C · Fixed** | Five transforms | Common weight for adjacent relative-pose penalties |
| **C · Adaptive** | Five transforms | Edge-specific penalty weights derived from candidate dispersion |

The fixed and adaptive models share the multivertebra representation, candidate initialization and refinement settings of B. The adaptive dispersion is a weighting signal; it is **not calibrated pose uncertainty**.

## Selected results

The frozen extension compared paired methods on two separately analyzed CT-source cohorts. Targets were generated with a Siddon integrator, while optimization retained the midpoint/trilinear renderer. The table reports **combined synthetic corruption** and CT-level means conditional on geometric eligibility.

| Cohort | Method | RMS-TRE, mm ↓ | Centroid-failure frequency ↓ |
|:--|:--|--:|--:|
| Reused · 47 eligible CTs | B · Independent | 1.031 | 29.79% |
| | C · Fixed, λ = 0.01 | 0.476 | 8.09% |
| | C · Adaptive | 0.439 | 9.36% |
| External source · 45 eligible CTs | B · Independent | 1.387 | 41.78% |
| | C · Fixed, λ = 0.01 | 0.671 | 6.22% |
| | C · Adaptive | 0.623 | 9.33% |

RMS-TRE is the mean boundary-point root-mean-square target-registration error. A target fails when any vertebral centroid error exceeds 3 mm. Five corrupted targets per CT are clustered replicates, not independent samples. These rows are a subset of the comparisons in the manuscript; A was not part of the frozen extension's matched final refinements.

Compared with fixed 0.01, adaptive weighting lowered mean RMS-TRE in both cohorts but had more observed centroid failures. B had the lowest clean-image RMS-TRE. The findings describe an accuracy–robustness tradeoff under the tested conditions, not a universal ranking. Confidence intervals and additional comparisons are retained in [`analysis/`](analysis/).

## Reproduce the statistics

Python and NumPy are required. The check runs locally without CT images, a GPU or network access once dependencies are installed:

```bash
python -m pip install numpy
python reproduce_statistics.py
```

The script recomputes CT-level summaries and paired bootstrap intervals from [`analysis/task_outcomes.csv`](analysis/task_outcomes.csv), then checks them against [`analysis/analysis_summary.json`](analysis/analysis_summary.json). It checks 2,208 target-method rows and 80 intervals or contrasts. It does **not** rerun image formation, registration or optimization.

[`build_validation_figures.py`](build_validation_figures.py) additionally requires Matplotlib. Full registration replay requires source data obtained under the providers' terms and a suitable reconstruction of the original computing environment; the repository does not establish end-to-end replay.

## Repository guide

| Path | Contents |
|:--|:--|
| [`experiments/`](experiments/) | Historical and external-source experiment code snapshots |
| [`analysis/`](analysis/) | Selected per-task outcomes, summaries and paired contrasts |
| [`protocol/`](protocol/) | Frozen protocol records and selected audit material |
| [`input_manifests/`](input_manifests/) | Selected input and cohort assignment records |
| [`numerical_diagnostics/`](numerical_diagnostics/) | Detector-sampling and midpoint-step diagnostics |
| [`MANIFEST.json`](MANIFEST.json) | SHA-256 checksums for selected files from the retained archive |

## Data and limitations

Native CT images, segmentations and target arrays are not included. PPCNet and TotalSegmentator data must be obtained separately under their respective provider terms. This repository contains selected records, not every historical experiment record.

The evaluation uses synthetic images and shared modeling assumptions. Patient independence across cohorts was not verified; the detector-sampling checks also limit interpretation. These results do not establish clinical accuracy, unseen-patient generalization or sampling-independent rankings. Validation against patient-matched real radiographs and reference poses remains necessary.

## License

Repository code is provided under the [MIT License](LICENSE). Third-party datasets, software and their associated materials remain subject to their own terms.
