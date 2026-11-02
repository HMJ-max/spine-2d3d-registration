# Amendment001: align design with the authoritative source-selection freeze

Date: 2026-10-26. This amendment occurs after source acquisition and two source/input QC passes, but before any external-case registration. No external registration outcome has been used.

Design v1 included a provisional generic candidate-order salt while the data acquisition agent had already frozen the source-specific TotalSegmentator order at 2026-10-26T12:43:07.205733+00:00. The source-specific order is authoritative: `IJCARS-20261026-TotalSegmentator-v1-adult-v3-labels`, ascending SHA256(salt + `|` + image_id), then image_id. v2 now references that exact existing list and its SHA256. No candidate has been reranked, substituted using outcomes or resampled because of this correction. The design_v1 snapshot and receipt remain preserved.

The amendment also records the actual source releases: v3 revised masks DOI10.5281/zenodo.22688904 and original v1 cohort DOI10.5281/zenodo.6802614. Unknown NIfTI spatial units are resolved only for derived copies after checking per-case1.5-unit affine spacing and the original cohort paper’s explicit “All images were resampled to 1.5-mm isotropic resolution” statement (DOI10.1148/ryai.230024). Original source files remain byte-identical.

Methods, primary comparison, endpoints, confidence-interval rules, seed offsets, target sampling, corruption construction, old-development convergence selection and engineering review flags are unchanged. Source/patient linkage and the final assigned50 list remain separate pending freeze layers; this amendment does not assert verified patient independence.
