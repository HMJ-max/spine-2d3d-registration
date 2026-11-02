# Finalize the external input assignment

The script reads only source, input-QC and fingerprint data. It does not read motion, registration, RMS or failure results and does not use the GPU.

Run a complete preflight once all60 source QC records are available:

```powershell
& 'D:/anaconda/python.exe' 'work/validation_20261026/protocol/finalize_external_assignment.py'
```

The default command never freezes assignment. While acquisition/QC is incomplete it writes an audit report, returns exit2 and prints `INCOMPLETE_OR_INSUFFICIENT_NO_FREEZE`. `--limit 2` performs a small read-only input check and is incompatible with `--freeze`.

Once preflight succeeds, execute the same complete checks and atomically freeze the first50 input-eligible scans:

```powershell
& 'D:/anaconda/python.exe' 'work/validation_20261026/protocol/finalize_external_assignment.py' --freeze
```

Output: `protocol/freezes/external_assignment_v1/external_assigned_manifest.json`, `full_input_and_pair_audit.json`, and `receipt.json`. An existing assignment version is never overwritten. The manifest provides `assigned_cases` in frozen order and `cases[case]` with immutable `case_index`, input directory, native paths, source-QC path/SHA256 and derived hashes. External seed bases remain20361024 and20361025.

The verifier requires the immutable protocol/source-order receipts, validates the complete1201-ID ordering and first60 mask-passing selection, and checks all original CT plus five mask files, source manifest, current QC code/criteria, QC record and derived-file hashes. It loads authenticated old75 fingerprint records directly from the retained archive. It compares all60 candidates to all75 old scans and every new-candidate pair:4500+1770=6270 pair slots. Invalid input-QC cases cannot supply valid fingerprints; their pair slots are explicitly marked not comparable and the case remains ineligible. Such a completed data rejection does not block assignment if at least50 other inputs pass. Missing/stale/provenance-invalid records do block freezing.

Every new endpoint of an image duplicate, near-match or anatomy-ambiguity flag is conservatively quarantined; the script does not retain an arbitrary member of a suspicious cluster or equate a flag with confirmed patient identity. Preliminary quarantines persist. Fewer than50 unflagged inputs produce a documented shortfall and no assignment. The extra ten candidates only cover input-QC losses; geometric or registration failures never trigger replacements.

Validation performed during implementation: two real source/QC cases,150 old-versus-new comparisons and one new-versus-new comparison; all hashes validated, no image flags, and no assignment was written. Full60-case verification and freezing remain the production task. Immutable input-stat checks detect files that change during the audit.
