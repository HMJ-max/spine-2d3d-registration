# Production entry static review

Read-only source review on2026-10-27; no external registration outcomes were read and no GPU code was executed by this reviewer.

Reviewed snapshots:

- `run_production.py`: SHA25657ae887f046c4345a960c505f6e6d1f820f61488408c86d3c645c0ba42a4eea1
- `run_siddon_validation_v2.py`: SHA25671e3da1b28242e5d11ad2be5dfd72885e5cce995bdbff4e11c9fc6b5f6d91d88
- `audit_validation_outputs.py`: SHA256a8b4fdb4542825cad0669bf67f2738ab3876371de6e69070bc228a36aa115a9c

Core scientific flow is consistent with frozen design_v3. The external base uses20361024 for motion/corruption/A and20361025 for candidate perturbations, with indices from the frozen assigned-case order. Geometry excludes cases without renumbering. Input-QC and derived hashes are checked before loading. The original motion function, proxy, camera, prior scales and optimizer are reused. Trilinear512-to256 references define corruption scales/masks; the objective receives independent Siddon512-to256 targets. New A and five prior-free candidates determine a common winner for B, fixed0.003, fixed0.01 and adaptive. Adaptive parameters are retained from old development. No observed-effect-based selection is implemented.

V2 adds reference-support cache provenance and explicit mask/plan/RNG hashes without altering the original estimator functions. The earlier AST audit established identity of20 critical retained definitions. Midpoint diagnostics use the locked old-development cases, fixed GT/parameter-offset poses and four predeclared step requests; production h1/cap192 sample-count agreement is explicitly reported.

Two audit/entry follow-ups were sent to the renderer agent before production source freeze:

1. The external entry should require50assigned and verify the assignment manifest SHA against the finalizer receipt, in addition to recording the supplied manifest hash.
2. Coverage auditing must require all six scenarios for every geometrically accepted assigned case while explicitly accounting for each config-consistent accepted=false geometry record. The initial audit assumed every manifest case had six results; that would misclassify legitimate external geometry exclusions as missing computation. Old assigned50/eligible47 denominators must also remain visible.

These follow-ups affect integrity/coverage checks only. They do not authorize weight, optimizer, stopping, sample-selection or endpoint changes. A follow-up source review should record resolution and new hashes after the renderer agent applies them.
