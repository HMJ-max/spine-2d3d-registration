"""Create an outcome-blind design manifest; no arrays or outcomes are loaded."""
from pathlib import Path
from hashlib import sha256
import json,datetime

HERE=Path(__file__).resolve().parent
WORK=HERE.parents[1]
SPLIT=WORK/'reviewer_statistics_data/split.json'
split=json.loads(SPLIT.read_text(encoding='utf8'))
salt='ijcars-independent-validation-20261026-convergence-v1'
selection=[]
for fold,ids in enumerate(split['development_folds'],start=1):
    ordered=sorted(ids,key=lambda x:(sha256(f'{salt}:{x}'.encode()).hexdigest(),x))
    selection.append({'development_fold':fold,'selection_rule':'first two by SHA256(salt + colon + case_id), independent of outcome','ordered_all_ids':ordered,'selected_ids':ordered[:2],'hashes':{x:sha256(f'{salt}:{x}'.encode()).hexdigest() for x in ordered}})
cases=[c for row in selection for c in row['selected_ids']]
assert len(cases)==len(set(cases))==10
assert set(cases)<=set(split['development'])
exclusions=['lumbar_0053','lumbar_0055','lumbar_0085']
eligible=sorted(set(split['evaluation'])-set(exclusions))
assert len(eligible)==47
convergence={'version':'1.0','purpose':'Numerical rendering diagnostics on old development anatomy only; no selection using registration performance','selection_salt':salt,'source_split_sha256':sha256(SPLIT.read_bytes()).hexdigest(),'source_split':'work/reviewer_statistics_data/split.json','selection':selection,'selected_ids':cases,'fallback':'No outcome-dependent replacement. Missing/corrupt sources must be logged; a replacement requires an amendment before running convergence on replacement cases.','evaluation_ids_included':False}
(HERE/'convergence_selection.json').write_text(json.dumps(convergence,indent=2),encoding='utf8')
protocol={
 'protocol_version':'20261026-v3-design-locked-midpoint-diagnostic',
 'status':'DESIGN_LOCKED_INPUT_AND_EXECUTION_FREEZES_PENDING',
 'scope':'Independent-integrator validation on old cohort and independent public cohort; separate analyses',
 'root_decisions':{'primary_contrast':['C_adaptive','C_lambda0.01'],'primary_endpoints':['surface_rms_tre_mm','failure_frequency_gt3mm'],'primary_condition':'combined_s0_to_s4_within_patient_mean','inference':'paired effect estimates and confidence intervals; no new omnibus superiority pass/fail gate','pointwise_ci_level':0.95,'primary_bonferroni_marginal_ci_level':0.975,'primary_family_size':2,'primary_family_nominal_coverage':0.95,'secondary_inference':'pointwise 95% intervals, descriptive only; no winner selection','cohorts_pooled':False,'sample_size_power_claim':False},
 'cohorts':{
   'old47_siddon':{'assigned_n':50,'geometrically_eligible_n':47,'assigned_ids':split['evaluation'],'execution_ids':eligible,'historical_geometry_exclusions':exclusions,'independence':'historically exposed CT cohort, patient linkage unverified','purpose':'full eligible-cohort independent-integrator sensitivity, not untouched validation'},
   'new_public_siddon':{'candidate_target_n':60,'assigned_target_n':50,'selection':'first 50 input-eligible distinct units in an outcome-blind salted-hash candidate order, before synthetic geometry checks; no replacement based on registration or generated motion outcomes','candidate_order_salt':'ijcars-independent-validation-20261026-new-candidates-v1','selection_replacements_allowed_for':['source archive integrity failure','invalid affine or voxel spacing','missing complete L1-L5 source labels','documented overlap with old75','documented repeated patient after one scan selected'],'geometry_exclusion_replacements':False,'outcome_based_replacements':False,'if_fewer_than50':'freeze all available qualifying units and report shortfall; do not silently add another source or select by outcomes','patient_unit':'verified unique patient if source metadata support it; otherwise verified source subject if only that level can be established, or CT volume with independence limitation','overlap_screen':'provider/source identifiers and patient linkage plus byte/content hashes where feasible; hashes alone do not prove different patients','manifest_required_before_outcomes':True}
 },
 'methods':{'B':{'prior_weight':0.0},'C_fixed':{'prior_weight':0.003},'C_lambda0.01':{'prior_weight':0.01},'C_adaptive':{'lambda_max':0.01,'tau':0.01389625625827594,'formula':'lambda_max * u_i / (u_i + tau)','development_gate':'historical pooled-development gate retained; do not refit or retest a run/no-run gate on new data'}},
 'fixed_method_settings':{'normalization_rotation_deg':3,'normalization_translation_mm':3,'candidate_count':5,'candidate_generation':'A-derived start plus four original deterministic perturbations, all refined without prior; common image-best start for all four methods','candidate_bank_shared_per_target':True,'final_predictions_shared_across_targets':False,'levels':[64,128,256],'adam_warmup_max':30,'adam_lr':0.2,'lbfgs_history':20,'lbfgs_max_evaluations_per_call':25,'lbfgs_line_search':'strong_wolfe','outer_attempts_per_level':500,'relative_tolerance':0.0001,'relative_patience':5,'gradient_tolerance':0.00001,'stagnation_unchanged_non_adam_iterates':3,'return':'best objective with rollback diagnostics','image_loss':'mean full-image NCC loss over the two views','prior':'mean of four weighted relative SE(3)-log penalties; summed six normalized squared twist coordinates per edge'},
 'targets':{'conditions':['clean','combined_s0','combined_s1','combined_s2','combined_s3','combined_s4'],'same_clean_motion_for_all_seeds':True,'optimizer_renderer':'frozen trilinear midpoint implementation','target_renderer':'independent Siddon exact piecewise-constant voxel-cell path lengths, float64 geometry/integration','production_target_detector_policy':'512x512 independent Siddon projections, exact nonoverlapping 2x2 area mean to 256x256, then generate corruption on the 256 grid; both old47 and new cohort','prediction_detector':256,'physics':'same bone-selective proxy, camera and synthetic motion/corruption; independent integrator is not clinical image formation','old_motion':'reuse saved geometry/GT and original case index; do not replace with recomputed motion','new_case_index':'immutable zero-based index in full frozen candidate manifest, retained after exclusions; no renumbering eligible cases','corruption_seed_formula':'20261024 + 1009 * case_index + 12347 * repeat_index','candidate_seed_formula':'20261025 + 1009 * case_index + 7919 * k; k=1..4; k=0 inherits A','motion_seed_formula':'original frozen run_final_study implementation with base seed 20261024 and immutable case_index; code audit required before freeze','reuse_old_candidate_banks':False,'reuse_old_projection_arrays_as_new_targets':False},
 'endpoints':{'surface_rms_tre_mm':'for each vertebra, sqrt(mean squared boundary-point displacement under estimated vs true transform)); mean five vertebra-specific RMS values; same deterministic up-to-1000 boundary-voxel-centre sampling as retained code','target_failure_gt3mm':'1 iff any of five vertebral centroid errors is strictly greater than 3 mm; threshold is an engineering criterion','within_unit_aggregation':'clean: one target; combined: arithmetic mean of five seed-level RMS values and arithmetic mean of five seed-level binary failure indicators separately','between_unit_aggregation':'equal independent-unit means; never pool seeds as independent observations','difference_direction':'first minus second; negative favorable','threshold_sensitivities_mm':[2,4,5],'threshold_sensitivities_scope':'descriptive, do not replace primary 3 mm'},
 'contrasts':{'primary':[['C_adaptive','C_lambda0.01']],'secondary_order':[['C_lambda0.01','B'],['C_adaptive','B'],['C_fixed','B'],['C_adaptive','C_fixed'],['C_lambda0.01','C_fixed']],'clean_contrasts':'same ordering, descriptive clean accuracy costs; separate from corrupted primary family'},
 'bootstrap':{'draws':10000,'seed':20261024,'rng':'numpy.random.default_rng','unit':'independent verified patient/subject; volume only when linkage cannot be established and labeled as such','patient_duplicates':'prefer one CT selected before outcomes; if multiple retained, average seeds within CT then CTs within patient, with one patient weight, and resample whole patient clusters','paired_indices_shared_across_methods_and_endpoints':True,'aggregation_before_bootstrap':True,'pointwise_quantiles':[0.025,0.975],'primary_family_bonferroni_quantiles':[0.0125,0.9875],'quantile_method':'numpy.quantile method=linear','no_new_hypothesis_tests':True,'historical_p_values':'not recomputed or presented as new confirmatory evidence','degenerate_bootstrap':'report zero empirical variance/number of discordant units; interval [0,0] from no observed failures does not establish zero population risk or equivalence','small_class_counts':'report units contributing failure/nonfailure and target counts; do not drop failed or constant bootstrap draws'},
 'missingness':{'geometry':'assess before optimization; count excluded units as zero pipeline success for all methods; no imputed RMS','transient_execution':'resume identical task with identical inputs/config/seeds; log retry; do not change tolerances or starts','persistent_execution_failure':'mark computational failure, not geometry; count as failed for composite all-assigned pipeline success; retain finite last/best pose only if frozen solver returns it; do not invent RMS or silently remove incomplete seeds','paired_numerical_rms':'use common complete eligible-unit set across all four methods and all five seeds; report exclusions and cause; if any persistent computational failures exist, label result conditional and provide completion counts rather than claim total-cohort accuracy','failures':'geometric centroid failure frequency among complete eligible targets and all-assigned pipeline failure reported separately','no_partial_seed_averages':True},
 'freeze_requirements':['final input candidate/assigned/eligible manifest and patient-linkage evidence','target detector policy decided','all source/config/seed/code SHA256 recorded','independent review of target/optimizer/candidate pairing/metrics','convergence cases selected before diagnostics','freeze receipt exists before any new-cohort outcome examination','amendments timestamped, justified and retain previous versions; no outcome-guided method/endpoint changes'],
 'convergence_selection_file':'convergence_selection.json',
 'convergence_unit_count':10,
 'claims':'No promise adaptive beats fixed0.01; report signs, magnitudes, uncertainty and failures regardless of direction. No clinical accuracy claim from synthetic targets.'
}
# Later root decisions are explicitly applied before writing the design snapshot.
protocol['cohorts']['new_public_siddon'].update(
 source_plan='TotalSegmentator revised v3 labels intersected with original v1 adult source IDs; sourceID/scans until linkage verified',
 full_source_order='Freeze all intersecting source IDs in salted-hash order before mask prefilter',
 candidate_prescreen='All five L1-L5 masks nonempty and no selected mask touching image boundary; first 60 complete candidates in frozen order',
 root_qc='CT/seg grid, affine, units, integrity and content/identity deduplication, then first 50 passing source-input QC in frozen candidate order',
 reserves='Remaining 10 are pre-registration input-completeness reserves only; no replacement for motion geometry or outcome',
 assigned_case_index='immutable final assigned order 0..49, frozen before motion generation; do not renumber after geometry exclusion')
protocol['targets'].update(
 new_case_index='immutable final assigned order 0..49, before motion generation; do not renumber after geometry exclusion',
 corruption_seed_formula={'old47':'20261024 + 1009 * original_all75_case_index + 12347 * repeat_index','new_public':'20361024 + 1009 * assigned_case_index + 12347 * repeat_index'},
 candidate_seed_formula={'old47':'20261025 + 1009 * original_all75_case_index + 7919 * k','new_public':'20361025 + 1009 * assigned_case_index + 7919 * k','k':'1..4; k=0 inherits newly optimized A'},
 motion_seed_formula={'old47':'reuse original saved GT; original base seed and original all75 index retained','new_public':'20361024 + 1009 * assigned_case_index, original ordered random draws and sign rule'},
 A_seed_formula={'old47':'20261024 + 1009 * original_all75_case_index','new_public':'20361024 + 1009 * assigned_case_index','per_CMA_start':'add original deterministic start index'},
 corruption_policy='controlled reference-based corruption: retain trilinear reference-derived masks, attenuation scales and per-view RNG start states; insert Siddon clean into the frozen attenuation/noise plan',
 old_corruption_reference='original retained trilinear clean512-to256 and original direct256 trilinear components',
 new_corruption_reference='generate original trilinear clean512-to256 and direct256 trilinear components at the frozen GT; references only define mask/scale/RNG, never replace Siddon optimization targets',
 corruption_provenance='save reference images, per-view masks, scales, RNG states and their hashes; Poisson/Gaussian values are generated for Siddon attenuation, not copied as residuals',
 corruption_limitation='references share trilinear geometry/model for corruption definition; disclose this controlled construction')
protocol['external_preprocessing']={'native_grid':'retain released CT/seg grid and physical affine; no resampling to improve outcomes','crop':'union L1-L5 plus 3 mm physical margin; preserve world coordinates','centering':'unweighted mean of five source vertebral centroids','proxy':'mask to five labels and clip(HU - 150, 0, 1800)','labels':'explicit source-label to internal20..24 mapping recorded','tuning':'none'}
protocol['convergence_review_flags']={'relative_l2_max_over_10cases_2views':0.005,'original_ncc_loss_max_over_10cases_2views':0.0001,'comparison':'Siddon512-to256 vs Siddon1024-to256 at saved old-development GT','scope':'engineering review flags only, neither clinical thresholds nor validation pass/fail','trigger_action':'inspect geometry/precision/aliasing; any denser production target setting requires a new design version based solely on old-development numerical evidence, without looking at evaluation method ranking'}
source_order_path=HERE.parent/'data_audit/frozen_candidate_order.json'
source_order=json.loads(source_order_path.read_text(encoding='utf8'))
protocol['cohorts']['new_public_siddon'].update(
 candidate_order_salt=source_order['order_salt'],
 candidate_order_algorithm=source_order['ordering'],
 authoritative_candidate_order_file='work/validation_20261026/data_audit/frozen_candidate_order.json',
 authoritative_candidate_order_sha256=sha256(source_order_path.read_bytes()).hexdigest(),
 source_order_frozen_utc=source_order['frozen_utc'],
 source_doi=source_order['source_doi'],
 original_cohort_doi=source_order['original_cohort_doi'],
 source_universe=source_order['universe'],
 selection_rule='Use the already frozen source-specific order verbatim; do not rerank using the provisional generic salt in design_v1',
 source_order_amendment='AMENDMENT_001_SOURCE_ORDER.md; correction before external registration outcomes, preserving the earlier data-order freeze')
protocol['external_preprocessing'].update(
 units_unknown_policy='Require per-case affine column norms1.5mm consistent with the published v1 cohort, valid CT/mask identical grid and unknown-or-mm header units; assign mm only to derived NIfTI copies, retain originals byte-identical',
 units_source_doi='10.1148/ryai.230024',
 units_source_exact_statement='All images were resampled to 1.5-mm isotropic resolution.',
 units_evidence_file='work/validation_20261026/data_audit/spatial_units_evidence.json')
protocol['midpoint_convergence']={
 'scope':'Old-development numerical ray-step diagnostic only; no registration or endpoint-based tuning',
 'case_selection_file':'convergence_selection.json',
 'detector_resolution':256,
 'views':['AP','LAT'],
 'requested_steps_mm':[1.0,0.5,0.25,0.125],
 'poses':['saved_ground_truth','saved_ground_truth_plus_predeclared_local_parameter_offset'],
 'perturbation':{'local_rotvec_x_add_deg':1.0,'local_tx_add_mm':1.0,'applies_to':'each of the five vertebrae; all other local rotation-vector/translation coordinates unchanged','definition':'add to original local parameter coordinates, not a composition with a global rigid transform'},
 'sample_cap_policy':'Choose a cap at least ceil(component physical diagonal / smallest requested h) for every tested component and assert no cap activation; record actual cap, N, diagonal and ray interval statistics',
 'comparison':'full-image neighboring-step relative L2 and original NCC loss, separately by case/view/pose; preserve intensity scale, detector, interpolation and integration bounds',
 'production_reference':'Also record whether uncapped h=1 N equals original h=1/cap192 production N for each component; any disagreement is a limitation, not silently treated as the production renderer',
 'outcome_use':'No new-cohort/evaluation registration results may select step, perturbation or acceptance criteria',
 'review_policy':'Describe numerical differences across all four h values. No retrospective success threshold or automatic optimizer change; any setting change requires a prospective protocol amendment using only old-development numerical evidence.',
 'amendment':'AMENDMENT_002_MIDPOINT.md'}
(HERE/'analysis_protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf8')
print(json.dumps({'convergence10':cases,'eligible_old_n':len(eligible),'status':protocol['status']}))
