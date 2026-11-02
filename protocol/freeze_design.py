"""Freeze only the pre-outcome design layer; input and execution need later receipts."""
from pathlib import Path
import json,hashlib,datetime,shutil,argparse
HERE=Path(__file__).resolve().parent
parser=argparse.ArgumentParser();parser.add_argument('--version',type=int,default=1);args=parser.parse_args()
if args.version<1:raise ValueError('Positive version required')
TARGET=HERE/f'freezes/design_v{args.version}';TARGET.mkdir(parents=True,exist_ok=True)
names=['PROTOCOL.md','analysis_protocol.json','convergence_selection.json','build_protocol_manifest.py']
if args.version>=2:names.append('AMENDMENT_001_SOURCE_ORDER.md')
if args.version>=3:names.append('AMENDMENT_002_MIDPOINT.md')
hashes={n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in names}
receipt_path=TARGET/'receipt.json'
if receipt_path.exists():
    previous=json.loads(receipt_path.read_text(encoding='utf8'))
    if previous['sha256']!=hashes:raise RuntimeError('This design version is already frozen with different content. Create a numbered amendment, not an overwrite.')
    print(json.dumps(previous));raise SystemExit(0)
for n in names:shutil.copy2(HERE/n,TARGET/n)
receipt={'freeze_id':f'20261026-design-v{args.version}','frozen_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'layer':'design_and_source_selection' if args.version>=2 else 'design_only','status':'FROZEN_DESIGN_PENDING_FINAL_INPUT_AND_EXECUTION_LOCKS','decision_authority':'Root approved design choices and technical review flags in the current authorized experiment task','new_outcomes_used_to_select_design':False,'outcome_files_read_by_this_script':False,'sha256':hashes,'remaining_locks':['final patient/source linkage and overlap audit','final assigned list and explicit case_index mapping','new external preprocessing/runner independent code review','production runner/config and input-source checksums','old-development convergence diagnostic and any prospective target-sampling amendment'],'independence_claim':'not asserted until source/linkage/overlap evidence supports it','note':'This receipt freezes planned interpretation, estimands, intervals, seeds, target/corruption policy, candidate selection and ten old-development convergence cases. It does not certify completed experiments or human manuscript approval.'}
if args.version>=2:
    sources={}
    for name in ['frozen_candidate_order.json','spatial_units_evidence.json','patient_linkage_evidence.json']:
        p=HERE.parent/'data_audit'/name
        sources[name]=hashlib.sha256(p.read_bytes()).hexdigest();shutil.copy2(p,TARGET/name)
    receipt['source_selection_sha256']=sources
receipt_path.write_text(json.dumps(receipt,indent=2),encoding='utf8')
print(json.dumps(receipt))
