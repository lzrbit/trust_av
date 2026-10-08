#!/usr/bin/env python3
"""Explicit portable entry point. No data download or implicit model fitting."""
import argparse,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
STAGES={
 'build':'build_data.py', 'describe':'describe_round2.py',
 'risk-distribution':'risk_distribution.py', 'dynamics':'analyse_dynamics.py',
 'country':'country_standardization.py', 'risk-models':'model_shap.py',
 'trust-models':'trust_prediction.py', 'trust-contrasts':'trust_coefficient_contrasts.py',
 'multiplicity':'multiplicity_sensitivity.py', 'mixed':'mixed_components.py',
 'prepost':'harmonize_prepost.py', 'parameters':'analyse_parameters.py',
 'differential-trust':'analyse_trust_contrast.py', 'whatif':'analyse_whatif.py',
 'hb-tests':'analyse_hb_tests.py', 'hb-boxes':'build_hb_boxstats.py', 'aggregate-figures':'plot_aggregates.py',
}
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('stages',nargs='*',choices=list(STAGES))
 p.add_argument('--workspace',type=Path,default=ROOT/'work',help='Private output workspace, never intended for publication.')
 p.add_argument('--data-dir',type=Path,help='Private directory containing derived parquet inputs.')
 p.add_argument('--questionnaire',type=Path,help='Original licensed questionnaire workbook; not included.')
 p.add_argument('--dry-run',action='store_true');p.add_argument('--list',action='store_true')
 args=p.parse_args()
 if args.list:
  for k,v in STAGES.items():print(f'{k:22s} analysis/{v}')
  return
 if not args.stages:p.error('Specify one or more stages, or --list. See README for dependencies.')
 env=os.environ.copy();env['TRUST_AV_WORKSPACE']=str(args.workspace.expanduser().resolve())
 if args.data_dir:env['TRUST_AV_DATA_DIR']=str(args.data_dir.expanduser().resolve())
 if args.questionnaire:env['TRUST_AV_QUESTIONNAIRE']=str(args.questionnaire.expanduser().resolve())
 for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:env[key]='2'
 env['MPLBACKEND']='Agg';env['PYTHONDONTWRITEBYTECODE']='1'
 for stage in args.stages:
  command=[sys.executable,str(ROOT/'analysis'/STAGES[stage])]
  if stage=='risk-models':command+=['--stage','all']
  print(json.dumps({'stage':stage,'command':command}),flush=True)
  if args.dry_run:continue
  logdir=args.workspace/'logs';logdir.mkdir(parents=True,exist_ok=True)
  with (logdir/(stage+'.log')).open('w') as stream:
   completed=subprocess.run(command,env=env,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
  if completed.returncode:raise SystemExit(f'{stage} failed; see {logdir/(stage+".log")}')
  print(f'{stage} completed',flush=True)
if __name__=='__main__':main()
