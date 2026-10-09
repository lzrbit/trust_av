#!/usr/bin/env python3
"""Explicit portable entry point. No data download or implicit model fitting."""
import argparse,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
SPECS=json.loads((ROOT/'config/analysis_stages.json').read_text())['stages']
STAGES={r['id']:Path(r['script']).name for r in SPECS}

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('stages',nargs='*',choices=list(STAGES))
 p.add_argument('--workspace',type=Path,default=ROOT/'work',help='Private output workspace, never intended for publication.')
 p.add_argument('--data-dir',type=Path,help='Private directory containing derived parquet inputs.')
 p.add_argument('--parameter-xlsx',type=Path,help='Optional original stimulus parameter workbook, read-only.')
 p.add_argument('--questionnaire',type=Path,help='Original licensed questionnaire workbook; not included.')
 p.add_argument('--dry-run',action='store_true');p.add_argument('--list',action='store_true')
 args=p.parse_args()
 if args.list:
  for k,v in STAGES.items():print(f'{k:22s} analysis/{v}')
  return
 if not args.stages:p.error('Specify one or more stages, or --list. See README for dependencies.')
 env=os.environ.copy();env['TRUST_AV_WORKSPACE']=str(args.workspace.expanduser().resolve())
 if args.data_dir:env['TRUST_AV_DATA_DIR']=str(args.data_dir.expanduser().resolve())
 if args.parameter_xlsx:env['TRUST_AV_PARAMETER_XLSX']=str(args.parameter_xlsx.expanduser().resolve())
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
