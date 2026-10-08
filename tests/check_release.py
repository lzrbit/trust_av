"""Check the publishable tree for accidental data/credential/path inclusion.

This is a disclosure guard, not a guarantee that arbitrary future data releases
are anonymous. Review every changed data file before publication.
"""
from pathlib import Path
import csv,json,re,subprocess
ROOT=Path(__file__).resolve().parents[1]
def main():
 found=subprocess.run(['git','ls-files','--cached','--others','--exclude-standard','-z'],cwd=ROOT,capture_output=True,text=True,check=True)
 paths=[ROOT/n for n in found.stdout.split('\0') if n]
 violations=[];csvs=[]
 forbidden={'.parquet','.joblib','.pkl','.pickle','.npz','.xlsx','.xls','.mat','.pem','.key','.pdf','.png','.svg'}
 patterns=[r'/Users/[^\s]+',r'/home/[^\s]+',r'C:\\Users\\',r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',r'github_pat_[A-Za-z0-9_]{20,}',r'ghp_[A-Za-z0-9]{20,}',r'sk-[A-Za-z0-9_-]{25,}',r'xox[baprs]-[A-Za-z0-9-]{20,}']
 protected={'participant_id','source_row_id','window_row_id','event_row_id','fliers_json','flier_index','participant_mean_risk'}
 for p in paths:
  if p.suffix.lower() in forbidden:violations.append({'file':str(p.relative_to(ROOT)),'reason':'participant/cache/key file type'})
  if p.is_symlink():violations.append({'file':str(p.relative_to(ROOT)),'reason':'unexpected symlink'})
  text=p.read_text(errors='replace')
  # The scanner's own regex definitions are expected; actual credentials are not.
  if p.resolve()!=Path(__file__).resolve():
   for pattern in patterns:
    if re.search(pattern,text):violations.append({'file':str(p.relative_to(ROOT)),'reason':'sensitive literal pattern'})
  if p.suffix=='.csv':
   with p.open(newline='') as f:rows=list(csv.DictReader(f))
   cols=set(rows[0]) if rows else set()
   if cols&protected:violations.append({'file':str(p.relative_to(ROOT)),'reason':'individual/linkage/outlier columns'})
   csvs.append({'file':str(p.relative_to(ROOT)),'rows':len(rows),'columns':sorted(cols)})
 report={'status':'pass' if not violations else 'fail','candidate_files':len(paths),'csv_tables':csvs,'violations':violations,'scope':'Literal/extension/key checks plus explicit aggregate review required; no private Git internals are read.'}
 print(json.dumps(report,indent=2));raise SystemExit(bool(violations))
if __name__=='__main__':main()
