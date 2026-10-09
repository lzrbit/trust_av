"""Synthetic vector-PDF interface test; no manuscript figures are required."""
import hashlib,json,sys,tempfile
from pathlib import Path
from pypdf import PdfReader,PdfWriter
from pypdf.generic import DictionaryObject,NameObject,DecodedStreamObject
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'analysis'))
from export_main_panels import export,MM

def main():
 with tempfile.TemporaryDirectory() as tmp:
  root=Path(tmp);source=root/'synthetic.pdf';writer=PdfWriter();page=writer.add_blank_page(width=100*MM,height=80*MM)
  font=DictionaryObject({NameObject('/Type'):NameObject('/Font'),NameObject('/Subtype'):NameObject('/Type1'),NameObject('/BaseFont'):NameObject('/Helvetica')})
  page[NameObject('/Resources')]=DictionaryObject({NameObject('/Font'):DictionaryObject({NameObject('/F1'):writer._add_object(font)})})
  stream=DecodedStreamObject();stream.set_data(b'0 0 1 rg 0 0 80 80 re f\nBT /F1 10 Tf 1 0 0 1 12 24 Tm (keep label) Tj ET\nBT /F1 10 Tf 1 0 0 1 30 60 Tm (foreign heading) Tj ET\n');page[NameObject('/Contents')]=writer._add_object(stream);writer.write(source)
  digest=hashlib.sha256(source.read_bytes()).hexdigest()
  config={'source_sha256':{'synthetic.pdf':digest},'panels':[{'id':'synthetic','file':'piece.pdf','source_pdf':'synthetic.pdf','canvas_mm':[44,44],'components':[{'crop_mm':[0,0,40,40],'at_mm':[2,2],'exclude_text_prefixes':['foreign']}]}]}
  cfg=root/'config.json';cfg.write_text(json.dumps(config));result=export(root,cfg,root/'out');page=PdfReader(root/'out/piece.pdf').pages[0]
  assert result['export_count']==1 and not list(page.images)
  assert abs(float(page.mediabox.width)/MM-44)<1e-6 and abs(float(page.mediabox.height)/MM-44)<1e-6
  assert 'keep label' in page.extract_text() and 'foreign heading' not in page.extract_text()
  assert result['exports'][0]['components_before_whitespace_trim'][0]['scale']==1
  assert hashlib.sha256(source.read_bytes()).hexdigest()==digest
  config['source_sha256']['synthetic.pdf']='0'*64;cfg.write_text(json.dumps(config))
  try:export(root,cfg,root/'rejected')
  except ValueError as error:assert 'hash mismatch' in str(error)
  else:raise AssertionError('A mismatched source was not rejected')
 print(json.dumps({'status':'pass','input':'synthetic vector PDF','checks':['millimetre dimensions and native scale','vector-only output','explicit foreign-text exclusion','source immutability','frozen-parent hash mismatch rejected']},indent=2))
if __name__=='__main__':main()
