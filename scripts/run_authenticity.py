import argparse,asyncio,os
from pathlib import Path
from authenticity import CASES,run
from volcperf.protocols import api_root
p=argparse.ArgumentParser();p.add_argument('case',choices=CASES);p.add_argument('--output-root',type=Path,default=Path('results'));a=p.parse_args()
key=os.getenv('VOLC_API_KEY');base=os.getenv('VOLC_BASE_URL');model=os.getenv('VOLC_MODEL')
if not key or not base or not model: raise SystemExit('缺少 VOLC_API_KEY/VOLC_BASE_URL/VOLC_MODEL')
base=api_root(os.getenv('VOLC_PROTOCOL','volcengine'),base)
path,summary=asyncio.run(run(a.case,base,key,model,a.output_root)); print(f'结果目录: {path.resolve()}'); raise SystemExit(0 if not summary['failed'] else 2)
