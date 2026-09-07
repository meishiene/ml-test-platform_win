from __future__ import annotations
import argparse, asyncio, json, os, time
from pathlib import Path
from .cases import CASES
from .core import GLM53Runner, GLMConfig, summarize_results, write_report

def main(argv=None):
 p=argparse.ArgumentParser(description="运行 GLM 5.3 专项测试"); p.add_argument("case",choices=sorted(CASES)); p.add_argument("--output-root",type=Path,default=Path("results")); a=p.parse_args(argv)
 key=os.getenv("GLM_API_KEY") or os.getenv("VOLC_API_KEY"); base=os.getenv("GLM_BASE_URL") or os.getenv("VOLC_BASE_URL")
 model=os.getenv("GLM_MODEL") or os.getenv("VOLC_MODEL")
 if not key or not base or not model: raise ValueError("缺少 GLM_API_KEY/GLM_BASE_URL/GLM_MODEL")
 out=a.output_root/f"{a.case}-{time.strftime('%Y%m%d-%H%M%S')}"
 async def go():
  async with GLM53Runner(GLMConfig(key,base,verify_ssl=os.getenv("GLM_VERIFY_SSL","true").lower() not in {"0","false"}), audit_dir=out) as r:
   s=CASES[a.case]; results=await r.run(model,s.priorities)
  summary=summarize_results(results); summary.update({"record_id":os.getenv("VOLC_RECORD_ID"),"test_kind":"glm","case":a.case,"model":model,"endpoint":base,"admission_passed":summary["all_passed"]})
  write_report(out,summary); print(json.dumps(summary,ensure_ascii=False,indent=2)); return 0 if summary["admission_passed"] else 2
 return asyncio.run(go())
