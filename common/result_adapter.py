"""Normalize every suite's summary to one verdict-bearing schema."""
from __future__ import annotations
from typing import Any
from .artifacts import SCHEMA_VERSION

def normalize_summary(summary: dict[str, Any]) -> dict[str, Any]:
    out=dict(summary); out["source_schema_version"] = summary.get("schema_version")
    out["schema_version"] = SCHEMA_VERSION
    results = out.get("results") or out.get("records") or []
    normalized=[]
    for item in results:
        row=dict(item) if isinstance(item,dict) else {"value":item}
        legacy_passed=row.get("case_passed", row.get("passed"))
        legacy_status=row.get("case_status")
        # Adapters for common external verifier result shapes.  A raw HTTP/status
        # row is intentionally not upgraded: it still needs a domain verdict.
        if legacy_passed is None and "correct" in row and row.get("status") == "ok":
            legacy_passed = bool(row.get("correct")); legacy_status = "passed" if legacy_passed else "failed"
        if legacy_passed is None and "reward" in row and row.get("evaluated", True):
            legacy_passed = float(row.get("reward") or 0) >= 1; legacy_status = "passed" if legacy_passed else "failed"
        if legacy_passed is None and "score" in row and row.get("evaluated", True):
            legacy_passed = float(row.get("score") or 0) >= 1; legacy_status = "passed" if legacy_passed else "failed"
        legacy_status = legacy_status or ("passed" if legacy_passed is True else "failed" if legacy_passed is False else "not_evaluable")
        row["case_status"] = legacy_status
        row["case_passed"] = bool(legacy_passed is True and legacy_status == "passed")
        normalized.append(row)
    if results: out["results"] = normalized
    # Missing evidence or unknown verdict is never upgraded to pass.
    if any(r.get("case_status") in {"failed","error","not_evaluable","blocked"} for r in normalized): out["admission_passed"] = False
    out.setdefault("admission_passed", False)
    out["verdict_contract"] = "validator-derived; transport success is not a case verdict"
    return out
