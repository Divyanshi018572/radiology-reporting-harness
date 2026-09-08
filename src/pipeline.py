"""Orchestrator: Stage1 -> Stage2 -> Stage3 -> assemble -> Stage4 validate."""
import logging
import pandas as pd
from src.merge import apply_merge, render_findings, is_normal_dictation
from src.extract import extract_findings
from src.impression import write_impression
from src.validator import validate_report
from src.scorer import parse_report

log = logging.getLogger(__name__)


def generate_report(row: pd.Series) -> str:
    """Generate one FINDINGS+IMPRESSION report, return string."""
    template = str(row["template_content"])
    _, tpl_imp = parse_report(template)
    if is_normal_dictation(str(row["dictation"])):
        return template if "IMPRESSION:" in template.upper() else \
            f"{template}\n\nIMPRESSION:\n{tpl_imp}"
    try:
        findings = extract_findings(row)
    except Exception as e:
        log.warning("%s: extraction failed %s, fallback", row["case_id"], e)
        return template
    fields, changed = apply_merge(template, findings,
                                  case_id=str(row["case_id"]))
    changed_texts = [f"{k}: {fields[k]}" for k in changed]
    try:
        imp = write_impression(changed_texts, tpl_imp)
    except Exception as e:
        log.warning("%s: impression failed %s", row["case_id"], e)
        imp = tpl_imp
    report = f"{render_findings(fields)}\n\nIMPRESSION:\n{imp.strip()}"
    ok, reason = validate_report(report, template, changed)
    if not ok:
        log.warning("%s: validator fail %s, retry once", row["case_id"],
                    reason)
        try:
            findings = extract_findings(row)
            fields, changed = apply_merge(template, findings,
                                          case_id=str(row["case_id"]))
            imp = write_impression([f"{k}: {fields[k]}" for k in changed],
                                   tpl_imp)
            report = f"{render_findings(fields)}\n\nIMPRESSION:\n{imp.strip()}"
            ok2, _ = validate_report(report, template, changed)
            if ok2:
                return report
        except Exception as e:
            log.warning("%s: retry failed %s", row["case_id"], e)
        return template
    return report
