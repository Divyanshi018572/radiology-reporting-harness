"""Stage 2 merge: pure-Python template + findings JSON -> findings dict."""
import logging
from src.scorer import parse_report, norm_label

log = logging.getLogger(__name__)


def is_normal_dictation(dictation: str) -> bool:
    """Check dictation is bare 'normal', return bool."""
    import re
    return re.sub(r"[^a-z]", "", dictation.lower()) == "normal"


def apply_merge(template_content: str, findings: list[dict],
                case_id: str = "") -> tuple[dict[str, str], set[str]]:
    """Splice new clauses into template fields, return (fields, changed)."""
    fields, _ = parse_report(template_content)
    order = list(fields.keys())
    norm_to_orig = {norm_label(k): k for k in fields}
    by_field: dict[str, list[dict]] = {}
    for f in findings:
        nl = norm_label(str(f.get("field_label", "")))
        by_field.setdefault(nl, []).append(f)
    changed: set[str] = set()
    other_nl = "OTHER FINDINGS"
    for nl, items in by_field.items():
        if nl not in norm_to_orig:
            if other_nl in norm_to_orig:
                orig = norm_to_orig[other_nl]
                for it in items:
                    txt = str(it.get("new_clause_text", "")).strip()
                    if txt:
                        fields[orig] = (fields[orig] + " " + txt).strip()
                        changed.add(orig)
            else:
                log.warning("%s: dropping unmapped field %s", case_id, nl)
            continue
        orig = norm_to_orig[nl]
        cur = fields[orig]
        occ = []
        for it in items:
            span = str(it.get("matched_template_span", ""))
            try:
                occ.append((cur.index(span), it))
            except ValueError:
                log.warning("%s: span not found in %s, keeping field",
                            case_id, orig)
        occ.sort(key=lambda x: x[0])
        for _, it in occ:
            span = str(it.get("matched_template_span", ""))
            new = str(it.get("new_clause_text", ""))
            if span and span in cur:
                cur = cur.replace(span, new, 1)
                changed.add(orig)
        fields[orig] = cur.strip()
    return ({k: fields[k] for k in order}, changed)


def render_findings(fields: dict[str, str]) -> str:
    """Render ordered fields dict to FINDINGS block, return string."""
    lines = ["FINDINGS:"]
    for k, v in fields.items():
        if k == "_UNLABELLED_":
            continue
        lines.append(f"{k}: {v}".strip())
        lines.append("")
    return "\n".join(lines).strip()
