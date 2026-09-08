"""Stage 4 validator: structural + R3/R4 checks."""
from src.scorer import parse_report, norm_label
import re


def validate_report(report: str, template_content: str,
                    changed: set[str] | None = None) -> tuple[bool, str | None]:
    """Validate report structure and preservation, return (pass, reason)."""
    if len(re.findall(r"(?im)^findings\s*:", report)) != 1:
        return False, "must contain exactly one FINDINGS:"
    if len(re.findall(r"(?im)^impression\s*:", report)) != 1:
        return False, "must contain exactly one IMPRESSION:"
    if re.search(r"```", report):
        return False, "markdown fences forbidden"
    sub_f, _ = parse_report(report)
    tpl_f, _ = parse_report(template_content)
    sub_labels = [norm_label(k) for k in sub_f if k != "_UNLABELLED_"]
    tpl_labels = [norm_label(k) for k in tpl_f]
    if set(sub_labels) != set(tpl_labels):
        return False, f"label set mismatch {set(tpl_labels) ^ set(sub_labels)}"
    if sub_labels != tpl_labels:
        return False, "label order mismatch"
    if "_UNLABELLED_" in sub_f and sub_f["_UNLABELLED_"].strip():
        return False, "unlabelled FINDINGS content"
    if changed is not None:
        for k in tpl_f:
            if k not in changed and sub_f.get(k, "").strip() != \
                    tpl_f[k].strip():
                return False, f"untouched field rewritten: {k}"
    return True, None
