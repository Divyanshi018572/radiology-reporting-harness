"""Tests for merge: preservation, splice, ordering, fallback, OTHER routing."""
from src.merge import apply_merge

TPL = ("FINDINGS:\nLUNGS: No focal airspace opacity or pulmonary edema.\n"
       "PLEURA: No pleural effusion or pneumothorax.\n"
       "OTHER FINDINGS:\n\nIMPRESSION:\nNormal.")


def test_untouched_preserved():
    fields, changed = apply_merge(TPL, [], case_id="t")
    assert changed == set()
    assert "pulmonary edema" in fields["PLEURA"] or "pneumothorax" in \
        fields["PLEURA"]


def test_touched_replaced_with_verification():
    f = [{"field_label": "PLEURA",
          "matched_template_span": "No pleural effusion or pneumothorax.",
          "unaffected_subclause": "No pneumothorax.",
          "new_clause_text": "Small right pleural effusion. "
                             "No pneumothorax."}]
    fields, changed = apply_merge(TPL, f, case_id="t")
    assert "PLEURA" in changed
    assert "Small right pleural effusion" in fields["PLEURA"]
    assert "LUNGS" not in changed


def test_partial_span_splice_preserves_sibling_verbatim():
    tpl = ("FINDINGS:\nLUNGS: No focal airspace opacity or pulmonary "
           "edema.\n\nIMPRESSION:\nNormal.")
    f = [{"field_label": "LUNGS",
          "matched_template_span": "No focal airspace opacity",
          "unaffected_subclause": "or pulmonary edema.",
          "new_clause_text": "Mild right basilar airspace opacity"}]
    fields, changed = apply_merge(tpl, f, case_id="t")
    assert fields["LUNGS"] == "Mild right basilar airspace opacity or " \
        "pulmonary edema."


def test_span_failure_fallback():
    f = [{"field_label": "PLEURA", "matched_template_span": "NOT IN TEMPLATE",
          "unaffected_subclause": "", "new_clause_text": "Bad edit."}]
    fields, changed = apply_merge(TPL, f, case_id="t")
    assert "PLEURA" not in changed
    assert "Bad edit" not in fields["PLEURA"]


def test_unmapped_routed_to_other():
    f = [{"field_label": "BRAIN", "matched_template_span": "",
          "unaffected_subclause": "", "new_clause_text": "Small cyst."}]
    fields, changed = apply_merge(TPL, f, case_id="t")
    assert "Small cyst" in fields["OTHER FINDINGS"]


def test_unmapped_dropped_without_other():
    tpl2 = "FINDINGS:\nLUNGS: Clear.\n\nIMPRESSION:\nNormal."
    f = [{"field_label": "BRAIN", "matched_template_span": "",
          "unaffected_subclause": "", "new_clause_text": "Small cyst."}]
    fields, _ = apply_merge(tpl2, f, case_id="t")
    assert all("Small cyst" not in v for v in fields.values())
