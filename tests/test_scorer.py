"""Tests for scorer: normalization, weights, RES behavior."""
from src.scorer import normalize_text, weighted_edit, res_case


def test_signed_number_and_hyphen():
    toks = normalize_text("-2 mm air-space opacity")
    assert "-2" in toks, toks
    assert "airspace" in toks, toks


def test_units_standardized():
    assert "mm" in normalize_text("5 millimeters nodule")
    assert "cm" in normalize_text("2 centimeters mass")


def test_identical_is_zero():
    assert weighted_edit(["no", "effusion"], ["no", "effusion"]) == 0.0


def test_critical_weights_more_than_function():
    base = ["small", "right", "effusion", "in", "the", "lung"]
    crit = ["small", "left", "effusion", "in", "the", "lung"]
    func = ["small", "right", "effusion", "in", "a", "lung"]
    assert weighted_edit(base, crit) > weighted_edit(base, func)


def test_worked_chest_example():
    tpl = ("FINDINGS:\nSUPPORT DEVICES: None.\nLUNGS: No focal airspace "
           "opacity or pulmonary edema.\nPLEURA: No pleural effusion or "
           "pneumothorax.\n\nIMPRESSION:\nNo acute cardiopulmonary "
           "abnormality.")
    ref = ("FINDINGS:\nSUPPORT DEVICES: None.\nLUNGS: Mild right basilar "
           "airspace opacity. No pulmonary edema.\nPLEURA: Small right "
           "pleural effusion. No pneumothorax.\n\nIMPRESSION:\nMild right "
           "basilar airspace opacity and small right pleural effusion.")
    good = res_case(ref, ref, tpl)
    bad = res_case(ref, tpl, tpl)
    assert good["RES"] == 0.0
    assert good["RES"] < bad["RES"]
    assert bad["RES"] > 0.3


def test_wrong_field_worse():
    tpl = "FINDINGS:\nLUNGS: No opacity.\nPLEURA: No effusion.\n\n" \
          "IMPRESSION:\nNormal."
    ref = "FINDINGS:\nLUNGS: No opacity.\nPLEURA: Small effusion.\n\n" \
          "IMPRESSION:\nSmall effusion."
    right = "FINDINGS:\nLUNGS: No opacity.\nPLEURA: Small effusion.\n\n" \
            "IMPRESSION:\nSmall effusion."
    wrong = "FINDINGS:\nLUNGS: Small effusion.\nPLEURA: No effusion.\n\n" \
            "IMPRESSION:\nSmall effusion."
    assert res_case(ref, right, tpl)["RES"] < res_case(ref, wrong, tpl)["RES"]


def test_extra_field_penalized():
    tpl = "FINDINGS:\nLUNGS: Clear.\n\nIMPRESSION:\nNormal."
    ref = "FINDINGS:\nLUNGS: Clear.\n\nIMPRESSION:\nNormal."
    clean = "FINDINGS:\nLUNGS: Clear.\n\nIMPRESSION:\nNormal."
    extra = "FINDINGS:\nLUNGS: Clear.\nEXTRA: Big mass here.\n\n" \
            "IMPRESSION:\nNormal."
    assert res_case(ref, extra, tpl)["RES"] > res_case(ref, clean, tpl)["RES"]
