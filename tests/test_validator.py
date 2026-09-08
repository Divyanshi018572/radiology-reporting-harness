"""Tests for validator: structure, labels, order, R3."""
from src.validator import validate_report

TPL = ("FINDINGS:\nLUNGS: Clear.\nPLEURA: No effusion.\n\n"
       "IMPRESSION:\nNormal.")
GOOD = ("FINDINGS:\nLUNGS: Clear.\nPLEURA: No effusion.\n\n"
        "IMPRESSION:\nNormal.")


def test_good_passes():
    assert validate_report(GOOD, TPL, set())[0]


def test_missing_field():
    bad = "FINDINGS:\nLUNGS: Clear.\n\nIMPRESSION:\nNormal."
    assert not validate_report(bad, TPL)[0]


def test_extra_field():
    bad = GOOD.replace("IMPRESSION:", "EXTRA: x.\n\nIMPRESSION:")
    assert not validate_report(bad, TPL)[0]


def test_wrong_order():
    bad = ("FINDINGS:\nPLEURA: No effusion.\nLUNGS: Clear.\n\n"
           "IMPRESSION:\nNormal.")
    assert not validate_report(bad, TPL)[0]


def test_malformed():
    assert not validate_report("just text", TPL)[0]


def test_untouched_rewrite_flagged():
    bad = GOOD.replace("LUNGS: Clear.", "LUNGS: Clear lungs!!")
    assert not validate_report(bad, TPL, set())[0]


def test_blank_lines_and_numbering_allowed():
    ok = ("FINDINGS:\nLUNGS: Clear.\n\nPLEURA: No effusion.\n\n"
          "IMPRESSION:\n1. Normal.")
    assert validate_report(ok, TPL, set())[0]
