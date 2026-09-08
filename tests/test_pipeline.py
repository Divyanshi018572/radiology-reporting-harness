"""Tests for pipeline fallback: LLM down -> template copy, still valid."""
import pandas as pd
from src import pipeline
from src.validator import validate_report

ROW = pd.Series({"case_id": "c1",
                 "template_content": "FINDINGS:\nLUNGS: Clear.\n\n"
                                     "IMPRESSION:\nNormal.",
                 "dictation": "big mass in lung"})


def test_extraction_failure_falls_back_to_valid_template(monkeypatch):
    monkeypatch.setattr(pipeline, "extract_findings",
                        lambda row: (_ for _ in ()).throw(IOError("down")))
    gen = pipeline.generate_report(ROW)
    assert gen == ROW["template_content"]
    assert validate_report(gen, ROW["template_content"])[0]
