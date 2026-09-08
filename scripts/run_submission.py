"""Run pipeline on test.csv, validate all rows, write submission.csv."""
import pandas as pd
from src.data_loader import load_csv, TEST_COLS
from src.pipeline import generate_report
from src.validator import validate_report

if __name__ == "__main__":
    te = load_csv("test.csv", TEST_COLS)
    assert len(te) == 132 and te["case_id"].nunique() == 132
    reports = []
    for _, r in te.iterrows():
        try:
            gen = generate_report(r)
        except Exception as e:
            print(f"FAIL {r['case_id']}: {e}")
            gen = str(r["template_content"])
        ok, reason = validate_report(gen, str(r["template_content"]))
        if not ok:
            print(f"VALIDATOR FAIL {r['case_id']}: {reason}")
            gen = str(r["template_content"])
        reports.append({"case_id": r["case_id"], "report": gen})
    sub = pd.DataFrame(reports)
    assert len(sub) == len(te) == 132 and sub["case_id"].nunique() == 132
    assert list(sub.columns) == ["case_id", "report"]
    sub.to_csv("outputs/submission.csv", index=False)
    print("wrote outputs/submission.csv", len(sub))
