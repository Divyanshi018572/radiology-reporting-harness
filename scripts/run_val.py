"""Run pipeline on val split, compute RES, save analysis CSV."""
import pandas as pd
from src.data_loader import load_csv, train_val_split, save_split
from src.pipeline import generate_report
from src.scorer import res_case, mean_res

if __name__ == "__main__":
    df = load_csv("train.csv", ["case_id", "modality", "body_part",
                                "study_description", "patient_age_band",
                                "patient_sex", "template_content",
                                "dictation", "report"])
    tr, val = train_val_split(df)
    save_split(tr, val)
    done_path = "outputs/val_results.csv"
    try:
        done = pd.read_csv(done_path)["case_id"].tolist()
    except (FileNotFoundError, pd.errors.EmptyDataError):
        done = []
    rows = []
    for _, r in val.iterrows():
        if r["case_id"] in done:
            continue
        try:
            gen = generate_report(r)
        except Exception as e:
            print(f"FAIL {r['case_id']}: {e}")
            gen = str(r["template_content"])
        s = res_case(str(r["report"]), gen, str(r["template_content"]))
        pd.DataFrame([{"case_id": r["case_id"], "RES": s["RES"], "F": s["F"],
                       "I": s["I"], "generated": gen,
                       "reference": r["report"]}]).to_csv(
            done_path, mode="a", index=False, header=not done)
        done.append(r["case_id"])
    out = pd.read_csv(done_path).drop_duplicates("case_id", keep="first")
    print(f"rows: {len(out)}/{len(val)}")
    print("mean RES:", out["RES"].mean())
