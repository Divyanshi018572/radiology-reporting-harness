"""Stage 0 data layer: load + validate csvs, resolve paths, train/val split."""
import os
from pathlib import Path
import pandas as pd

TRAIN_COLS = ["case_id", "modality", "body_part", "study_description",
              "patient_age_band", "patient_sex", "template_content",
              "dictation", "report"]
TEST_COLS = [c for c in TRAIN_COLS if c != "report"]


def resolve_data_path(filename: str) -> Path:
    """Search local data/ then Kaggle input dirs, return existing path."""
    name = Path(filename).name
    direct = Path(filename)
    if direct.exists() and direct.is_file():
        return direct
    candidates = [Path("data") / name,
                  Path("/kaggle/input") / name,
                  Path("/kaggle/input/radiometry") / name]
    for p in candidates:
        if p.exists():
            return p
    for base in [Path("/kaggle/input")]:
        if base.exists():
            hits = list(base.rglob(name))
            if hits:
                return hits[0]
    return Path("data") / name


def load_csv(filename: str, expected_cols: list[str]) -> pd.DataFrame:
    """Load csv and assert required columns exist, return dataframe."""
    path = resolve_data_path(filename)
    df = pd.read_csv(str(path))
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{filename} missing cols {missing} at {path}")
    if df["case_id"].duplicated().any():
        raise ValueError(f"{filename} has duplicate case_id")
    return df


def train_val_split(df: pd.DataFrame, frac: float = 0.2,
                    seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified-by-modality split if possible, else random; input -> (train, val)."""
    try:
        train_parts, val_parts = [], []
        for _, g in df.groupby("modality"):
            v = g.sample(frac=frac, random_state=seed)
            train_parts.append(g.drop(v.index))
            val_parts.append(v)
        import pandas as pd
        return (pd.concat(train_parts).sample(frac=1.0, random_state=seed)
                .reset_index(drop=True),
                pd.concat(val_parts).sample(frac=1.0, random_state=seed)
                .reset_index(drop=True))
    except (ValueError, KeyError):
        v = df.sample(frac=frac, random_state=seed)
        return df.drop(v.index).reset_index(drop=True), \
            v.reset_index(drop=True)


def save_split(train_df: pd.DataFrame, val_df: pd.DataFrame,
               out: str = "outputs/split.json") -> None:
    """Persist val case_ids for reproducibility, return None."""
    import json
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump({"train_ids": train_df["case_id"].tolist(),
                   "val_ids": val_df["case_id"].tolist(), "seed": 42}, f)
