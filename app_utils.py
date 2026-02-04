import os
import pandas as pd
from joblib import load

MODEL_PATH = "dropout_binary_model_option1.joblib"
FEATURES_PATH = "dropout_model_features_option1.joblib"
DATASET_PATH = "dataset.csv"


def load_artifacts():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Missing {MODEL_PATH}")
    if not os.path.exists(FEATURES_PATH):
        raise FileNotFoundError(f"Missing {FEATURES_PATH}")

    model = load(MODEL_PATH)
    feature_list = load(FEATURES_PATH)
    return model, feature_list


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns
        .str.strip()
        .str.replace(" ", "_")
        .str.replace("/", "_")
        .str.replace("(", "", regex=False)
        .str.replace(")", "", regex=False)
        .str.replace("'", "", regex=False)
    )
    df = df.rename(columns={
        "Nacionality": "Nationality",
        "Age_at_enrollment": "Age"
    })
    return df


def add_feature_engineering_for_defaults(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    pairs = [
        ("Curricular_units_1st_sem_evaluations", "Curricular_units_2nd_sem_evaluations", "avg_evaluations"),
        ("Curricular_units_1st_sem_approved", "Curricular_units_2nd_sem_approved", "avg_approved"),
        ("Curricular_units_1st_sem_grade", "Curricular_units_2nd_sem_grade", "avg_grade"),
    ]
    for c1, c2, out in pairs:
        if c1 in df.columns and c2 in df.columns:
            df[out] = df[[c1, c2]].mean(axis=1)
    return df


def load_defaults():
    if not os.path.exists(DATASET_PATH):
        return None
    df = pd.read_csv(DATASET_PATH)
    df = clean_columns(df)
    df = add_feature_engineering_for_defaults(df)
    return df


def default_num(df_defaults, col, fallback_min, fallback_max, fallback_val):
    if df_defaults is None or col not in df_defaults.columns:
        return fallback_min, fallback_max, fallback_val

    s = pd.to_numeric(df_defaults[col], errors="coerce").dropna()
    if s.empty:
        return fallback_min, fallback_max, fallback_val

    mn, mx, med = float(s.min()), float(s.max()), float(s.median())
    if mn == mx:
        return mn, mx, mn
    return mn, mx, med


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def gpa_to_grade20(gpa: float) -> float:
    # Matches your example: 3.0→16, 2.0→12
    return clamp(gpa * 4.0 + 4.0, 0.0, 20.0)


def build_comparison_df(df_defaults, user_vals):
    if df_defaults is None:
        return None

    compare_cols = ["avg_evaluations", "avg_approved", "avg_grade", "Age"]
    rows = []

    for c in compare_cols:
        if c in df_defaults.columns:
            median = pd.to_numeric(df_defaults[c], errors="coerce").median()
            rows.append({
                "Feature": c,
                "User": float(user_vals[c]),
                "Dataset median": float(median)
            })

    return pd.DataFrame(rows) if rows else None


def safe_feature_importance_df(model, feature_list):
    if not hasattr(model, "feature_importances_"):
        return None

    imp = pd.Series(model.feature_importances_, index=feature_list).sort_values(ascending=False)
    out = imp.reset_index()
    out.columns = ["Feature", "Importance"]
    return out


def numeric_cols_only(df: pd.DataFrame):
    if df is None or df.empty:
        return []
    cols = []
    for c in df.columns:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().sum() > 0:
            cols.append(c)
    return cols
