import json
import pickle
from pathlib import Path
from typing import Any

import pandas as pd


# =========================
# Paths
# =========================

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"


# =========================
# Helpers
# =========================

def risk_label_from_prob(prob: float) -> str:
    if prob < 0.15:
        return "Low"
    elif prob < 0.35:
        return "Elevated"
    elif prob < 0.60:
        return "Moderate"
    elif prob < 0.80:
        return "High"
    return "Severe"


def load_pickle(path: Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def load_json(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


# =========================
# Artifact Loading
# =========================

def load_model_artifacts(models_dir: Path = MODELS_DIR) -> dict:
    feature_table = pd.read_parquet(models_dir / "feature_table.parquet")

    horizons = [3, 6, 12]
    artifacts = {
        "feature_table": feature_table,
        "models": {},
        "selected_features": {},
        "thresholds": {},
        "metrics": {},
        "horizons": horizons,
    }

    for h in horizons:
        artifacts["models"][h] = load_pickle(models_dir / f"lightgbm_{h}m.pkl")
        artifacts["selected_features"][h] = load_pickle(models_dir / f"selected_features_{h}m.pkl")
        artifacts["thresholds"][h] = load_json(models_dir / f"threshold_{h}m.json")["threshold"]
        artifacts["metrics"][h] = load_json(models_dir / f"metrics_{h}m.json")

    return artifacts


ARTIFACTS = load_model_artifacts()


# =========================
# Core Prediction Functions
# =========================

def get_available_date_on_or_before(input_date: str | pd.Timestamp) -> pd.Timestamp:
    input_date = pd.to_datetime(input_date)

    feature_table = ARTIFACTS["feature_table"]
    available_dates = feature_table.index[feature_table.index <= input_date]

    if len(available_dates) == 0:
        raise ValueError("No available data on or before the requested date.")

    return available_dates[-1]


def get_recession_risk_for_date(input_date: str | pd.Timestamp) -> dict:
    chosen_date = get_available_date_on_or_before(input_date)
    feature_table = ARTIFACTS["feature_table"]

    full_row = feature_table.loc[[chosen_date]]

    out = {
        "ReferenceDateUsed": chosen_date.date().isoformat()
    }

    for h in ARTIFACTS["horizons"]:
        model = ARTIFACTS["models"][h]
        selected_features = ARTIFACTS["selected_features"][h]
        threshold = ARTIFACTS["thresholds"][h]

        x_row = full_row[selected_features]
        prob = float(model.predict_proba(x_row)[:, 1][0])

        out[f"Risk_{h}M"] = prob
        out[f"Label_{h}M"] = risk_label_from_prob(prob)
        out[f"AboveThreshold_{h}M"] = bool(prob >= threshold)
        out[f"Threshold_{h}M"] = float(threshold)

    p3 = out["Risk_3M"]
    p6 = out["Risk_6M"]
    p12 = out["Risk_12M"]

    if p12 >= 0.60 and p3 < 0.35:
        out["OverallInterpretation"] = "Elevated long-term risk"
    elif p6 >= 0.60 or p3 >= 0.60:
        out["OverallInterpretation"] = "High near-term risk"
    elif max(p3, p6, p12) >= 0.35:
        out["OverallInterpretation"] = "Moderate recession risk"
    else:
        out["OverallInterpretation"] = "Low recession risk"

    return out


# =========================
# Historical Series Helpers
# =========================

def predict_probability_series(horizon: int) -> pd.DataFrame:
    if horizon not in ARTIFACTS["horizons"]:
        raise ValueError(f"Horizon must be one of {ARTIFACTS['horizons']}")

    feature_table = ARTIFACTS["feature_table"]
    model = ARTIFACTS["models"][horizon]
    selected_features = ARTIFACTS["selected_features"][horizon]
    threshold = ARTIFACTS["thresholds"][horizon]

    x = feature_table[selected_features].copy()
    probs = model.predict_proba(x)[:, 1]

    result = pd.DataFrame({
        "PredictedProb": probs,
    }, index=feature_table.index)

    result["Threshold"] = threshold
    result["RiskLabel"] = result["PredictedProb"].apply(risk_label_from_prob)

    return result


def get_all_probability_series() -> pd.DataFrame:
    feature_table = ARTIFACTS["feature_table"]
    out = pd.DataFrame(index=feature_table.index)

    for h in ARTIFACTS["horizons"]:
        series_df = predict_probability_series(h)
        out[f"Risk_{h}M"] = series_df["PredictedProb"]
        out[f"Threshold_{h}M"] = series_df["Threshold"]

    return out


def get_probability_window(
    center_date: str | pd.Timestamp,
    months_before: int = 24,
    months_after: int = 24
) -> pd.DataFrame:
    center_date = get_available_date_on_or_before(center_date)
    all_probs = get_all_probability_series()

    start_date = center_date - pd.DateOffset(months=months_before)
    end_date = center_date + pd.DateOffset(months=months_after)

    return all_probs.loc[(all_probs.index >= start_date) & (all_probs.index <= end_date)].copy()


# =========================
# Simple CLI Test
# =========================

if __name__ == "__main__":
    test_date = "2008-09-30"
    result = get_recession_risk_for_date(test_date)

    print(f"Risk output for {test_date}:")
    for key, value in result.items():
        print(f"{key}: {value}")