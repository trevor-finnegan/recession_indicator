import json
import pickle
from pathlib import Path

from fredapi import Fred
import numpy as np
import pandas as pd

from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


# =========================
# Config
# =========================

DATA_SERIES = {
    # Labor Market
    "Unemployment": "UNRATE",
    "LFPR": "CIVPART",
    "EPopRatio": "EMRATIO",
    "MonthlyJobChange": "PAYEMS",
    "JobOpenings": "JTSJOL",
    "SahmRule": "SAHMCURRENT",

    # Inflation
    "CPI": "CPIAUCSL",
    "CoreCPI": "CPILFESL",
    "EmploymentCostIndex": "ECIALLCIV",
    "InflationExpectation": "MICH",
    "10yrBreakeven": "T10YIE",
    "5y5yForward": "T5YIFR",

    # Growth Expectation
    "GDPNow": "GDPNOW",

    # Sentiment & Uncertainty
    "ConsumerSentiment": "UMCSENT",
    "TradeUncertainty": "EPUTRADE",
    "PolicyUncertainty": "USEPUINDXM",

    # Financial Markets
    "SP500": "SP500",
    "NFCI": "NFCI",
    "FFR": "FEDFUNDS",
    "10y2ySpread": "T10Y2Y",

    # Recession
    "Recession": "USREC",
}

HORIZONS = [3, 6, 12]
START_DATE = "1930-01-01"
FEATURE_TOP_N = 40
THRESHOLD_OBJECTIVE = "f1"

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# Data Pull / Cleaning
# =========================

def pull_fred_data(fred_key: str) -> pd.DataFrame:
    fred = Fred(api_key=fred_key)

    clean = {}
    for series_name, series_id in DATA_SERIES.items():
        series = fred.get_series(series_id)
        df = pd.DataFrame(series, columns=[series_name])
        df.index = pd.to_datetime(df.index)
        clean[series_name] = df

    for clean_name, clean_df in clean.items():
        clean[clean_name] = clean_df.resample("ME").last()

    data = pd.concat(clean.values(), axis=1)
    data = data.loc[START_DATE:].copy()
    data = data.sort_index().ffill().bfill()

    return data


def clean_data(df: pd.DataFrame, new_name: str) -> pd.DataFrame:
    df = df[["date", "value"]].copy()
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df.rename(columns={"value": new_name}, inplace=True)
    df.set_index("date", inplace=True)
    return df


# =========================
# Feature Engineering
# =========================

def add_tabular_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    base_cols = list(df.columns)

    for col in base_cols:
        df[f"{col}_lag1"] = df[col].shift(1)
        df[f"{col}_lag3"] = df[col].shift(3)
        df[f"{col}_chg1"] = df[col] - df[col].shift(1)
        df[f"{col}_chg3"] = df[col] - df[col].shift(3)
        df[f"{col}_ma3"] = df[col].rolling(3).mean()
        df[f"{col}_std6"] = df[col].rolling(6).std()

    if "SP500" in df.columns:
        df["SP500_ret3"] = df["SP500"].pct_change(3)
        df["SP500_ret12"] = df["SP500"].pct_change(12)
        df["SP500_drawdown12"] = df["SP500"] / df["SP500"].rolling(12).max() - 1

    if "10y2ySpread" in df.columns:
        df["YieldCurveInverted"] = (df["10y2ySpread"] < 0).astype(int)
        df["YieldCurveInversionDuration6"] = (
            (df["10y2ySpread"] < 0).astype(int).rolling(6).sum()
        )

    if "Unemployment" in df.columns:
        df["Unemployment_chg6"] = df["Unemployment"] - df["Unemployment"].shift(6)

    if "SahmRule" in df.columns:
        df["SahmRule_chg3"] = df["SahmRule"] - df["SahmRule"].shift(3)

    return df


def build_forward_window_target(recession_series: pd.Series, horizon: int) -> pd.Series:
    """
    Target = 1 if a recession occurs at any point within the next `horizon` months.
    """
    return (
        recession_series
        .rolling(window=horizon, min_periods=horizon)
        .max()
        .shift(-horizon + 1)
    )


# =========================
# Threshold Tuning
# =========================

def find_best_threshold(
    y_true: pd.Series,
    y_prob: np.ndarray,
    objective: str = "f1"
) -> tuple[float, float]:
    thresholds = np.linspace(0.05, 0.95, 91)

    best_threshold = 0.5
    best_score = -1.0

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)

        if objective == "f1":
            score = f1_score(y_true, y_pred, zero_division=0)
        elif objective == "recall":
            from sklearn.metrics import recall_score
            score = recall_score(y_true, y_pred, zero_division=0)
        else:
            raise ValueError("objective must be 'f1' or 'recall'")

        if score > best_score:
            best_score = score
            best_threshold = float(t)

    return best_threshold, float(best_score)


# =========================
# Model Training
# =========================

def train_lightgbm_for_horizon(
    data_source: pd.DataFrame,
    x_features: pd.DataFrame,
    horizon: int,
    feature_top_n: int = FEATURE_TOP_N,
    threshold_objective: str = THRESHOLD_OBJECTIVE,
) -> tuple[dict, dict]:
    temp = data_source.copy()
    temp["Target"] = build_forward_window_target(temp["Recession"], horizon)
    temp = temp.dropna(subset=["Target"]).copy()
    temp["Target"] = temp["Target"].astype(int)

    x_h = x_features.loc[temp.index].copy()
    y_h = temp["Target"].copy()

    split_idx = int(len(temp) * 0.8)

    x_train = x_h.iloc[:split_idx].copy()
    x_test = x_h.iloc[split_idx:].copy()
    y_train = y_h.iloc[:split_idx].copy()
    y_test = y_h.iloc[split_idx:].copy()

    base_model = LGBMClassifier(
        n_estimators=800,
        learning_rate=0.02,
        max_depth=-1,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        class_weight="balanced",
        random_state=42,
    )

    base_model.fit(x_train, y_train)

    importance = pd.DataFrame({
        "Feature": x_train.columns,
        "Importance": base_model.feature_importances_,
    }).sort_values("Importance", ascending=False)

    selected_features = importance.head(feature_top_n)["Feature"].tolist()

    x_train_sel = x_train[selected_features].copy()
    x_test_sel = x_test[selected_features].copy()

    lgbm_model = LGBMClassifier(
        n_estimators=800,
        learning_rate=0.02,
        max_depth=-1,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        class_weight="balanced",
        random_state=42,
    )

    calibrated_model = CalibratedClassifierCV(
        estimator=lgbm_model,
        method="sigmoid",
        cv=3,
    )

    calibrated_model.fit(x_train_sel, y_train)

    y_prob = calibrated_model.predict_proba(x_test_sel)[:, 1]

    best_threshold, best_score = find_best_threshold(
        y_true=y_test,
        y_prob=y_prob,
        objective=threshold_objective,
    )

    metrics = {
        "horizon": horizon,
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "pr_auc": float(average_precision_score(y_test, y_prob)),
        "best_threshold": best_threshold,
        "best_threshold_score": best_score,
        "selected_features": selected_features,
        "train_start": str(x_train_sel.index.min().date()),
        "train_end": str(x_train_sel.index.max().date()),
        "test_start": str(x_test_sel.index.min().date()),
        "test_end": str(x_test_sel.index.max().date()),
        "n_train": int(len(x_train_sel)),
        "n_test": int(len(x_test_sel)),
        "positive_rate_train": float(y_train.mean()),
        "positive_rate_test": float(y_test.mean()),
    }

    model_package = {
        "model": calibrated_model,
        "selected_features": selected_features,
        "threshold": best_threshold,
        "horizon": horizon,
    }

    return model_package, metrics


# =========================
# Saving Artifacts
# =========================

def save_model_package(model_package: dict, metrics: dict, model_dir: Path = MODEL_DIR) -> None:
    horizon = model_package["horizon"]

    with open(model_dir / f"lightgbm_{horizon}m.pkl", "wb") as f:
        pickle.dump(model_package["model"], f)

    with open(model_dir / f"selected_features_{horizon}m.pkl", "wb") as f:
        pickle.dump(model_package["selected_features"], f)

    threshold_payload = {
        "horizon": horizon,
        "threshold": model_package["threshold"],
    }
    with open(model_dir / f"threshold_{horizon}m.json", "w") as f:
        json.dump(threshold_payload, f, indent=2)

    with open(model_dir / f"metrics_{horizon}m.json", "w") as f:
        json.dump(metrics, f, indent=2)


def save_training_data_artifact(
    x_features: pd.DataFrame,
    model_dir: Path = MODEL_DIR,
) -> None:
    x_features.to_parquet(model_dir / "feature_table.parquet")


# =========================
# Main Training Entry Point
# =========================

def main():
    fred_key = "17cb00a71dda9a4ee025fbb4c1adfed9"

    print("Pulling and cleaning data...")
    data = pull_fred_data(fred_key)

    print("Building feature table...")
    feature_base = data.drop(columns=["Recession"]).copy()
    x_tabular_full = add_tabular_features(feature_base).ffill().bfill()

    save_training_data_artifact(x_tabular_full)

    summary_rows = []

    for horizon in HORIZONS:
        print(f"\nTraining calibrated LightGBM for {horizon}M horizon...")
        model_package, metrics = train_lightgbm_for_horizon(
            data_source=data,
            x_features=x_tabular_full,
            horizon=horizon,
            feature_top_n=FEATURE_TOP_N,
            threshold_objective=THRESHOLD_OBJECTIVE,
        )

        save_model_package(model_package, metrics)

        summary_rows.append({
            "HorizonMonths": horizon,
            "ROC_AUC": metrics["roc_auc"],
            "PR_AUC": metrics["pr_auc"],
            "BestThreshold": metrics["best_threshold"],
            "BestThresholdScore": metrics["best_threshold_score"],
            "PositiveRateTrain": metrics["positive_rate_train"],
            "PositiveRateTest": metrics["positive_rate_test"],
        })

    summary_df = pd.DataFrame(summary_rows).sort_values("HorizonMonths")
    print("\nTraining summary:")
    print(summary_df)

    summary_df.to_csv(MODEL_DIR / "training_summary.csv", index=False)


if __name__ == "__main__":
    main()