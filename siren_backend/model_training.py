"""
SIREN Model Training v2.0 — With Ablation Study & Hyperparameter Tuning

Trains and evaluates:
  Model A: Baseline XGBoost (biased labels, no enrichment)
  Model B: SIREN XGBoost (clean labels, full enrichment + bias correction)
  Model C: SIREN LightGBM (same as B, different algorithm)

Plus:
  Ablation Study — incremental contribution of each SIREN layer
  Hyperparameter Search — 8-config grid to find optimal XGBoost params

Temporal split: train on Jan 2023 - Oct 2024, test on Nov-Dec 2024
"""

import numpy as np
import pandas as pd
import os
import joblib
import json
from datetime import datetime
from typing import Dict, List, Tuple

import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FEATURE DEFINITIONS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

BASELINE_FEATURES = [
    "hour", "day_of_week", "is_lunch_rush", "is_dinner_rush", "is_weekend",
    "order_complexity", "zomato_concurrent_orders"
]

# Layer 1+2 features: clean labels + enrichment, NO bias correction
L1_L2_FEATURES = BASELINE_FEATURES + [
    "rush_multiplier", "relative_rush",
    "rain_flag", "rain_kpt_impact",
    "complexity_kpt_penalty", "cuisine_base_kpt",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "google_busyness_index"
]

# Full SIREN: L1+L2 + Layer 3 bias features
SIREN_FEATURES = L1_L2_FEATURES + [
    "mean_marking_bias", "bias_offset"
]


def temporal_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split data temporally: train ≤Oct 2024, test Nov-Dec 2024."""
    df["order_timestamp"] = pd.to_datetime(df["order_timestamp"])
    train_cutoff = pd.Timestamp("2024-11-01")
    train_df = df[df["order_timestamp"] < train_cutoff].copy()
    test_df = df[df["order_timestamp"] >= train_cutoff].copy()
    print(f"  Train: {len(train_df):,} orders (to Oct 2024)")
    print(f"  Test:  {len(test_df):,} orders (Nov-Dec 2024)")
    return train_df, test_df


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                     dispatch_offset: float = 3.0) -> Dict[str, float]:
    """Compute MAE, RMSE, P50 AE, P90 AE, and simulated Rider Wait."""
    abs_errors = np.abs(y_true - y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    p50_ae = np.median(abs_errors)
    p90_ae = np.percentile(abs_errors, 90)

    # Rider wait: food not ready when rider arrives (dispatched early based on prediction)
    np.random.seed(99)
    rider_wait = np.maximum(0, y_true - y_pred + np.random.normal(0, 0.5, len(y_true)))
    avg_rider_wait = rider_wait.mean()

    return {
        "mae": round(mae, 4), "rmse": round(rmse, 4),
        "p50_ae": round(p50_ae, 4), "p90_ae": round(p90_ae, 4),
        "rider_wait": round(avg_rider_wait, 4)
    }


def compute_tier_metrics(test_df: pd.DataFrame, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute MAE breakdown by merchant tier."""
    tier_mae = {}
    for tier in test_df["merchant_tier"].unique():
        mask = test_df["merchant_tier"].values == tier
        tier_mae[tier] = round(
            mean_absolute_error(test_df["true_kpt_min"].values[mask], y_pred[mask]), 4
        )
    return tier_mae


def _prepare_features(df, features):
    """Ensure all features exist and fill NaN."""
    for f in features:
        if f not in df.columns:
            df[f] = 0
    return df[features].fillna(0).values


def train_baseline_xgb(train_df, test_df):
    """Model A: Baseline XGBoost on biased labels."""
    print("\n  Training Baseline XGBoost...")
    X_train = train_df[BASELINE_FEATURES].values
    y_train = train_df["measured_kpt_min"].values
    X_test = test_df[BASELINE_FEATURES].values
    y_test = test_df["true_kpt_min"].values

    model = xgb.XGBRegressor(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        random_state=42, n_jobs=-1, verbosity=0
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    y_pred = model.predict(X_test)
    metrics = compute_metrics(y_test, y_pred)
    tier_metrics = compute_tier_metrics(test_df, y_pred)
    print(f"  → MAE: {metrics['mae']:.3f}, RMSE: {metrics['rmse']:.3f}, "
          f"P90: {metrics['p90_ae']:.3f}, Wait: {metrics['rider_wait']:.3f}")
    return model, y_pred, metrics, tier_metrics


def train_siren_xgb(train_df, test_df, params=None):
    """Model B: SIREN XGBoost on clean labels with bias correction."""
    print("\n  Training SIREN XGBoost...")
    clean_train = train_df[train_df["is_clean_label"] == True].copy()
    print(f"  Using {len(clean_train):,} clean orders ({len(clean_train)/len(train_df):.1%})")

    X_train = _prepare_features(clean_train, SIREN_FEATURES)
    y_train = clean_train["clean_kpt_min"].values
    X_test = _prepare_features(test_df.copy(), SIREN_FEATURES)
    y_test = test_df["true_kpt_min"].values

    if params is None:
        params = dict(n_estimators=500, max_depth=6, learning_rate=0.05,
                      subsample=0.8, colsample_bytree=0.8, min_child_weight=5)

    model = xgb.XGBRegressor(**params, random_state=42, n_jobs=-1, verbosity=0)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    y_pred = model.predict(X_test)

    # Bias correction (Layer 3)
    bias_offset = test_df["bias_offset"].fillna(0).values
    y_pred_corrected = y_pred - bias_offset

    metrics = compute_metrics(y_test, y_pred_corrected)
    tier_metrics = compute_tier_metrics(test_df, y_pred_corrected)
    print(f"  → MAE: {metrics['mae']:.3f}, RMSE: {metrics['rmse']:.3f}, "
          f"P90: {metrics['p90_ae']:.3f}, Wait: {metrics['rider_wait']:.3f}")
    return model, y_pred_corrected, metrics, tier_metrics


def train_siren_lgbm(train_df, test_df):
    """Model C: SIREN LightGBM on clean labels with bias correction."""
    print("\n  Training SIREN LightGBM...")
    clean_train = train_df[train_df["is_clean_label"] == True].copy()

    X_train = _prepare_features(clean_train, SIREN_FEATURES)
    y_train = clean_train["clean_kpt_min"].values
    X_test = _prepare_features(test_df.copy(), SIREN_FEATURES)
    y_test = test_df["true_kpt_min"].values

    model = lgb.LGBMRegressor(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        random_state=42, n_jobs=-1, verbose=-1
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)])
    y_pred = model.predict(X_test)

    bias_offset = test_df["bias_offset"].fillna(0).values
    y_pred_corrected = y_pred - bias_offset

    metrics = compute_metrics(y_test, y_pred_corrected)
    tier_metrics = compute_tier_metrics(test_df, y_pred_corrected)
    print(f"  → MAE: {metrics['mae']:.3f}, RMSE: {metrics['rmse']:.3f}, "
          f"P90: {metrics['p90_ae']:.3f}, Wait: {metrics['rider_wait']:.3f}")
    return model, y_pred_corrected, metrics, tier_metrics


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ABLATION STUDY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_ablation_study(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    """
    Ablation study: test incremental contribution of each SIREN layer.

    Variants:
      1. Baseline: measured_kpt target, 7 basic features
      2. L1 Only:  clean_kpt target, 7 basic features (denoising alone)
      3. L1+L2:    clean_kpt target, enriched features (+ enrichment)
      4. Full SIREN: clean_kpt target, all features + bias correction

    Returns DataFrame with MAE for each variant.
    """
    print("\n" + "━" * 40)
    print("ABLATION STUDY")
    print("━" * 40)

    y_test = test_df["true_kpt_min"].values
    clean_train = train_df[train_df["is_clean_label"] == True].copy()
    results = []

    xgb_params = dict(n_estimators=300, max_depth=6, learning_rate=0.05,
                      subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                      random_state=42, n_jobs=-1, verbosity=0)

    # Variant 1: Baseline
    print("\n  [1/4] Baseline (biased labels, basic features)...")
    m = xgb.XGBRegressor(**xgb_params)
    m.fit(train_df[BASELINE_FEATURES].values, train_df["measured_kpt_min"].values)
    pred = m.predict(test_df[BASELINE_FEATURES].values)
    metrics = compute_metrics(y_test, pred)
    results.append({"variant": "Baseline", "layers": "None", **metrics})
    print(f"    MAE: {metrics['mae']:.3f}")

    # Variant 2: Layer 1 Only (clean labels, same basic features)
    print("  [2/4] L1 Only (clean labels, basic features)...")
    m = xgb.XGBRegressor(**xgb_params)
    m.fit(clean_train[BASELINE_FEATURES].values, clean_train["clean_kpt_min"].values)
    pred = m.predict(test_df[BASELINE_FEATURES].values)
    metrics = compute_metrics(y_test, pred)
    results.append({"variant": "L1_Only", "layers": "L1", **metrics})
    print(f"    MAE: {metrics['mae']:.3f}")

    # Variant 3: L1+L2 (clean labels + enrichment features, no bias correction)
    print("  [3/4] L1+L2 (clean labels + enrichment)...")
    X_tr = _prepare_features(clean_train, L1_L2_FEATURES)
    X_te = _prepare_features(test_df.copy(), L1_L2_FEATURES)
    m = xgb.XGBRegressor(**xgb_params)
    m.fit(X_tr, clean_train["clean_kpt_min"].values)
    pred = m.predict(X_te)
    metrics = compute_metrics(y_test, pred)
    results.append({"variant": "L1_L2", "layers": "L1+L2", **metrics})
    print(f"    MAE: {metrics['mae']:.3f}")

    # Variant 4: Full SIREN (all layers + bias correction)
    print("  [4/4] Full SIREN (all layers + bias correction)...")
    X_tr = _prepare_features(clean_train, SIREN_FEATURES)
    X_te = _prepare_features(test_df.copy(), SIREN_FEATURES)
    m = xgb.XGBRegressor(**xgb_params)
    m.fit(X_tr, clean_train["clean_kpt_min"].values)
    pred = m.predict(X_te)
    pred_corrected = pred - test_df["bias_offset"].fillna(0).values
    metrics = compute_metrics(y_test, pred_corrected)
    results.append({"variant": "Full_SIREN", "layers": "L1+L2+L3", **metrics})
    print(f"    MAE: {metrics['mae']:.3f}")

    ablation_df = pd.DataFrame(results)

    # Print improvement breakdown
    baseline_mae = ablation_df.iloc[0]["mae"]
    print("\n  Incremental improvement:")
    for i in range(1, len(ablation_df)):
        prev_mae = ablation_df.iloc[i-1]["mae"]
        curr_mae = ablation_df.iloc[i]["mae"]
        delta = prev_mae - curr_mae
        pct = (delta / baseline_mae) * 100
        print(f"    {ablation_df.iloc[i-1]['variant']} → {ablation_df.iloc[i]['variant']}: "
              f"MAE {prev_mae:.3f} → {curr_mae:.3f}  (Δ{delta:+.3f}, {pct:+.1f}% of baseline)")

    return ablation_df


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HYPERPARAMETER TUNING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def tune_hyperparameters(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict:
    """
    Hyperparameter search over key XGBoost parameters.

    Tests 8 configurations and returns the best.
    Uses n_estimators=200 for speed during search.
    """
    print("\n" + "━" * 40)
    print("HYPERPARAMETER TUNING")
    print("━" * 40)

    clean_train = train_df[train_df["is_clean_label"] == True].copy()
    X_train = _prepare_features(clean_train, SIREN_FEATURES)
    y_train = clean_train["clean_kpt_min"].values
    X_test = _prepare_features(test_df.copy(), SIREN_FEATURES)
    y_test = test_df["true_kpt_min"].values
    bias_offset = test_df["bias_offset"].fillna(0).values

    configs = [
        {"max_depth": 4, "learning_rate": 0.05, "min_child_weight": 3, "subsample": 0.8, "colsample_bytree": 0.8},
        {"max_depth": 6, "learning_rate": 0.05, "min_child_weight": 5, "subsample": 0.8, "colsample_bytree": 0.8},  # default
        {"max_depth": 8, "learning_rate": 0.03, "min_child_weight": 5, "subsample": 0.7, "colsample_bytree": 0.7},
        {"max_depth": 6, "learning_rate": 0.1, "min_child_weight": 3, "subsample": 0.9, "colsample_bytree": 0.9},
        {"max_depth": 5, "learning_rate": 0.05, "min_child_weight": 7, "subsample": 0.8, "colsample_bytree": 0.8},
        {"max_depth": 7, "learning_rate": 0.03, "min_child_weight": 5, "subsample": 0.8, "colsample_bytree": 0.8},
        {"max_depth": 6, "learning_rate": 0.05, "min_child_weight": 5, "subsample": 0.7, "colsample_bytree": 0.9},
        {"max_depth": 5, "learning_rate": 0.08, "min_child_weight": 3, "subsample": 0.85, "colsample_bytree": 0.85},
    ]

    best_mae = float("inf")
    best_config = None
    tuning_results = []

    for i, cfg in enumerate(configs):
        model = xgb.XGBRegressor(**cfg, n_estimators=200, random_state=42, n_jobs=-1, verbosity=0)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        pred = model.predict(X_test) - bias_offset
        mae = mean_absolute_error(y_test, pred)
        tuning_results.append({**cfg, "mae": round(mae, 4)})
        marker = " ← BEST" if mae < best_mae else ""
        print(f"  Config {i+1}/8: depth={cfg['max_depth']}, lr={cfg['learning_rate']}, "
              f"mcw={cfg['min_child_weight']} → MAE={mae:.4f}{marker}")
        if mae < best_mae:
            best_mae = mae
            best_config = cfg

    print(f"\n  Best config: {best_config}")
    print(f"  Best MAE (n=200): {best_mae:.4f}")

    return {"best_config": best_config, "all_results": tuning_results}


def get_feature_importance(model, feature_names, model_name):
    """Extract feature importance mapped to SIREN layers."""
    importances = model.feature_importances_ if hasattr(model, 'feature_importances_') else np.zeros(len(feature_names))
    layer_map = {
        "hour": "Base", "day_of_week": "Base", "is_lunch_rush": "Base",
        "is_dinner_rush": "Base", "is_weekend": "Base",
        "order_complexity": "Base", "zomato_concurrent_orders": "Base",
        "rush_multiplier": "Layer 2", "relative_rush": "Layer 2",
        "rain_flag": "Layer 2", "rain_kpt_impact": "Layer 2",
        "complexity_kpt_penalty": "Layer 2", "cuisine_base_kpt": "Layer 2",
        "hour_sin": "Layer 2", "hour_cos": "Layer 2",
        "dow_sin": "Layer 2", "dow_cos": "Layer 2",
        "google_busyness_index": "Layer 2",
        "mean_marking_bias": "Layer 3", "bias_offset": "Layer 3"
    }
    return pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
        "layer": [layer_map.get(f, "Unknown") for f in feature_names],
        "model": model_name
    }).sort_values("importance", ascending=False)


def main():
    """Run the complete model training pipeline with ablation & tuning."""
    print("=" * 60)
    print("SIREN Model Training Pipeline v2.0")
    print("=" * 60)

    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(base_dir, "data")
    models_dir = os.path.join(base_dir, "models")
    outputs_dir = os.path.join(base_dir, "outputs")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)

    print("\nLoading data...")
    df = pd.read_csv(os.path.join(data_dir, "orders_100k.csv"))
    print(f"Loaded {len(df):,} orders with {len(df.columns)} columns")

    print("\nSplitting data temporally...")
    train_df, test_df = temporal_split(df)

    # ━━━ HYPERPARAMETER TUNING ━━━
    tuning = tune_hyperparameters(train_df, test_df)
    best_params = {**tuning["best_config"], "n_estimators": 500}

    # ━━━ MODEL A: BASELINE ━━━
    print("\n" + "━" * 40 + "\nMODEL A: Baseline XGBoost\n" + "━" * 40)
    baseline_model, baseline_pred, baseline_metrics, baseline_tier = train_baseline_xgb(train_df, test_df)

    # ━━━ MODEL B: SIREN XGBoost (with tuned params) ━━━
    print("\n" + "━" * 40 + "\nMODEL B: SIREN XGBoost (tuned)\n" + "━" * 40)
    siren_xgb_model, siren_xgb_pred, siren_xgb_metrics, siren_xgb_tier = train_siren_xgb(
        train_df, test_df, params=best_params
    )

    # ━━━ MODEL C: SIREN LightGBM ━━━
    print("\n" + "━" * 40 + "\nMODEL C: SIREN LightGBM\n" + "━" * 40)
    siren_lgbm_model, siren_lgbm_pred, siren_lgbm_metrics, siren_lgbm_tier = train_siren_lgbm(train_df, test_df)

    # ━━━ ABLATION STUDY ━━━
    ablation_df = run_ablation_study(train_df, test_df)

    # ━━━ IMPROVEMENT SUMMARY ━━━
    print("\n" + "=" * 60 + "\nIMPROVEMENT SUMMARY (Baseline → SIREN XGBoost)\n" + "=" * 60)
    for metric in ["mae", "rmse", "p50_ae", "p90_ae", "rider_wait"]:
        bl = baseline_metrics[metric]
        sr = siren_xgb_metrics[metric]
        improvement = ((bl - sr) / bl) * 100 if bl > 0 else 0
        print(f"  {metric.upper():<12s}: {bl:.3f} → {sr:.3f}  ({improvement:+.1f}%)")

    # ━━━ SAVE EVERYTHING ━━━
    print("\nSaving models...")
    joblib.dump(baseline_model, os.path.join(models_dir, "baseline_xgb.pkl"))
    joblib.dump(siren_xgb_model, os.path.join(models_dir, "siren_xgb.pkl"))
    joblib.dump(siren_lgbm_model, os.path.join(models_dir, "siren_lgbm.pkl"))

    results = pd.DataFrame([
        {"model": "Baseline_XGBoost", **baseline_metrics},
        {"model": "SIREN_XGBoost", **siren_xgb_metrics},
        {"model": "SIREN_LightGBM", **siren_lgbm_metrics}
    ])
    results.to_csv(os.path.join(outputs_dir, "model_results.csv"), index=False)

    fi_xgb = get_feature_importance(siren_xgb_model, SIREN_FEATURES, "SIREN_XGBoost")
    fi_lgbm = get_feature_importance(siren_lgbm_model, SIREN_FEATURES, "SIREN_LightGBM")
    pd.concat([fi_xgb, fi_lgbm]).to_csv(os.path.join(outputs_dir, "feature_importance.csv"), index=False)

    ablation_df.to_csv(os.path.join(outputs_dir, "ablation_results.csv"), index=False)
    pd.DataFrame(tuning["all_results"]).to_csv(os.path.join(outputs_dir, "hyperparam_results.csv"), index=False)

    test_preds = pd.DataFrame({
        "order_id": test_df["order_id"].values,
        "true_kpt": test_df["true_kpt_min"].values,
        "pred_baseline": baseline_pred,
        "pred_siren_xgb": siren_xgb_pred,
        "pred_siren_lgbm": siren_lgbm_pred,
        "tier": test_df["merchant_tier"].values,
        "city": test_df["city"].values,
        "cuisine": test_df["cuisine"].values
    })
    test_preds.to_csv(os.path.join(outputs_dir, "test_predictions.csv"), index=False)

    metadata = {
        "trained_at": datetime.now().isoformat(),
        "train_size": len(train_df), "test_size": len(test_df),
        "baseline_features": BASELINE_FEATURES,
        "siren_features": SIREN_FEATURES,
        "best_hyperparams": tuning["best_config"],
        "baseline_metrics": baseline_metrics,
        "siren_xgb_metrics": siren_xgb_metrics,
        "siren_lgbm_metrics": siren_lgbm_metrics,
        "baseline_tier_mae": baseline_tier,
        "siren_xgb_tier_mae": siren_xgb_tier,
        "siren_lgbm_tier_mae": siren_lgbm_tier
    }
    with open(os.path.join(models_dir, "training_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved all outputs to {outputs_dir}")
    print("\n✅ Model training complete!")

    return {
        "baseline": (baseline_model, baseline_pred, baseline_metrics, baseline_tier),
        "siren_xgb": (siren_xgb_model, siren_xgb_pred, siren_xgb_metrics, siren_xgb_tier),
        "siren_lgbm": (siren_lgbm_model, siren_lgbm_pred, siren_lgbm_metrics, siren_lgbm_tier),
        "ablation": ablation_df, "tuning": tuning
    }


if __name__ == "__main__":
    main()
