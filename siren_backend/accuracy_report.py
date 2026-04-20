"""
SIREN Accuracy Report v2.0 — With Ablation, Limitations & CI

Generates comprehensive report covering:
  - Dataset summary
  - Layer 1/2/3 performance
  - Model A/B/C metrics with confidence intervals
  - Ablation study results
  - Hyperparameter sensitivity
  - Limitations & future work (required for academic rigor)

Saves: outputs/accuracy_report.txt
"""

import numpy as np
import pandas as pd
import os
import json
from datetime import datetime
from scipy import stats


def bootstrap_ci(y_true, y_pred, metric_fn, n_boot=1000, alpha=0.05):
    """Compute bootstrap confidence interval for a metric."""
    np.random.seed(42)
    n = len(y_true)
    scores = []
    for _ in range(n_boot):
        idx = np.random.choice(n, n, replace=True)
        scores.append(metric_fn(y_true[idx], y_pred[idx]))
    lo = np.percentile(scores, 100 * alpha / 2)
    hi = np.percentile(scores, 100 * (1 - alpha / 2))
    return round(lo, 4), round(hi, 4)


def generate_report(base_dir: str) -> str:
    """Generate the full accuracy report as a formatted string."""
    data_dir = os.path.join(base_dir, "data")
    models_dir = os.path.join(base_dir, "models")
    outputs_dir = os.path.join(base_dir, "outputs")

    df = pd.read_csv(os.path.join(data_dir, "orders_100k.csv"))
    merchants_df = pd.read_csv(os.path.join(data_dir, "merchants_1k.csv"))
    results = pd.read_csv(os.path.join(outputs_dir, "model_results.csv"))
    fi = pd.read_csv(os.path.join(outputs_dir, "feature_importance.csv"))
    preds = pd.read_csv(os.path.join(outputs_dir, "test_predictions.csv"))

    # Optional files
    ablation_path = os.path.join(outputs_dir, "ablation_results.csv")
    ablation = pd.read_csv(ablation_path) if os.path.exists(ablation_path) else None

    hp_path = os.path.join(outputs_dir, "hyperparam_results.csv")
    hp_results = pd.read_csv(hp_path) if os.path.exists(hp_path) else None

    meta_path = os.path.join(models_dir, "training_metadata.json")
    metadata = json.load(open(meta_path)) if os.path.exists(meta_path) else {}

    lines = []
    def S(t): lines.append(""); lines.append("=" * 70); lines.append(f"  {t}"); lines.append("=" * 70)
    def SS(t): lines.append(""); lines.append(f"  ─── {t} ───")

    # ━━━ HEADER ━━━
    lines.append("╔══════════════════════════════════════════════════════════════════╗")
    lines.append("║                    SIREN ACCURACY REPORT v2.0                   ║")
    lines.append("║          Signal-Informed Restaurant ETA Network                 ║")
    lines.append("║          Kitchen Prep Time (KPT) Prediction System              ║")
    lines.append("╚══════════════════════════════════════════════════════════════════╝")
    lines.append(f"\n  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 1. DATASET SUMMARY
    S("1. DATASET SUMMARY")
    dates = pd.to_datetime(df["order_timestamp"])
    contamination = df["is_rider_triggered_FOR"].mean()
    lines.append(f"  Total Orders:         {len(df):,}")
    lines.append(f"  Total Merchants:      {len(merchants_df):,}")
    lines.append(f"  Date Range:           {dates.min().date()} to {dates.max().date()}")
    lines.append(f"  Contamination Rate:   {contamination:.1%} (rider-triggered FORs)")
    lines.append(f"  IoT Beacon Coverage:  {df['has_iot_beacon'].mean():.1%}")
    lines.append(f"  Cities:               {df['city'].nunique()}")
    lines.append(f"  Cuisines:             {df['cuisine'].nunique()}")

    SS("KPT Statistics")
    lines.append(f"    True KPT:     mean={df['true_kpt_min'].mean():.2f}, "
                 f"median={df['true_kpt_min'].median():.2f}, std={df['true_kpt_min'].std():.2f}")
    lines.append(f"    Measured KPT: mean={df['measured_kpt_min'].mean():.2f}, "
                 f"median={df['measured_kpt_min'].median():.2f}, std={df['measured_kpt_min'].std():.2f}")

    if "merchant_skill" in merchants_df.columns:
        SS("Hidden Confounders (not used as model features)")
        lines.append(f"    merchant_skill: mean={merchants_df['merchant_skill'].mean():.3f}, "
                     f"std={merchants_df['merchant_skill'].std():.3f}")
        lines.append(f"    This creates irreducible prediction error → models cannot achieve MAE=0")

    # 2. LAYER 1
    S("2. LAYER 1 — FOR SIGNAL DE-NOISING")
    if "flag_rider_delta" in df.columns:
        predicted = df["flag_rider_delta"].astype(bool) | df["flag_percentile"].astype(bool)
        actual = df["is_rider_triggered_FOR"].astype(bool)
        tp = int((predicted & actual).sum())
        fp = int((predicted & ~actual).sum())
        fn = int((~predicted & actual).sum())
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        bias_before = (df["measured_kpt_min"] - df["true_kpt_min"]).mean()
        clean_mask = df["is_clean_label"] == True
        bias_after = (df.loc[clean_mask, "clean_kpt_min"] - df.loc[clean_mask, "true_kpt_min"]).mean() if clean_mask.sum() > 0 else 0
        lines.append(f"  Filter Precision:     {p:.4f}")
        lines.append(f"  Filter Recall:        {r:.4f}")
        lines.append(f"  Filter F1 Score:      {f1:.4f}")
        lines.append(f"  Filter Rate:          {predicted.mean():.1%}")
        lines.append(f"  Bias BEFORE:          {bias_before:.4f} min")
        lines.append(f"  Bias AFTER:           {bias_after:.4f} min")
        lines.append(f"  Label Efficiency:     {clean_mask.mean():.1%} labels retained")
        if "flag_iot_override" in df.columns:
            lines.append(f"  IoT Overrides:        {df['flag_iot_override'].sum():,}")

    # 3. LAYER 2
    S("3. LAYER 2 — SIGNAL CORRELATIONS WITH true_kpt_min")
    lines.append(f"  NOTE: Layer 2 features are PROXY signals with different coefficients")
    lines.append(f"  than the true data-generating process. This is by design.")
    signals = ["rush_multiplier", "relative_rush", "rain_kpt_impact",
               "cuisine_base_kpt", "complexity_kpt_penalty",
               "google_busyness_index", "hour_sin", "hour_cos", "dow_sin", "dow_cos"]
    for sig in signals:
        if sig in df.columns:
            mask = df[sig].notna() & df["true_kpt_min"].notna()
            if mask.sum() > 2:
                r_val, p_val = stats.pearsonr(df.loc[mask, sig], df.loc[mask, "true_kpt_min"])
                stars = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
                lines.append(f"    {sig:<28s} r = {r_val:+.4f}  p = {p_val:.2e}  {stars}")

    # 4. LAYER 3
    S("4. LAYER 3 — BIAS CORRECTION")
    if "bias_offset" in df.columns:
        err_b = df["measured_kpt_min"] - df["true_kpt_min"]
        corrected = df["measured_kpt_min"] - df["bias_offset"]
        err_a = corrected - df["true_kpt_min"]
        lines.append(f"  RMSE before correction: {np.sqrt((err_b**2).mean()):.4f} min")
        lines.append(f"  RMSE after correction:  {np.sqrt((err_a**2).mean()):.4f} min")
        SS("Per-Tier RMSE")
        for tier in sorted(df["merchant_tier"].unique()):
            mask = df["merchant_tier"] == tier
            rb = np.sqrt((err_b[mask]**2).mean())
            ra = np.sqrt((err_a[mask]**2).mean())
            imp = ((rb - ra) / rb) * 100
            lines.append(f"    {tier:<20s}: {rb:.3f} → {ra:.3f}  ({imp:+.1f}%)")

    # 5. MODEL PERFORMANCE WITH CONFIDENCE INTERVALS
    S("5. MODEL PERFORMANCE (with 95% Bootstrap CI)")
    for _, row in results.iterrows():
        SS(f"Model: {row['model']}")
        lines.append(f"    MAE:          {row['mae']:.4f} min")
        lines.append(f"    RMSE:         {row['rmse']:.4f} min")
        lines.append(f"    P50 AE:       {row['p50_ae']:.4f} min")
        lines.append(f"    P90 AE:       {row['p90_ae']:.4f} min")
        lines.append(f"    Rider Wait:   {row['rider_wait']:.4f} min")

    # Add CI for SIREN XGBoost
    if not preds.empty:
        y_true = preds["true_kpt"].values
        y_pred = preds["pred_siren_xgb"].values
        mae_fn = lambda yt, yp: np.abs(yt - yp).mean()
        ci_lo, ci_hi = bootstrap_ci(y_true, y_pred, mae_fn, n_boot=500)
        lines.append(f"\n  SIREN XGBoost MAE 95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]")

    # 6. IMPROVEMENT
    S("6. IMPROVEMENT SUMMARY (Baseline → SIREN XGBoost)")
    baseline = results[results["model"] == "Baseline_XGBoost"].iloc[0]
    siren = results[results["model"] == "SIREN_XGBoost"].iloc[0]
    for m in ["mae", "rmse", "p50_ae", "p90_ae", "rider_wait"]:
        bl, sr = baseline[m], siren[m]
        imp = ((bl - sr) / bl) * 100
        lines.append(f"    {m.upper():<12s}: {bl:.3f} → {sr:.3f}  ({imp:+.1f}%)")

    # 7. TOP FEATURES
    S("7. TOP 5 MOST IMPORTANT FEATURES (SIREN XGBoost)")
    fi_xgb = fi[fi["model"] == "SIREN_XGBoost"].sort_values("importance", ascending=False).head(5)
    for i, (_, row) in enumerate(fi_xgb.iterrows(), 1):
        lines.append(f"    {i}. {row['feature']:<28s} importance={row['importance']:.4f}  ({row['layer']})")

    # 8. ABLATION STUDY
    S("8. ABLATION STUDY — Incremental Layer Contributions")
    if ablation is not None:
        lines.append(f"  {'Variant':<20s} {'Layers':<12s} {'MAE':>8s} {'RMSE':>8s} {'P50':>8s} {'P90':>8s}")
        lines.append(f"  {'─'*20} {'─'*12} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")
        for _, row in ablation.iterrows():
            lines.append(f"  {str(row['variant']):<20s} {str(row['layers']):<12s} "
                         f"{float(row['mae']):>8.3f} {float(row['rmse']):>8.3f} "
                         f"{float(row['p50_ae']):>8.3f} {float(row['p90_ae']):>8.3f}")

        baseline_mae = ablation.iloc[0]["mae"]
        SS("Layer Contribution Breakdown")
        for i in range(1, len(ablation)):
            prev = ablation.iloc[i-1]["mae"]
            curr = ablation.iloc[i]["mae"]
            delta = prev - curr
            pct = (delta / baseline_mae) * 100
            lines.append(f"    {ablation.iloc[i]['variant']}: "
                         f"ΔMA = {delta:+.3f} min ({pct:+.1f}% of baseline MAE)")
    else:
        lines.append("  [Ablation results not found — run model_training.py]")

    # 9. HYPERPARAMETER SENSITIVITY
    S("9. HYPERPARAMETER SENSITIVITY")
    if hp_results is not None:
        lines.append(f"  {'depth':>5s} {'lr':>6s} {'mcw':>4s} {'sub':>5s} {'col':>5s} {'MAE':>8s}")
        lines.append(f"  {'─'*5} {'─'*6} {'─'*4} {'─'*5} {'─'*5} {'─'*8}")
        for _, row in hp_results.iterrows():
            lines.append(f"  {int(row['max_depth']):>5d} {row['learning_rate']:>6.3f} "
                         f"{int(row['min_child_weight']):>4d} {row['subsample']:>5.2f} "
                         f"{row['colsample_bytree']:>5.2f} {row['mae']:>8.4f}")
        best = hp_results.loc[hp_results["mae"].idxmin()]
        lines.append(f"\n  Best: depth={int(best['max_depth'])}, lr={best['learning_rate']}, "
                     f"mcw={int(best['min_child_weight'])} → MAE={best['mae']:.4f}")
    else:
        lines.append("  [Hyperparameter results not found]")

    # 10. PER-CITY
    S("10. PER-CITY MAE BREAKDOWN")
    for city in sorted(preds["city"].unique()):
        mask = preds["city"] == city
        mae_bl = np.abs(preds.loc[mask, "pred_baseline"] - preds.loc[mask, "true_kpt"]).mean()
        mae_sr = np.abs(preds.loc[mask, "pred_siren_xgb"] - preds.loc[mask, "true_kpt"]).mean()
        imp = ((mae_bl - mae_sr) / mae_bl) * 100
        lines.append(f"    {city:<15s}: Baseline={mae_bl:.3f}  SIREN={mae_sr:.3f}  ({imp:+.1f}%)")

    # 11. PER-CUISINE
    S("11. PER-CUISINE MAE BREAKDOWN")
    for cuisine in sorted(preds["cuisine"].unique()):
        mask = preds["cuisine"] == cuisine
        mae_bl = np.abs(preds.loc[mask, "pred_baseline"] - preds.loc[mask, "true_kpt"]).mean()
        mae_sr = np.abs(preds.loc[mask, "pred_siren_xgb"] - preds.loc[mask, "true_kpt"]).mean()
        imp = ((mae_bl - mae_sr) / mae_bl) * 100
        lines.append(f"    {cuisine:<15s}: Baseline={mae_bl:.3f}  SIREN={mae_sr:.3f}  ({imp:+.1f}%)")

    # 12. LIMITATIONS
    S("12. LIMITATIONS & FUTURE WORK")
    lines.append("""
  LIMITATIONS:
  1. SYNTHETIC DATA: All results are on synthetic data. The data-generating
     process was designed to simulate realistic bias mechanisms, but real-world
     performance may differ significantly due to unmeasured confounders.

  2. PROXY FEATURES: Layer 2 enrichment features (rush_multiplier, rain_impact,
     complexity_penalty) use domain-knowledge approximations with different
     coefficients than the true data-generating process. This is deliberate
     to prevent circular feature→target mapping, but means the features are
     imperfect proxies — as they would be in production.

  3. HIDDEN CONFOUNDERS: The data includes a hidden 'merchant_skill' factor
     (0.75-1.25) that is NOT available to any model. This creates irreducible
     prediction error (~2-3 min MAE floor), simulating real-world unmeasured
     factors like chef skill, kitchen layout, and ingredient freshness.

  4. TEMPORAL SPLIT ONLY: Models were evaluated on a single Nov-Dec 2024
     holdout period. K-fold temporal cross-validation would provide more
     robust performance estimates.

  5. NO REAL-WORLD VALIDATION: Production deployment would require A/B
     testing on live Zomato data with actual rider GPS and FOR timestamps.

  FUTURE WORK:
  - Validate on real Zomato/Swiggy FOR data if available
  - Test with real Google Maps Popular Times API integration
  - Implement online learning for Layer 3 bias adaptation
  - Add confidence calibration (prediction intervals)
  - Explore neural network architectures (LSTM for sequential orders)""")

    lines.append("\n" + "=" * 70)
    lines.append("  END OF REPORT")
    lines.append("=" * 70)
    return "\n".join(lines)


def main():
    """Generate and save the accuracy report."""
    print("=" * 60)
    print("SIREN Accuracy Report Generator v2.0")
    print("=" * 60)

    base_dir = os.path.dirname(__file__)
    outputs_dir = os.path.join(base_dir, "outputs")

    print("\nGenerating report...")
    report = generate_report(base_dir)
    print("\n" + report)

    report_path = os.path.join(outputs_dir, "accuracy_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n✅ Report saved to {report_path}")


if __name__ == "__main__":
    main()
