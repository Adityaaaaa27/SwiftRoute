"""
SIREN Visualizations — 6 High-Resolution Publication-Quality Charts

Generates:
1. 01_label_bias_distribution.png — KPT label error histograms (before/after de-noising)
2. 02_theta_sweep.png — Precision, Recall, Filter Rate vs θ (30s to 300s)
3. 03_rush_signal.png — Google busyness vs true KPT + hourly heatmap
4. 04_merchant_bias.png — Per-merchant bias distribution + correction by behavior
5. 05_model_comparison.png — 6-panel model comparison dashboard
6. 06_feature_importance.png — Feature importance color-coded by SIREN layer
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os
from typing import Dict, Optional

# Style configuration
plt.rcParams.update({
    "figure.facecolor": "#0d1117",
    "axes.facecolor": "#161b22",
    "axes.edgecolor": "#30363d",
    "axes.labelcolor": "#c9d1d9",
    "text.color": "#c9d1d9",
    "xtick.color": "#8b949e",
    "ytick.color": "#8b949e",
    "grid.color": "#21262d",
    "grid.alpha": 0.6,
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "figure.titlesize": 16,
    "figure.titleweight": "bold"
})

# SIREN color palette
COLORS = {
    "primary": "#58a6ff",
    "secondary": "#f778ba",
    "success": "#3fb950",
    "warning": "#d29922",
    "danger": "#f85149",
    "purple": "#bc8cff",
    "cyan": "#56d4dd",
    "orange": "#f0883e",
    "layer1": "#f85149",
    "layer2": "#58a6ff",
    "layer3": "#3fb950",
    "base": "#8b949e",
    "baseline": "#f85149",
    "siren_xgb": "#58a6ff",
    "siren_lgbm": "#3fb950"
}


def plot_01_label_bias(df: pd.DataFrame, output_dir: str) -> None:
    """
    Plot KPT label error distribution before and after de-noising.
    
    Shows two overlaid histograms:
    - Before: measured_kpt - true_kpt (contaminated)
    - After: clean_kpt - true_kpt (de-noised)
    
    Args:
        df: DataFrame with measured_kpt_min, true_kpt_min, clean_kpt_min, is_clean_label.
        output_dir: Directory to save the chart.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("SIREN Layer 1: Label Bias Distribution — Before vs After De-noising",
                 fontsize=16, fontweight="bold", color="#c9d1d9")
    
    # Before
    error_before = df["measured_kpt_min"] - df["true_kpt_min"]
    axes[0].hist(error_before, bins=100, color=COLORS["danger"], alpha=0.7, 
                 edgecolor="none", density=True)
    axes[0].axvline(x=0, color="#c9d1d9", linestyle="--", alpha=0.5)
    axes[0].axvline(x=error_before.mean(), color=COLORS["warning"], linestyle="-", 
                    linewidth=2, label=f"Mean bias: {error_before.mean():.2f} min")
    axes[0].set_title("BEFORE De-noising", color=COLORS["danger"])
    axes[0].set_xlabel("Label Error (measured - true) [min]")
    axes[0].set_ylabel("Density")
    axes[0].legend(fontsize=10)
    axes[0].set_xlim(-25, 15)
    axes[0].grid(True, alpha=0.3)
    
    # After
    clean_mask = df["is_clean_label"] == True
    error_after = df.loc[clean_mask, "clean_kpt_min"] - df.loc[clean_mask, "true_kpt_min"]
    axes[1].hist(error_after.dropna(), bins=100, color=COLORS["success"], alpha=0.7,
                 edgecolor="none", density=True)
    axes[1].axvline(x=0, color="#c9d1d9", linestyle="--", alpha=0.5)
    axes[1].axvline(x=error_after.mean(), color=COLORS["warning"], linestyle="-",
                    linewidth=2, label=f"Mean bias: {error_after.mean():.2f} min")
    axes[1].set_title("AFTER De-noising", color=COLORS["success"])
    axes[1].set_xlabel("Label Error (clean - true) [min]")
    axes[1].set_ylabel("Density")
    axes[1].legend(fontsize=10)
    axes[1].set_xlim(-15, 15)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    path = os.path.join(output_dir, "01_label_bias_distribution.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_02_theta_sweep(df: pd.DataFrame, output_dir: str) -> None:
    """
    Plot Precision, Recall, Filter Rate vs θ threshold (30s to 300s).
    
    Helps choose the optimal rider-delta threshold for Layer 1.
    
    Args:
        df: DataFrame with rider_delta_sec and is_rider_triggered_FOR columns.
        output_dir: Directory to save the chart.
    """
    thetas = np.arange(30, 310, 10)
    precisions, recalls, f1s, filter_rates = [], [], [], []
    
    actual = df["is_rider_triggered_FOR"].astype(bool)
    
    for theta in thetas:
        predicted = (df["rider_delta_sec"].abs() < theta)
        tp = (predicted & actual).sum()
        fp = (predicted & ~actual).sum()
        fn = (~predicted & actual).sum()
        
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0
        fr = predicted.mean()
        
        precisions.append(p)
        recalls.append(r)
        f1s.append(f)
        filter_rates.append(fr)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle("SIREN Layer 1: Theta (θ) Sweep — Rider-Delta Filter Tuning",
                 fontsize=16, fontweight="bold")
    
    ax.plot(thetas, precisions, color=COLORS["primary"], linewidth=2.5, 
            label="Precision", marker="o", markersize=3)
    ax.plot(thetas, recalls, color=COLORS["success"], linewidth=2.5, 
            label="Recall", marker="s", markersize=3)
    ax.plot(thetas, f1s, color=COLORS["purple"], linewidth=2.5, 
            label="F1 Score", marker="^", markersize=3)
    ax.plot(thetas, filter_rates, color=COLORS["warning"], linewidth=2,
            linestyle="--", label="Filter Rate", alpha=0.7)
    
    # Mark default θ=90
    ax.axvline(x=90, color=COLORS["danger"], linestyle=":", linewidth=1.5,
               label="θ=90s (default)")
    
    ax.set_xlabel("θ Threshold (seconds)")
    ax.set_ylabel("Score / Rate")
    ax.legend(fontsize=10, loc="center right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    
    plt.tight_layout()
    path = os.path.join(output_dir, "02_theta_sweep.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_03_rush_signal(df: pd.DataFrame, output_dir: str) -> None:
    """
    Plot Google busyness vs true KPT scatter + hourly KPT heatmap.
    
    Shows how external rush signals correlate with kitchen prep times.
    
    Args:
        df: DataFrame with google_busyness_index, true_kpt_min, hour, city.
        output_dir: Directory to save the chart.
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle("SIREN Layer 2: Rush Signal Analysis",
                 fontsize=16, fontweight="bold")
    
    # Scatter: busyness vs true KPT
    sample = df.sample(min(5000, len(df)), random_state=42)
    scatter = axes[0].scatter(
        sample["google_busyness_index"],
        sample["true_kpt_min"],
        c=sample["hour"],
        cmap="plasma",
        alpha=0.3,
        s=8,
        edgecolors="none"
    )
    plt.colorbar(scatter, ax=axes[0], label="Hour of Day")
    
    # Add regression line
    z = np.polyfit(sample["google_busyness_index"], sample["true_kpt_min"], 1)
    p = np.poly1d(z)
    x_line = np.linspace(0, 100, 100)
    axes[0].plot(x_line, p(x_line), color=COLORS["danger"], linewidth=2,
                 linestyle="--", label=f"Slope: {z[0]:.3f} min/busyness")
    
    axes[0].set_xlabel("Google Busyness Index")
    axes[0].set_ylabel("True KPT (min)")
    axes[0].set_title("Busyness vs Kitchen Prep Time")
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Heatmap: hour × city mean KPT
    pivot = df.pivot_table(
        values="true_kpt_min",
        index="city",
        columns="hour",
        aggfunc="mean"
    )
    sns.heatmap(
        pivot,
        ax=axes[1],
        cmap="YlOrRd",
        annot=False,
        cbar_kws={"label": "Mean KPT (min)"},
        linewidths=0.5,
        linecolor="#21262d"
    )
    axes[1].set_title("Hourly KPT Heatmap by City")
    axes[1].set_xlabel("Hour of Day")
    axes[1].set_ylabel("")
    
    plt.tight_layout()
    path = os.path.join(output_dir, "03_rush_signal.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_04_merchant_bias(df: pd.DataFrame, merchants_df: pd.DataFrame, 
                           output_dir: str) -> None:
    """
    Plot per-merchant bias offset distribution + correction by behavior class.
    
    Shows how SIREN Layer 3 identifies and corrects merchant-specific biases.
    
    Args:
        df: Orders DataFrame with bias_offset and behavior columns.
        merchants_df: Merchants DataFrame with behavior_class.
        output_dir: Directory to save the chart.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("SIREN Layer 3: Per-Merchant Bias Correction",
                 fontsize=16, fontweight="bold")
    
    # Per-merchant bias distribution
    merchant_bias = df.groupby("merchant_id")["bias_offset"].mean()
    axes[0].hist(merchant_bias, bins=50, color=COLORS["primary"], alpha=0.7,
                 edgecolor="none", density=True)
    axes[0].axvline(x=0, color="#c9d1d9", linestyle="--", alpha=0.5)
    axes[0].axvline(x=merchant_bias.mean(), color=COLORS["warning"], linewidth=2,
                    label=f"Mean: {merchant_bias.mean():.2f} min")
    axes[0].set_xlabel("Bias Offset (min)")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Distribution of Merchant Bias Offsets")
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # By behavior class
    behavior_col = "behavior_class_predicted" if "behavior_class_predicted" in df.columns else "behavior_class"
    if behavior_col not in df.columns:
        # Merge from merchants
        merged = df.merge(merchants_df[["merchant_id", "behavior_class"]], on="merchant_id", how="left")
        behavior_col = "behavior_class"
    else:
        merged = df
    
    behaviors = ["early_marker", "accurate_marker", "late_marker"]
    colors_beh = [COLORS["danger"], COLORS["success"], COLORS["warning"]]
    
    error_before = merged["measured_kpt_min"] - merged["true_kpt_min"]
    corrected = merged["measured_kpt_min"] - merged["bias_offset"]
    error_after = corrected - merged["true_kpt_min"]
    
    before_means = []
    after_means = []
    
    for beh in behaviors:
        mask = merged[behavior_col] == beh
        before_means.append(error_before[mask].mean())
        after_means.append(error_after[mask].mean())
    
    x_pos = np.arange(len(behaviors))
    width = 0.35
    
    bars1 = axes[1].bar(x_pos - width/2, before_means, width, 
                         color=COLORS["danger"], alpha=0.7, label="Before Correction")
    bars2 = axes[1].bar(x_pos + width/2, after_means, width,
                         color=COLORS["success"], alpha=0.7, label="After Correction")
    
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels([b.replace("_", "\n") for b in behaviors])
    axes[1].set_ylabel("Mean Error (min)")
    axes[1].set_title("Bias Correction by Behavior Class")
    axes[1].axhline(y=0, color="#c9d1d9", linestyle="--", alpha=0.5)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    path = os.path.join(output_dir, "04_merchant_bias.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_05_model_comparison(output_dir: str) -> None:
    """
    Generate 6-panel model comparison dashboard.
    
    Panels:
    1. Residual distributions
    2. Predicted vs True scatter
    3. CDF of absolute errors
    4. MAE bar chart
    5. Rider wait by tier
    6. Improvement percentages
    
    Args:
        output_dir: Directory to save the chart.
    """
    # Load test predictions
    preds = pd.read_csv(os.path.join(output_dir, "test_predictions.csv"))
    results = pd.read_csv(os.path.join(output_dir, "model_results.csv"))
    
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.3)
    
    fig.suptitle("SIREN Model Comparison Dashboard", fontsize=18, fontweight="bold", y=0.98)
    
    # 1. Residual distributions
    ax1 = fig.add_subplot(gs[0, 0])
    for col, name, color in [
        ("pred_baseline", "Baseline", COLORS["baseline"]),
        ("pred_siren_xgb", "SIREN XGB", COLORS["siren_xgb"]),
        ("pred_siren_lgbm", "SIREN LGBM", COLORS["siren_lgbm"])
    ]:
        residuals = preds[col] - preds["true_kpt"]
        ax1.hist(residuals, bins=80, alpha=0.4, label=name, color=color,
                 density=True, edgecolor="none")
    ax1.axvline(x=0, color="#c9d1d9", linestyle="--", alpha=0.5)
    ax1.set_xlabel("Residual (pred - true) [min]")
    ax1.set_ylabel("Density")
    ax1.set_title("Residual Distributions")
    ax1.legend(fontsize=9)
    ax1.set_xlim(-20, 20)
    ax1.grid(True, alpha=0.3)
    
    # 2. Predicted vs True scatter
    ax2 = fig.add_subplot(gs[0, 1])
    sample = preds.sample(min(3000, len(preds)), random_state=42)
    ax2.scatter(sample["true_kpt"], sample["pred_baseline"],
                alpha=0.15, s=5, color=COLORS["baseline"], label="Baseline")
    ax2.scatter(sample["true_kpt"], sample["pred_siren_xgb"],
                alpha=0.15, s=5, color=COLORS["siren_xgb"], label="SIREN XGB")
    ax2.plot([0, 60], [0, 60], color="#c9d1d9", linestyle="--", alpha=0.5)
    ax2.set_xlabel("True KPT (min)")
    ax2.set_ylabel("Predicted KPT (min)")
    ax2.set_title("Predicted vs True KPT")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 60)
    ax2.set_ylim(0, 60)
    
    # 3. CDF of absolute errors
    ax3 = fig.add_subplot(gs[0, 2])
    for col, name, color in [
        ("pred_baseline", "Baseline", COLORS["baseline"]),
        ("pred_siren_xgb", "SIREN XGB", COLORS["siren_xgb"]),
        ("pred_siren_lgbm", "SIREN LGBM", COLORS["siren_lgbm"])
    ]:
        ae = np.abs(preds[col] - preds["true_kpt"])
        sorted_ae = np.sort(ae)
        cdf = np.arange(1, len(sorted_ae) + 1) / len(sorted_ae)
        ax3.plot(sorted_ae, cdf, color=color, linewidth=2, label=name)
    ax3.set_xlabel("Absolute Error (min)")
    ax3.set_ylabel("Cumulative Probability")
    ax3.set_title("CDF of Absolute Errors")
    ax3.legend(fontsize=9)
    ax3.set_xlim(0, 25)
    ax3.grid(True, alpha=0.3)
    
    # 4. MAE bar chart
    ax4 = fig.add_subplot(gs[1, 0])
    models = results["model"].values
    maes = results["mae"].values
    bar_colors = [COLORS["baseline"], COLORS["siren_xgb"], COLORS["siren_lgbm"]]
    bars = ax4.bar(models, maes, color=bar_colors, alpha=0.85, edgecolor="none")
    for bar, mae in zip(bars, maes):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f"{mae:.2f}", ha="center", fontsize=10, fontweight="bold")
    ax4.set_ylabel("MAE (min)")
    ax4.set_title("Model MAE Comparison")
    ax4.set_xticklabels([m.replace("_", "\n") for m in models], fontsize=9)
    ax4.grid(True, alpha=0.3, axis="y")
    
    # 5. Metrics comparison radar (simplified as grouped bar)
    ax5 = fig.add_subplot(gs[1, 1])
    metrics_to_show = ["mae", "rmse", "p50_ae", "p90_ae", "rider_wait"]
    x_pos = np.arange(len(metrics_to_show))
    width = 0.25
    
    for i, (_, row) in enumerate(results.iterrows()):
        values = [row[m] for m in metrics_to_show]
        ax5.bar(x_pos + i * width, values, width, 
                color=bar_colors[i], alpha=0.8, label=row["model"])
    
    ax5.set_xticks(x_pos + width)
    ax5.set_xticklabels([m.upper().replace("_", "\n") for m in metrics_to_show], fontsize=8)
    ax5.set_ylabel("Minutes")
    ax5.set_title("All Metrics Comparison")
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3, axis="y")
    
    # 6. Improvement percentages
    ax6 = fig.add_subplot(gs[1, 2])
    baseline_row = results[results["model"] == "Baseline_XGBoost"].iloc[0]
    siren_row = results[results["model"] == "SIREN_XGBoost"].iloc[0]
    
    improvements = []
    for m in metrics_to_show:
        imp = ((baseline_row[m] - siren_row[m]) / baseline_row[m]) * 100
        improvements.append(imp)
    
    bars = ax6.barh(metrics_to_show, improvements, color=COLORS["success"], alpha=0.8)
    for bar, imp in zip(bars, improvements):
        ax6.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f"{imp:.1f}%", va="center", fontsize=10, fontweight="bold",
                color=COLORS["success"])
    ax6.set_xlabel("Improvement (%)")
    ax6.set_title("SIREN vs Baseline Improvement")
    ax6.grid(True, alpha=0.3, axis="x")
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(output_dir, "05_model_comparison.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_06_feature_importance(output_dir: str) -> None:
    """
    Plot feature importance from SIREN XGBoost, color-coded by SIREN layer.
    
    Args:
        output_dir: Directory to save the chart.
    """
    fi = pd.read_csv(os.path.join(output_dir, "feature_importance.csv"))
    fi_xgb = fi[fi["model"] == "SIREN_XGBoost"].sort_values("importance", ascending=True)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.suptitle("SIREN XGBoost — Feature Importance (Gain)",
                 fontsize=16, fontweight="bold")
    
    layer_colors = {
        "Base": COLORS["base"],
        "Layer 2": COLORS["layer2"],
        "Layer 3": COLORS["layer3"],
        "Unknown": "#8b949e"
    }
    
    colors = [layer_colors.get(layer, "#8b949e") for layer in fi_xgb["layer"]]
    
    bars = ax.barh(fi_xgb["feature"], fi_xgb["importance"], color=colors, alpha=0.85)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS["base"], alpha=0.85, label="Base Features"),
        Patch(facecolor=COLORS["layer2"], alpha=0.85, label="Layer 2 (Enrichment)"),
        Patch(facecolor=COLORS["layer3"], alpha=0.85, label="Layer 3 (Bias Correction)")
    ]
    ax.legend(handles=legend_elements, fontsize=10, loc="lower right")
    
    ax.set_xlabel("Feature Importance (Gain)")
    ax.grid(True, alpha=0.3, axis="x")
    
    plt.tight_layout()
    path = os.path.join(output_dir, "06_feature_importance.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_07_ablation(output_dir: str) -> None:
    """
    Plot ablation study results showing incremental contribution of each SIREN layer.

    Shows grouped bar chart: MAE, RMSE for each ablation variant.

    Args:
        output_dir: Directory to save the chart.
    """
    ablation_path = os.path.join(output_dir, "ablation_results.csv")
    if not os.path.exists(ablation_path):
        print("  ⚠ ablation_results.csv not found, skipping")
        return

    abl = pd.read_csv(ablation_path)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("SIREN Ablation Study — Incremental Layer Contributions",
                 fontsize=16, fontweight="bold")

    variants = abl["variant"].values
    x = np.arange(len(variants))
    layer_colors = [COLORS["baseline"], COLORS["layer1"], COLORS["layer2"], COLORS["layer3"]]
    colors = layer_colors[:len(variants)]

    # MAE bars
    bars = axes[0].bar(x, abl["mae"].values, color=colors, alpha=0.85, edgecolor="none")
    for bar, mae in zip(bars, abl["mae"].values):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f"{mae:.2f}", ha="center", fontsize=10, fontweight="bold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([v.replace("_", "\n") for v in variants], fontsize=9)
    axes[0].set_ylabel("MAE (min)")
    axes[0].set_title("MAE by Ablation Variant")
    axes[0].grid(True, alpha=0.3, axis="y")

    # Improvement waterfall
    baseline_mae = abl.iloc[0]["mae"]
    deltas = [0]
    for i in range(1, len(abl)):
        deltas.append(abl.iloc[i-1]["mae"] - abl.iloc[i]["mae"])

    bars2 = axes[1].bar(x[1:], deltas[1:], color=colors[1:], alpha=0.85)
    for bar, d in zip(bars2, deltas[1:]):
        pct = (d / baseline_mae) * 100
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f"{d:.2f}\n({pct:.0f}%)", ha="center", fontsize=9, fontweight="bold")
    axes[1].set_xticks(x[1:])
    axes[1].set_xticklabels([v.replace("_", "\n") for v in variants[1:]], fontsize=9)
    axes[1].set_ylabel("MAE Reduction (min)")
    axes[1].set_title("Incremental Improvement per Layer")
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = os.path.join(output_dir, "07_ablation_study.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def main():
    """Generate all 7 visualization charts."""
    print("=" * 60)
    print("SIREN Visualizations")
    print("=" * 60)

    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(base_dir, "data")
    outputs_dir = os.path.join(base_dir, "outputs")

    print("\nLoading data...")
    df = pd.read_csv(os.path.join(data_dir, "orders_100k.csv"))
    merchants_df = pd.read_csv(os.path.join(data_dir, "merchants_1k.csv"))
    print(f"Loaded {len(df):,} orders, {len(merchants_df):,} merchants")

    print("\nGenerating charts...")

    print("\n[1/7] Label bias distribution...")
    plot_01_label_bias(df, outputs_dir)

    print("[2/7] Theta sweep...")
    plot_02_theta_sweep(df, outputs_dir)

    print("[3/7] Rush signal analysis...")
    plot_03_rush_signal(df, outputs_dir)

    print("[4/7] Merchant bias correction...")
    plot_04_merchant_bias(df, merchants_df, outputs_dir)

    print("[5/7] Model comparison dashboard...")
    plot_05_model_comparison(outputs_dir)

    print("[6/7] Feature importance...")
    plot_06_feature_importance(outputs_dir)

    print("[7/7] Ablation study...")
    plot_07_ablation(outputs_dir)

    print("\n✅ All 7 visualizations saved to outputs/")


if __name__ == "__main__":
    main()
