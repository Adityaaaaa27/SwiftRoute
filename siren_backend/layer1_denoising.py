"""
SIREN Layer 1 — FOR Signal De-noising

De-noises measured KPT labels by detecting and filtering rider-triggered FOR events.

Three filtering mechanisms:
1. Rider-arrival delta filter: flags orders where rider_delta_sec < θ (default 90s)
2. Per-merchant KPT percentile check: flags if measured_kpt < merchant's 5th percentile
3. IoT beacon override: uses exact IoT KPT as ground truth when available

Output columns added to the DataFrame:
    flag_rider_delta, flag_percentile, flag_iot_override,
    is_clean_label, clean_kpt_min

Performance metrics computed against is_rider_triggered_FOR ground truth.
"""

import numpy as np
import pandas as pd
import os
from typing import Tuple, Dict


def rider_delta_filter(df: pd.DataFrame, theta: float = 90.0) -> pd.Series:
    """
    Flag orders where the rider arrival delta is suspiciously small.
    
    If a rider's GPS shows them arriving within θ seconds of the FOR mark,
    it strongly suggests the FOR was triggered by rider presence, not food readiness.
    
    Args:
        df: Orders DataFrame with 'rider_delta_sec' column.
        theta: Threshold in seconds. Orders with |rider_delta_sec| < θ are flagged.
        
    Returns:
        Boolean Series where True = flagged as rider-triggered.
    """
    return (df["rider_delta_sec"].abs() < theta).astype(bool)


def percentile_filter(df: pd.DataFrame, percentile: float = 5.0) -> pd.Series:
    """
    Flag orders where measured KPT is below the merchant's historical 5th percentile.
    
    Early/rider-triggered FORs often produce unrealistically low KPT values for a merchant.
    If measured_kpt is below the 5th percentile of that merchant's non-flagged orders,
    it's likely contaminated.
    
    Args:
        df: Orders DataFrame with 'merchant_id' and 'measured_kpt_min' columns.
        percentile: Percentile threshold (default 5th).
        
    Returns:
        Boolean Series where True = flagged as anomalously low KPT.
    """
    # Compute per-merchant percentile from all orders (initial pass)
    merchant_p5 = df.groupby("merchant_id")["measured_kpt_min"].quantile(
        percentile / 100.0
    ).rename("merchant_kpt_p5")
    
    merged = df.merge(merchant_p5, on="merchant_id", how="left")
    return (df["measured_kpt_min"] < merged["merchant_kpt_p5"]).astype(bool)


def iot_override(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """
    For merchants with IoT beacons, use the exact IoT KPT as ground truth.
    
    IoT shelf sensors provide millisecond-accurate timestamps of when food
    is placed on the pickup shelf, giving exact true KPT for beacon-equipped merchants.
    
    Args:
        df: Orders DataFrame with 'has_iot_beacon' and 'iot_kpt_min' columns.
        
    Returns:
        Tuple of (flag_series, iot_kpt_values).
        flag_series: True where IoT override is available.
        iot_kpt_values: IoT KPT where available, NaN elsewhere.
    """
    has_iot = (df["has_iot_beacon"] == 1) & df["iot_kpt_min"].notna()
    return has_iot.astype(bool), df["iot_kpt_min"]


def compute_clean_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine all three filters to produce clean KPT labels.
    
    Priority:
    1. If IoT beacon available → use iot_kpt_min (trusted ground truth)
    2. If neither rider-delta nor percentile flag → use measured_kpt_min (clean enough)
    3. If either flag triggered → label is NOT clean, exclude from training
    
    Args:
        df: Orders DataFrame with all required columns.
        
    Returns:
        DataFrame with additional columns:
            flag_rider_delta, flag_percentile, flag_iot_override,
            is_clean_label, clean_kpt_min
    """
    result = df.copy()
    
    # Apply filters
    result["flag_rider_delta"] = rider_delta_filter(df)
    result["flag_percentile"] = percentile_filter(df)
    result["flag_iot_override"], iot_values = iot_override(df)
    
    # Determine clean labels
    is_contaminated = result["flag_rider_delta"] | result["flag_percentile"]
    
    # IoT override takes priority — always clean
    result["is_clean_label"] = (~is_contaminated) | result["flag_iot_override"]
    
    # Set clean KPT values
    result["clean_kpt_min"] = np.where(
        result["flag_iot_override"],
        iot_values,
        np.where(
            result["is_clean_label"],
            result["measured_kpt_min"],
            np.nan
        )
    )
    
    return result


def evaluate_denoising(df: pd.DataFrame) -> Dict[str, float]:
    """
    Evaluate de-noising performance against ground truth.
    
    Computes precision, recall, F1 of the combined filter against the
    is_rider_triggered_FOR ground truth column.
    
    Also computes label bias before and after de-noising.
    
    Args:
        df: DataFrame with filter flags and ground truth columns.
        
    Returns:
        Dictionary with filter_rate, precision, recall, f1, bias_before, bias_after.
    """
    # Predicted contaminated = flagged by rider_delta OR percentile
    predicted_contaminated = df["flag_rider_delta"] | df["flag_percentile"]
    actual_contaminated = df["is_rider_triggered_FOR"].astype(bool)
    
    # Confusion matrix
    tp = (predicted_contaminated & actual_contaminated).sum()
    fp = (predicted_contaminated & ~actual_contaminated).sum()
    fn = (~predicted_contaminated & actual_contaminated).sum()
    tn = (~predicted_contaminated & ~actual_contaminated).sum()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    filter_rate = predicted_contaminated.mean()
    
    # Bias metrics
    bias_before = (df["measured_kpt_min"] - df["true_kpt_min"]).mean()
    
    clean_mask = df["is_clean_label"]
    if clean_mask.sum() > 0:
        bias_after = (df.loc[clean_mask, "clean_kpt_min"] - df.loc[clean_mask, "true_kpt_min"]).mean()
    else:
        bias_after = float("nan")
    
    metrics = {
        "filter_rate": round(filter_rate, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "bias_before": round(bias_before, 4),
        "bias_after": round(bias_after, 4),
        "total_orders": len(df),
        "clean_orders": int(clean_mask.sum()),
        "filtered_orders": int(predicted_contaminated.sum()),
        "iot_overrides": int(df["flag_iot_override"].sum()),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn)
    }
    
    return metrics


def main():
    """Run Layer 1 de-noising on the generated dataset."""
    print("=" * 60)
    print("SIREN Layer 1 — FOR Signal De-noising")
    print("=" * 60)
    
    # Load data
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    orders_path = os.path.join(data_dir, "orders_100k.csv")
    
    print(f"\nLoading {orders_path}...")
    df = pd.read_csv(orders_path)
    print(f"Loaded {len(df):,} orders")
    
    # Apply de-noising
    print("\nApplying de-noising filters...")
    df_clean = compute_clean_labels(df)
    
    # Evaluate
    metrics = evaluate_denoising(df_clean)
    
    print("\n━━━ LAYER 1 RESULTS ━━━")
    print(f"Filter rate:     {metrics['filter_rate']:.1%}")
    print(f"Precision:       {metrics['precision']:.4f}")
    print(f"Recall:          {metrics['recall']:.4f}")
    print(f"F1 Score:        {metrics['f1']:.4f}")
    print(f"Bias BEFORE:     {metrics['bias_before']:.3f} min")
    print(f"Bias AFTER:      {metrics['bias_after']:.3f} min")
    print(f"Clean orders:    {metrics['clean_orders']:,} / {metrics['total_orders']:,} "
          f"({metrics['clean_orders']/metrics['total_orders']:.1%})")
    print(f"IoT overrides:   {metrics['iot_overrides']:,}")
    
    # Save processed data
    output_path = os.path.join(data_dir, "orders_100k.csv")
    df_clean.to_csv(output_path, index=False)
    print(f"\nSaved de-noised data to {output_path}")
    
    print("\n✅ Layer 1 complete!")
    return df_clean, metrics


if __name__ == "__main__":
    main()
