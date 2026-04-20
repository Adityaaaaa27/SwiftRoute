"""
SIREN Layer 3 — Per-Merchant Bias Correction

Corrects systematic merchant marking biases using:
1. Merchant behavior classification (early/accurate/late marker)
2. KMeans clustering on merchant operational features
3. Rolling bias offset per merchant (last 30 clean orders)
4. Cold start handling (new merchants inherit cluster centroid bias)

Output columns: behavior_class, mean_marking_bias, bias_offset
Saves: models/layer3_kmeans.pkl, models/layer3_scaler.pkl
"""

import numpy as np
import pandas as pd
import os
import joblib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from typing import Dict, Tuple


def classify_merchant_behavior(df: pd.DataFrame, merchants_df: pd.DataFrame) -> pd.DataFrame:
    """
    Classify each merchant's marking behavior based on mean residual of clean orders.
    
    Categories:
    - early_marker: mean residual < -1.0 min (marks food ready BEFORE it's actually ready)
    - accurate_marker: -1.0 <= mean residual <= 1.0 min
    - late_marker: mean residual > 1.0 min (marks food ready AFTER it's been sitting)
    
    Args:
        df: Orders DataFrame with 'is_clean_label', 'measured_kpt_min', 'true_kpt_min'.
        merchants_df: Merchants DataFrame.
        
    Returns:
        Updated merchants DataFrame with 'predicted_behavior' column.
    """
    clean = df[df["is_clean_label"] == True].copy()
    
    # Compute residual: measured - true (positive = late marking)
    clean["residual"] = clean["measured_kpt_min"] - clean["true_kpt_min"]
    
    # Per-merchant mean residual
    merchant_residual = clean.groupby("merchant_id").agg(
        mean_residual=("residual", "mean"),
        std_residual=("residual", "std"),
        clean_count=("residual", "count")
    ).reset_index()
    
    # Classify
    def classify(row):
        """Classify merchant behavior based on mean residual."""
        if row["mean_residual"] < -1.0:
            return "early_marker"
        elif row["mean_residual"] > 1.0:
            return "late_marker"
        else:
            return "accurate_marker"
    
    merchant_residual["predicted_behavior"] = merchant_residual.apply(classify, axis=1)
    
    # Merge back to merchants
    merchants_df = merchants_df.merge(
        merchant_residual[["merchant_id", "predicted_behavior", "mean_residual", 
                           "std_residual", "clean_count"]],
        on="merchant_id",
        how="left"
    )
    
    # Fill NaN for merchants with no clean orders
    merchants_df["predicted_behavior"] = merchants_df["predicted_behavior"].fillna("accurate_marker")
    merchants_df["mean_residual"] = merchants_df["mean_residual"].fillna(0.0)
    merchants_df["std_residual"] = merchants_df["std_residual"].fillna(1.0)
    merchants_df["clean_count"] = merchants_df["clean_count"].fillna(0).astype(int)
    
    return merchants_df


def cluster_merchants(merchants_df: pd.DataFrame, n_clusters: int = 3) -> Tuple[pd.DataFrame, KMeans, StandardScaler]:
    """
    KMeans clustering on merchant operational features.
    
    Features used:
    - avg_daily_orders: Order volume
    - mean_residual: Average marking bias
    - std_residual: Bias consistency
    - bias_mean: Configured bias (from generator, simulates historical pattern)
    
    Args:
        merchants_df: Merchants DataFrame with computed features.
        n_clusters: Number of clusters (default 3 for early/accurate/late).
        
    Returns:
        Tuple of (updated merchants_df, fitted KMeans model, fitted StandardScaler).
    """
    features = ["avg_daily_orders", "mean_residual", "std_residual", "bias_mean"]
    
    X = merchants_df[features].fillna(0).values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    merchants_df["cluster_id"] = kmeans.fit_predict(X_scaled)
    
    # Compute cluster centroids in original space
    centroids = scaler.inverse_transform(kmeans.cluster_centers_)
    print("\n  Cluster centroids:")
    for i, centroid in enumerate(centroids):
        count = (merchants_df["cluster_id"] == i).sum()
        print(f"    Cluster {i}: orders={centroid[0]:.0f}, bias={centroid[1]:.2f}, "
              f"std={centroid[2]:.2f}, configured_bias={centroid[3]:.2f} ({count} merchants)")
    
    return merchants_df, kmeans, scaler


def compute_rolling_bias(df: pd.DataFrame, window: int = 30) -> pd.DataFrame:
    """
    Compute rolling bias offset per merchant from their last N clean orders.
    
    For each order, the bias offset is the rolling mean of
    (measured_kpt - true_kpt) over the last `window` clean orders for that merchant.
    
    This captures time-varying merchant behavior shifts.
    
    Args:
        df: Orders DataFrame sorted by timestamp.
        window: Number of recent clean orders to average (default 30).
        
    Returns:
        DataFrame with 'bias_offset' column added.
    """
    df = df.sort_values(["merchant_id", "order_timestamp"]).copy()
    
    # Compute per-order residual for clean orders only
    df["order_residual"] = np.where(
        df["is_clean_label"] == True,
        df["measured_kpt_min"] - df["true_kpt_min"],
        np.nan
    )
    
    # Rolling mean per merchant (only over clean orders)
    df["bias_offset"] = (
        df.groupby("merchant_id")["order_residual"]
        .transform(lambda x: x.rolling(window=window, min_periods=1).mean())
    )
    
    # Forward-fill bias_offset for non-clean orders (use last known bias)
    df["bias_offset"] = df.groupby("merchant_id")["bias_offset"].ffill()
    
    return df


def handle_cold_start(df: pd.DataFrame, merchants_df: pd.DataFrame, 
                       min_clean_orders: int = 10) -> pd.DataFrame:
    """
    Handle cold-start merchants with fewer than min_clean_orders clean orders.
    
    New merchants inherit their cluster centroid's mean bias as the bias_offset.
    
    Args:
        df: Orders DataFrame with 'bias_offset'.
        merchants_df: Merchants DataFrame with 'cluster_id' and 'mean_residual'.
        min_clean_orders: Minimum clean orders before trusting merchant's own bias.
        
    Returns:
        DataFrame with cold-start bias offsets filled.
    """
    # Compute cluster mean bias
    cluster_bias = merchants_df.groupby("cluster_id")["mean_residual"].mean()
    
    # Find cold-start merchants
    cold_merchants = merchants_df[merchants_df["clean_count"] < min_clean_orders]["merchant_id"].values
    
    if len(cold_merchants) > 0:
        # Map merchant → cluster → cluster bias
        merchant_cluster = merchants_df.set_index("merchant_id")["cluster_id"]
        
        for mid in cold_merchants:
            mask = df["merchant_id"] == mid
            cid = merchant_cluster.get(mid, 0)
            df.loc[mask, "bias_offset"] = df.loc[mask, "bias_offset"].fillna(cluster_bias.get(cid, 0))
    
    # Fill any remaining NaN bias offsets with 0
    df["bias_offset"] = df["bias_offset"].fillna(0.0)
    
    return df


def add_merchant_bias_features(df: pd.DataFrame, merchants_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add merchant-level bias features to the orders DataFrame.
    
    Adds: behavior_class, mean_marking_bias (from merchant profile)
    
    Args:
        df: Orders DataFrame.
        merchants_df: Merchants DataFrame with behavior classifications.
        
    Returns:
        DataFrame with merchant bias features.
    """
    merge_cols = ["merchant_id"]
    new_cols = ["predicted_behavior", "mean_residual", "cluster_id"]
    
    available = [c for c in new_cols if c in merchants_df.columns]
    
    df = df.merge(
        merchants_df[merge_cols + available],
        on="merchant_id",
        how="left",
        suffixes=("", "_merchant")
    )
    
    # Rename for clarity
    if "predicted_behavior" in df.columns:
        df = df.rename(columns={"predicted_behavior": "behavior_class_predicted"})
    if "mean_residual" in df.columns:
        df = df.rename(columns={"mean_residual": "mean_marking_bias"})
    
    return df


def evaluate_bias_correction(df: pd.DataFrame) -> Dict[str, float]:
    """
    Evaluate the effectiveness of bias correction.
    
    Computes RMSE of bias correction per merchant tier and overall.
    
    Args:
        df: DataFrame with bias_offset, measured_kpt_min, true_kpt_min.
        
    Returns:
        Dictionary with evaluation metrics.
    """
    metrics = {}
    
    # Overall
    corrected = df["measured_kpt_min"] - df["bias_offset"]
    error_before = df["measured_kpt_min"] - df["true_kpt_min"]
    error_after = corrected - df["true_kpt_min"]
    
    metrics["rmse_before"] = round(np.sqrt((error_before ** 2).mean()), 4)
    metrics["rmse_after"] = round(np.sqrt((error_after ** 2).mean()), 4)
    metrics["mae_before"] = round(error_before.abs().mean(), 4)
    metrics["mae_after"] = round(error_after.abs().mean(), 4)
    
    # Per tier
    for tier in df["merchant_tier"].unique():
        mask = df["merchant_tier"] == tier
        err_b = error_before[mask]
        err_a = error_after[mask]
        metrics[f"rmse_{tier}_before"] = round(np.sqrt((err_b ** 2).mean()), 4)
        metrics[f"rmse_{tier}_after"] = round(np.sqrt((err_a ** 2).mean()), 4)
    
    return metrics


def main():
    """Run Layer 3 bias correction on the enriched dataset."""
    print("=" * 60)
    print("SIREN Layer 3 — Per-Merchant Bias Correction")
    print("=" * 60)
    
    # Load data
    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(base_dir, "data")
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    orders_path = os.path.join(data_dir, "orders_100k.csv")
    merchants_path = os.path.join(data_dir, "merchants_1k.csv")
    
    print(f"\nLoading data...")
    df = pd.read_csv(orders_path)
    merchants_df = pd.read_csv(merchants_path)
    print(f"Loaded {len(df):,} orders, {len(merchants_df):,} merchants")
    
    # Step 1: Classify merchant behavior
    print("\n[1/5] Classifying merchant behavior...")
    merchants_df = classify_merchant_behavior(df, merchants_df)
    
    behavior_counts = merchants_df["predicted_behavior"].value_counts()
    print(f"  Behavior classification:")
    for beh, count in behavior_counts.items():
        print(f"    {beh}: {count} merchants")
    
    # Step 2: Cluster merchants
    print("\n[2/5] Clustering merchants...")
    merchants_df, kmeans, scaler = cluster_merchants(merchants_df)
    
    # Step 3: Add merchant features to orders
    print("\n[3/5] Adding merchant bias features...")
    df = add_merchant_bias_features(df, merchants_df)
    
    # Step 4: Compute rolling bias offset
    print("\n[4/5] Computing rolling bias offset...")
    df = compute_rolling_bias(df)
    
    # Step 5: Handle cold start
    print("\n[5/5] Handling cold-start merchants...")
    df = handle_cold_start(df, merchants_df)
    
    # Evaluate
    metrics = evaluate_bias_correction(df)
    
    print("\n━━━ LAYER 3 RESULTS ━━━")
    print(f"RMSE before correction: {metrics['rmse_before']:.3f} min")
    print(f"RMSE after correction:  {metrics['rmse_after']:.3f} min")
    print(f"MAE before correction:  {metrics['mae_before']:.3f} min")
    print(f"MAE after correction:   {metrics['mae_after']:.3f} min")
    
    for tier in ["large_chain", "mid_independent", "small_kiosk"]:
        key_b = f"rmse_{tier}_before"
        key_a = f"rmse_{tier}_after"
        if key_b in metrics:
            print(f"  {tier}: RMSE {metrics[key_b]:.3f} → {metrics[key_a]:.3f}")
    
    # Save models
    kmeans_path = os.path.join(models_dir, "layer3_kmeans.pkl")
    scaler_path = os.path.join(models_dir, "layer3_scaler.pkl")
    joblib.dump(kmeans, kmeans_path)
    joblib.dump(scaler, scaler_path)
    print(f"\nSaved KMeans model to {kmeans_path}")
    print(f"Saved Scaler to {scaler_path}")
    
    # Save processed data
    df.to_csv(orders_path, index=False)
    merchants_df.to_csv(merchants_path, index=False)
    print(f"Saved processed data")
    
    print("\n✅ Layer 3 complete!")
    return df, merchants_df, metrics


if __name__ == "__main__":
    main()
