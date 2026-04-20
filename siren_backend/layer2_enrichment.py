"""
SIREN Layer 2 — External Signal Enrichment

Enriches the order dataset with computed features derived from external signals:
- Rush multiplier from Google busyness index
- Relative rush (per city-hour normalization)
- Weather impact on KPT
- Cyclical time features (hour_sin, hour_cos, dow_sin, dow_cos)
- Cuisine KPT prior lookup
- Order complexity penalty

Prints Pearson correlation of each enrichment signal with true_kpt_min.
"""

import numpy as np
import pandas as pd
import os
from scipy import stats
from typing import Dict, List, Tuple


# Cuisine base KPT lookup table (minutes)
CUISINE_BASE_KPT = {
    "Indian": 18, "Biryani": 22, "Chinese": 15, "Pizza": 12,
    "Burger": 8, "South_Indian": 14, "Thai": 14, "Mughlai": 20,
    "Desserts": 6, "Healthy": 8
}


def add_rush_multiplier(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rush multiplier from Google busyness index.
    
    Formula: rush_multiplier = 1.0 + 1.5 × (google_busyness_index / 100)
    
    Higher busyness → longer kitchen prep times due to order volume.
    
    Args:
        df: Orders DataFrame with 'google_busyness_index' column.
        
    Returns:
        DataFrame with 'rush_multiplier' column added.
    """
    df["rush_multiplier"] = 1.0 + 1.5 * (df["google_busyness_index"] / 100.0)
    return df


def add_relative_rush(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute relative rush by normalizing busyness against city-hour averages.
    
    This captures whether a specific order's rush level is above or below
    what's typical for that city at that hour.
    
    Formula: relative_rush = busyness / city_hour_mean_busyness
    
    Args:
        df: Orders DataFrame with 'google_busyness_index', 'city', 'hour' columns.
        
    Returns:
        DataFrame with 'relative_rush' column added.
    """
    city_hour_mean = df.groupby(["city", "hour"])["google_busyness_index"].transform("mean")
    # Avoid division by zero
    city_hour_mean = city_hour_mean.replace(0, 1)
    df["relative_rush"] = df["google_busyness_index"] / city_hour_mean
    return df


def add_weather_impact(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rain's impact on KPT.
    
    Rain increases kitchen prep time through multiple mechanisms:
    - Delivery delays cause order backlog
    - Higher order volumes during rain
    - Supply chain slowdowns
    
    Formula: rain_kpt_impact = rain_flag × rain_severity × 3.5
    
    Args:
        df: Orders DataFrame with 'rain_flag' and 'rain_severity' columns.
        
    Returns:
        DataFrame with 'rain_kpt_impact' column added.
    """
    df["rain_kpt_impact"] = df["rain_flag"] * df["rain_severity"] * 3.5
    return df


def add_cyclical_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add cyclical time features using sine/cosine encoding.
    
    Cyclical encoding preserves the circular nature of time:
    - Hour 23 is close to hour 0
    - Sunday is close to Monday
    
    Features: hour_sin, hour_cos, dow_sin, dow_cos
    
    Args:
        df: Orders DataFrame with 'hour' and 'day_of_week' columns.
        
    Returns:
        DataFrame with cyclical time features added.
    """
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    return df


def add_cuisine_prior(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add cuisine base KPT as a feature (lookup table).
    
    Provides the model with domain knowledge about typical prep times
    per cuisine type.
    
    Args:
        df: Orders DataFrame with 'cuisine' column.
        
    Returns:
        DataFrame with 'cuisine_base_kpt' column added.
    """
    df["cuisine_base_kpt"] = df["cuisine"].map(CUISINE_BASE_KPT)
    return df


def add_complexity_penalty(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute order complexity KPT penalty.
    
    More items = more prep time. Penalty is linear with item count.
    
    Formula: complexity_kpt_penalty = (order_complexity - 1) × 2.5
    
    Args:
        df: Orders DataFrame with 'order_complexity' column.
        
    Returns:
        DataFrame with 'complexity_kpt_penalty' column added.
    """
    df["complexity_kpt_penalty"] = (df["order_complexity"] - 1) * 2.5
    return df


def enrich_all(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all Layer 2 enrichment steps to the DataFrame.
    
    Args:
        df: Orders DataFrame after Layer 1 de-noising.
        
    Returns:
        Enriched DataFrame with all new signal columns.
    """
    df = add_rush_multiplier(df)
    df = add_relative_rush(df)
    df = add_weather_impact(df)
    df = add_cyclical_time(df)
    df = add_cuisine_prior(df)
    df = add_complexity_penalty(df)
    return df


def compute_correlations(df: pd.DataFrame) -> Dict[str, float]:
    """
    Compute Pearson correlation of each enrichment signal with true_kpt_min.
    
    Args:
        df: Enriched orders DataFrame with 'true_kpt_min' column.
        
    Returns:
        Dictionary mapping signal name to Pearson r value.
    """
    signals = [
        "rush_multiplier", "relative_rush", "rain_kpt_impact",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos",
        "cuisine_base_kpt", "complexity_kpt_penalty",
        "google_busyness_index", "rain_flag", "rain_severity",
        "order_complexity", "zomato_concurrent_orders"
    ]
    
    correlations = {}
    for signal in signals:
        if signal in df.columns:
            mask = df[signal].notna() & df["true_kpt_min"].notna()
            if mask.sum() > 2:
                r, p = stats.pearsonr(df.loc[mask, signal], df.loc[mask, "true_kpt_min"])
                correlations[signal] = round(r, 4)
    
    return correlations


def main():
    """Run Layer 2 enrichment on the de-noised dataset."""
    print("=" * 60)
    print("SIREN Layer 2 — External Signal Enrichment")
    print("=" * 60)
    
    # Load data
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    orders_path = os.path.join(data_dir, "orders_100k.csv")
    
    print(f"\nLoading {orders_path}...")
    df = pd.read_csv(orders_path)
    print(f"Loaded {len(df):,} orders")
    
    # Check if Layer 1 columns exist
    if "is_clean_label" not in df.columns:
        print("WARNING: Layer 1 columns not found. Run layer1_denoising.py first.")
        return None, {}
    
    # Apply enrichment
    print("\nApplying enrichment features...")
    df = enrich_all(df)
    
    # Compute correlations
    correlations = compute_correlations(df)
    
    print("\n━━━ LAYER 2 — SIGNAL CORRELATIONS WITH true_kpt_min ━━━")
    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    for signal, r in sorted_corr:
        bar = "█" * int(abs(r) * 40)
        sign = "+" if r > 0 else "-"
        print(f"  {signal:<28s} r = {sign}{abs(r):.4f}  {bar}")
    
    # Summary stats on new features
    print("\n━━━ ENRICHMENT FEATURE STATS ━━━")
    new_features = [
        "rush_multiplier", "relative_rush", "rain_kpt_impact",
        "cuisine_base_kpt", "complexity_kpt_penalty"
    ]
    for feat in new_features:
        print(f"  {feat:<28s} mean={df[feat].mean():.3f}  std={df[feat].std():.3f}  "
              f"min={df[feat].min():.3f}  max={df[feat].max():.3f}")
    
    # Save enriched data
    output_path = os.path.join(data_dir, "orders_100k.csv")
    df.to_csv(output_path, index=False)
    print(f"\nSaved enriched data to {output_path}")
    
    print("\n✅ Layer 2 complete!")
    return df, correlations


if __name__ == "__main__":
    main()
