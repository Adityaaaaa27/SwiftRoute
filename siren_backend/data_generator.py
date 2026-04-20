"""
SIREN Data Generator v2.0 — Realistic Synthetic Dataset

Key improvements over v1.0:
  - Hidden merchant_skill factor (0.75-1.25) creates irreducible prediction error
  - Rush effect reduced (max ~1.35 vs old 2.5) for realistic KPT values (~20 min mean)
  - Complexity is additive, not multiplicative
  - Wider rider_delta distribution → Layer 1 F1 target ~0.88-0.92 (not 0.99)
  - 10% of clean orders have small rider_delta (false positive challenge)
  - Increased noise (std=2.5 vs 1.5)
  - Layer 2 features are PROXIES with different coefficients than the true formula

Output:
    data/orders_100k.csv  — 100,000 order records
    data/merchants_1k.csv — 1,000 merchant metadata records
"""

import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

np.random.seed(42)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONSTANTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

NUM_ORDERS = 100_000
NUM_MERCHANTS = 1_000
DATE_START = datetime(2023, 1, 1)
DATE_END = datetime(2024, 12, 31)

CITIES = [
    "Bangalore", "Mumbai", "Delhi", "Hyderabad", "Pune",
    "Chennai", "Kolkata", "Ahmedabad", "Surat", "Jaipur"
]
CITY_WEIGHTS = [0.18, 0.16, 0.15, 0.12, 0.10, 0.08, 0.07, 0.05, 0.05, 0.04]

CUISINES = {
    "Indian": 18, "Biryani": 22, "Chinese": 15, "Pizza": 12,
    "Burger": 8, "South_Indian": 14, "Thai": 14, "Mughlai": 20,
    "Desserts": 6, "Healthy": 8
}
CUISINE_WEIGHTS = [0.20, 0.12, 0.15, 0.12, 0.10, 0.08, 0.05, 0.06, 0.05, 0.07]

MERCHANT_TIERS = {"large_chain": 100, "mid_independent": 400, "small_kiosk": 500}
MERCHANT_BEHAVIOR = {"early_marker": 0.25, "accurate_marker": 0.55, "late_marker": 0.20}

RIDER_TRIGGERED_RATE = 0.38
COMPLEXITY_DIST = {1: 0.40, 2: 0.30, 3: 0.20, 4: 0.10}
RAIN_PROBABILITY = 0.20

# Real-world IoT beacon penetration by tier (India, 2024)
# Large chains: ~40% (e.g. McDonald's, Domino's — only flagship outlets)
# Mid-independent: ~3% (rare early adopters)
# Small kiosk: 0% (not economically viable)
IOT_BEACON_PROBS = {
    "large_chain":      0.40,
    "mid_independent":  0.03,
    "small_kiosk":      0.00,
}

CITY_BUSYNESS_BASE = {
    "Bangalore": 55, "Mumbai": 65, "Delhi": 60, "Hyderabad": 50,
    "Pune": 45, "Chennai": 48, "Kolkata": 52, "Ahmedabad": 40,
    "Surat": 35, "Jaipur": 38
}

# ━━━━ DATA GENERATION FORMULA COEFFICIENTS ━━━━
# NOTE: These are DIFFERENT from Layer 2's proxy coefficients.
# Layer 2 uses rush_mult = 1 + 1.5*busyness/100 (a proxy).
# Reality (here) uses 1 + 0.25*busyness/100 + 0.03*concurrent.
# This deliberate mismatch prevents circular feature→target mapping.
TRUE_RUSH_BUSYNESS_COEFF = 0.25      # Layer 2 proxy uses 1.5
TRUE_RUSH_CONCURRENT_COEFF = 0.03    # Layer 2 proxy doesn't include this
TRUE_COMPLEXITY_COEFF = 2.0          # Layer 2 proxy uses 2.5
TRUE_RAIN_COEFF = 2.0                # Layer 2 proxy uses 3.5
TRUE_NOISE_STD = 2.5                 # was 1.5 in v1


def generate_merchants() -> pd.DataFrame:
    """Generate 1,000 merchant profiles with hidden skill factor."""
    merchants = []
    merchant_id = 0

    for tier, count in MERCHANT_TIERS.items():
        for _ in range(count):
            city = np.random.choice(CITIES, p=CITY_WEIGHTS)
            cuisine = np.random.choice(list(CUISINES.keys()), p=CUISINE_WEIGHTS)
            behavior = np.random.choice(
                list(MERCHANT_BEHAVIOR.keys()),
                p=list(MERCHANT_BEHAVIOR.values())
            )

            if behavior == "early_marker":
                bias_mean = np.random.uniform(-4.0, -1.5)
                bias_std = np.random.uniform(0.3, 0.8)
            elif behavior == "accurate_marker":
                bias_mean = np.random.uniform(-0.5, 0.5)
                bias_std = np.random.uniform(0.1, 0.4)
            else:
                bias_mean = np.random.uniform(1.5, 5.0)
                bias_std = np.random.uniform(0.3, 1.0)

            # Realistic IoT: only a fraction of merchants have shelf sensors
            has_iot = np.random.random() < IOT_BEACON_PROBS[tier]

            # Human-readable merchant name
            city_short = city[:3]
            tier_short = {"large_chain": "LC", "mid_independent": "MI", "small_kiosk": "SK"}[tier]
            merchant_name = f"{cuisine} {city} #{merchant_id} ({tier_short})"

            if tier == "large_chain":
                avg_daily_orders = np.random.randint(80, 200)
            elif tier == "mid_independent":
                avg_daily_orders = np.random.randint(20, 80)
            else:
                avg_daily_orders = np.random.randint(5, 30)

            # HIDDEN FACTOR: merchant skill (0.75-1.25)
            # Not used as a model feature — creates irreducible prediction error.
            # Represents unmeasured factors: chef experience, kitchen layout, etc.
            merchant_skill = np.random.uniform(0.75, 1.25)

            merchants.append({
                "merchant_id": f"M{merchant_id:04d}",
                "merchant_name": merchant_name,
                "merchant_tier": tier,
                "cuisine": cuisine,
                "city": city,
                "behavior_class": behavior,
                "bias_mean": round(bias_mean, 3),
                "bias_std": round(bias_std, 3),
                "has_iot_beacon": has_iot,
                "avg_daily_orders": avg_daily_orders,
                "merchant_skill": round(merchant_skill, 4),  # HIDDEN — not a model feature
            })
            merchant_id += 1

    df = pd.DataFrame(merchants)
    print(f"Generated {len(df)} merchants:")
    print(f"  Tiers: {df['merchant_tier'].value_counts().to_dict()}")
    print(f"  Behaviors: {df['behavior_class'].value_counts().to_dict()}")
    print(f"  IoT beacons: {df['has_iot_beacon'].sum()}")
    print(f"  Merchant skill: mean={df['merchant_skill'].mean():.3f}, "
          f"std={df['merchant_skill'].std():.3f}")
    return df


def compute_google_busyness(hour: int, day_of_week: int, city: str, rain: bool) -> float:
    """Simulate Google Maps Popular Times busyness index (0-100)."""
    base = CITY_BUSYNESS_BASE.get(city, 45)

    if 12 <= hour <= 14:
        hour_factor = 1.6
    elif 19 <= hour <= 21:
        hour_factor = 1.8
    elif 11 <= hour <= 15:
        hour_factor = 1.3
    elif 18 <= hour <= 22:
        hour_factor = 1.4
    elif 7 <= hour <= 9:
        hour_factor = 0.8
    else:
        hour_factor = 0.4

    weekend_factor = 1.15 if day_of_week >= 5 else 1.0
    if day_of_week == 4 and hour >= 18:
        weekend_factor = 1.25

    rain_factor = 1.1 if rain else 1.0
    busyness = base * hour_factor * weekend_factor * rain_factor
    busyness += np.random.normal(0, 5)
    return np.clip(busyness, 0, 100)


def generate_orders(merchants_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate 100,000 orders with realistic bias mechanisms.

    Key design choices for research validity:
    - true_kpt uses coefficients DIFFERENT from Layer 2 proxy features
    - merchant_skill is a hidden confound (not available to models)
    - Rider-triggered contamination has wider delta → F1 ~0.90
    - 10% of clean orders have small rider_delta → false positives
    """
    orders = []
    merchant_ids = merchants_df["merchant_id"].values
    merchant_weights = merchants_df["avg_daily_orders"].values.astype(float)
    merchant_weights /= merchant_weights.sum()

    total_seconds = int((DATE_END - DATE_START).total_seconds())

    for i in range(NUM_ORDERS):
        m_idx = np.random.choice(len(merchant_ids), p=merchant_weights)
        m = merchants_df.iloc[m_idx]

        random_seconds = np.random.randint(0, total_seconds)
        order_ts = DATE_START + timedelta(seconds=random_seconds)
        hour = order_ts.hour
        if hour < 8:
            hour = np.random.randint(8, 23)
            order_ts = order_ts.replace(hour=hour)

        day_of_week = order_ts.weekday()
        is_weekend = day_of_week >= 5
        is_lunch_rush = 12 <= hour <= 14
        is_dinner_rush = 19 <= hour <= 21

        complexity = np.random.choice(
            list(COMPLEXITY_DIST.keys()), p=list(COMPLEXITY_DIST.values())
        )

        base_concurrent = m["avg_daily_orders"] / 14
        rush_mult = 1.8 if is_lunch_rush or is_dinner_rush else 1.0
        concurrent = max(1, int(np.random.poisson(base_concurrent * rush_mult)))

        rain_flag = np.random.random() < RAIN_PROBABILITY
        rain_severity = np.random.uniform(0.3, 1.0) if rain_flag else 0.0
        busyness = compute_google_busyness(hour, day_of_week, m["city"], rain_flag)

        # ━━ TRUE KPT — uses DIFFERENT coefficients than Layer 2 proxies ━━
        cuisine_base = CUISINES[m["cuisine"]]
        rush_effect = (1.0
                       + TRUE_RUSH_BUSYNESS_COEFF * (busyness / 100)
                       + TRUE_RUSH_CONCURRENT_COEFF * min(concurrent, 10))

        # Hidden merchant_skill — NOT available to the model
        skill = m["merchant_skill"]

        # Additive complexity (not multiplicative as in v1)
        complexity_add = (complexity - 1) * TRUE_COMPLEXITY_COEFF
        rain_add = rain_severity * TRUE_RAIN_COEFF if rain_flag else 0.0

        # Per-order latent variation (ingredient freshness, staff mood, etc.)
        latent_variation = np.random.normal(0, 1.0)

        true_kpt = (
            cuisine_base * rush_effect * skill  # base × rush × skill (hidden)
            + complexity_add
            + rain_add
            + latent_variation                   # hidden per-order factor
            + np.random.normal(0, TRUE_NOISE_STD)
        )
        true_kpt = max(2.0, true_kpt)

        # ━━ MERCHANT BIAS INJECTION ━━
        merchant_bias = np.random.normal(m["bias_mean"], m["bias_std"])

        # ━━ RIDER-TRIGGERED FOR ━━
        is_rider_triggered = np.random.random() < RIDER_TRIGGERED_RATE
        rider_arrival = np.random.uniform(8, 28)

        if is_rider_triggered:
            # Rider triggers FOR — measured_kpt collapses to rider_arrival
            measured_kpt = rider_arrival + np.random.normal(0, 1.0)
            # Rider delta is small (~Normal(0, 55)) — harder to detect than v1
            rider_delta_sec = np.random.normal(0, 55)
            rider_delta_sec = np.clip(rider_delta_sec, -300, 300)
        else:
            # Normal FOR — merchant marks when food is ready
            measured_kpt = true_kpt + merchant_bias
            # 90% have large delta, 10% have accidentally small delta (false positives)
            if np.random.random() < 0.10:
                rider_delta_sec = np.random.uniform(-150, 150)
            else:
                rider_delta_sec = np.random.choice([
                    np.random.uniform(-600, -200),
                    np.random.uniform(200, 600),
                ])

        measured_kpt = max(1.0, measured_kpt)

        # ━━ IoT BEACON ━━
        has_iot = m["has_iot_beacon"]
        iot_kpt = true_kpt + np.random.normal(0, 0.5) if has_iot else np.nan

        orders.append({
            "order_id": f"ORD{i:06d}",
            "merchant_id": m["merchant_id"],
            "merchant_tier": m["merchant_tier"],
            "cuisine": m["cuisine"],
            "city": m["city"],
            "order_timestamp": order_ts.strftime("%Y-%m-%d %H:%M:%S"),
            "hour": hour,
            "day_of_week": day_of_week,
            "is_lunch_rush": int(is_lunch_rush),
            "is_dinner_rush": int(is_dinner_rush),
            "is_weekend": int(is_weekend),
            "order_complexity": complexity,
            "zomato_concurrent_orders": concurrent,
            "google_busyness_index": round(busyness, 2),
            "rain_flag": int(rain_flag),
            "rain_severity": round(rain_severity, 3),
            "rider_arrival_minutes": round(rider_arrival, 2),
            "rider_delta_sec": round(rider_delta_sec, 1),
            "measured_kpt_min": round(measured_kpt, 2),
            "true_kpt_min": round(true_kpt, 2),
            "is_rider_triggered_FOR": int(is_rider_triggered),
            "has_iot_beacon": int(has_iot),
            "iot_kpt_min": round(iot_kpt, 2) if not np.isnan(iot_kpt) else np.nan
        })

        if (i + 1) % 20000 == 0:
            print(f"  Generated {i + 1:,} / {NUM_ORDERS:,} orders...")

    df = pd.DataFrame(orders)
    return df


def validate_dataset(orders_df: pd.DataFrame, merchants_df: pd.DataFrame) -> None:
    """Validate generated dataset against expected properties."""
    print("\n━━━ DATASET VALIDATION ━━━")
    print(f"Orders: {len(orders_df):,} rows, {len(orders_df.columns)} columns")
    print(f"Merchants: {len(merchants_df):,} rows")

    rt_rate = orders_df["is_rider_triggered_FOR"].mean()
    print(f"Rider-triggered FOR rate: {rt_rate:.1%} (target: 38%)")

    iot_rate = orders_df["has_iot_beacon"].mean()
    print(f"IoT beacon coverage: {iot_rate:.1%}")

    bias = orders_df["measured_kpt_min"] - orders_df["true_kpt_min"]
    print(f"Label bias (measured - true): mean={bias.mean():.2f}, std={bias.std():.2f}")

    merged = orders_df.merge(
        merchants_df[["merchant_id", "behavior_class"]], on="merchant_id"
    )
    for behavior in ["early_marker", "accurate_marker", "late_marker"]:
        mask = merged["behavior_class"] == behavior
        b = bias[mask]
        print(f"  {behavior}: mean bias = {b.mean():.2f} min")

    print(f"\nTrue KPT: mean={orders_df['true_kpt_min'].mean():.1f}, "
          f"median={orders_df['true_kpt_min'].median():.1f}, "
          f"std={orders_df['true_kpt_min'].std():.1f}")
    print(f"Measured KPT: mean={orders_df['measured_kpt_min'].mean():.1f}, "
          f"median={orders_df['measured_kpt_min'].median():.1f}, "
          f"std={orders_df['measured_kpt_min'].std():.1f}")

    dates = pd.to_datetime(orders_df["order_timestamp"])
    print(f"Date range: {dates.min()} to {dates.max()}")

    # Contamination difficulty preview
    rt_mask = orders_df["is_rider_triggered_FOR"] == 1
    rt_delta = orders_df.loc[rt_mask, "rider_delta_sec"].abs()
    clean_delta = orders_df.loc[~rt_mask, "rider_delta_sec"].abs()
    print(f"\nContamination detection difficulty:")
    print(f"  RT orders |delta| < 90s: {(rt_delta < 90).mean():.1%}")
    print(f"  Clean orders |delta| < 90s: {(clean_delta < 90).mean():.1%} (false positives)")


def main():
    """Generate and save the complete SIREN dataset."""
    print("=" * 60)
    print("SIREN Data Generator v2.0")
    print("=" * 60)

    data_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(data_dir, exist_ok=True)

    print("\n[1/3] Generating 1,000 merchants...")
    merchants_df = generate_merchants()

    print(f"\n[2/3] Generating {NUM_ORDERS:,} orders...")
    orders_df = generate_orders(merchants_df)

    validate_dataset(orders_df, merchants_df)

    orders_path = os.path.join(data_dir, "orders_100k.csv")
    merchants_path = os.path.join(data_dir, "merchants_1k.csv")

    orders_df.to_csv(orders_path, index=False)
    merchants_df.to_csv(merchants_path, index=False)

    print(f"\n[3/3] Saved:")
    print(f"  {orders_path} ({os.path.getsize(orders_path) / 1e6:.1f} MB)")
    print(f"  {merchants_path} ({os.path.getsize(merchants_path) / 1e3:.1f} KB)")
    print("\n✅ Data generation complete!")


if __name__ == "__main__":
    main()
