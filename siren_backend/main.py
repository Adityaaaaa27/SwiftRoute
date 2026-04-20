"""
SIREN FastAPI Backend Server — Zomato Kitchen Prep Time Prediction

Endpoints:
    GET  /api/health
    GET  /api/dashboard/summary
    GET  /api/orders/live
    POST /api/predict
    GET  /api/merchants
    GET  /api/merchants/{merchant_id}
    GET  /api/signal-quality
    GET  /api/model-results
    GET  /api/simulation
    GET  /api/merchants/{merchant_id}/export

Loads trained .pkl model files at startup via lifespan context.
All data served from the actual generated/trained dataset.
"""

import numpy as np
import pandas as pd
import os
import json
import io
import joblib
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GLOBALS — loaded at startup
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

# State dict populated at startup
state: Dict[str, Any] = {}

# Cuisine base KPT lookup (same as in the pipeline)
CUISINE_BASE_KPT = {
    "Indian": 18, "Biryani": 22, "Chinese": 15, "Pizza": 12,
    "Burger": 8, "South_Indian": 14, "Thai": 14, "Mughlai": 20,
    "Desserts": 6, "Healthy": 8,
}

# SIREN features list (must match model_training.py)
BASELINE_FEATURES = [
    "hour", "day_of_week", "is_lunch_rush", "is_dinner_rush", "is_weekend",
    "order_complexity", "zomato_concurrent_orders",
]

SIREN_FEATURES = BASELINE_FEATURES + [
    "rush_multiplier", "relative_rush",
    "rain_flag", "rain_kpt_impact",
    "complexity_kpt_penalty", "cuisine_base_kpt",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "mean_marking_bias", "bias_offset",
    "google_busyness_index",
]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PYDANTIC MODELS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    dataset_rows: int
    trained_at: str


class DashboardSummary(BaseModel):
    """Dashboard summary response."""
    avg_kpt_today: float
    rider_wait_avg: float
    p50_eta_error: float
    p90_eta_error: float
    total_orders_today: int
    on_time_rate: float
    city_rush_index: Dict[str, float]


class LiveOrder(BaseModel):
    """Single live order in the live-orders list."""
    order_id: str
    merchant_name: str
    cuisine: str
    predicted_kpt: float
    actual_kpt: float
    status: str
    rider_assigned: bool
    for_signal_quality: str


class PredictRequest(BaseModel):
    """Prediction request body."""
    merchant_id: str
    cuisine: str
    order_complexity: int = Field(ge=1, le=10)
    hour: int = Field(ge=0, le=23)
    day_of_week: int = Field(ge=0, le=6)
    google_busyness_index: float = Field(ge=0, le=100)
    rain_flag: int = Field(ge=0, le=1)
    rain_severity: float = Field(ge=0, le=1)


class FeatureContribution(BaseModel):
    """Single feature contribution in prediction response."""
    feature: str
    value: float
    importance: float


class PredictResponse(BaseModel):
    """Prediction response."""
    predicted_kpt_min: float
    confidence_interval: List[float]
    for_signal_warning: bool
    recommended_dispatch_offset_min: float
    feature_contributions: List[FeatureContribution]


class MerchantSummary(BaseModel):
    """Merchant entry in the paginated list."""
    merchant_id: str
    merchant_name: str
    merchant_tier: str
    cuisine: str
    city: str
    behavior_class: str
    avg_daily_orders: int
    has_iot_beacon: bool
    bias_mean: float


class MerchantDetail(BaseModel):
    """Detailed merchant information."""
    merchant_id: str
    merchant_name: str
    merchant_tier: str
    cuisine: str
    city: str
    behavior_class: str
    avg_daily_orders: int
    has_iot_beacon: bool
    bias_mean: float
    recent_orders: List[Dict[str, Any]]
    kpt_distribution: Dict[str, float]
    bias_trend: List[float]


class SignalQualityResponse(BaseModel):
    """Signal quality metrics."""
    layer1: Dict[str, Any]
    layer2: Dict[str, Any]
    layer3: Dict[str, Any]


class ModelResultsResponse(BaseModel):
    """Model comparison results."""
    models: List[Dict[str, Any]]
    feature_importance: List[Dict[str, Any]]
    tier_breakdown: List[Dict[str, Any]]


class SimulationResponse(BaseModel):
    """Simulation result."""
    baseline_mae: float
    siren_mae: float
    labels_filtered_pct: float
    rider_wait_baseline: float
    rider_wait_siren: float
    improvement_pct: float


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LIFESPAN — load models & data at startup
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models and data once at startup, clean up on shutdown."""
    print("[STARTUP] SIREN server starting -- loading models & data...")

    # Load datasets
    orders_path = os.path.join(DATA_DIR, "orders_100k.csv")
    merchants_path = os.path.join(DATA_DIR, "merchants_1k.csv")

    if os.path.exists(orders_path):
        state["orders"] = pd.read_csv(orders_path)
        print(f"   Loaded {len(state['orders']):,} orders")
    else:
        state["orders"] = pd.DataFrame()
        print("   [WARN] orders_100k.csv not found")

    if os.path.exists(merchants_path):
        state["merchants"] = pd.read_csv(merchants_path)
        print(f"   Loaded {len(state['merchants']):,} merchants")
    else:
        state["merchants"] = pd.DataFrame()
        print("   [WARN] merchants_1k.csv not found")

    # Load models
    model_files = {
        "baseline_xgb": "baseline_xgb.pkl",
        "siren_xgb": "siren_xgb.pkl",
        "siren_lgbm": "siren_lgbm.pkl",
        "layer3_kmeans": "layer3_kmeans.pkl",
        "layer3_scaler": "layer3_scaler.pkl",
    }
    state["models"] = {}
    for key, fname in model_files.items():
        path = os.path.join(MODELS_DIR, fname)
        if os.path.exists(path):
            state["models"][key] = joblib.load(path)
            print(f"   Loaded {fname}")
        else:
            print(f"   [WARN] {fname} not found")

    # Load metadata
    meta_path = os.path.join(MODELS_DIR, "training_metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            state["metadata"] = json.load(f)
    else:
        state["metadata"] = {}

    # Load results
    results_path = os.path.join(OUTPUTS_DIR, "model_results.csv")
    if os.path.exists(results_path):
        state["results"] = pd.read_csv(results_path)
    else:
        state["results"] = pd.DataFrame()

    fi_path = os.path.join(OUTPUTS_DIR, "feature_importance.csv")
    if os.path.exists(fi_path):
        state["feature_importance"] = pd.read_csv(fi_path)
    else:
        state["feature_importance"] = pd.DataFrame()

    preds_path = os.path.join(OUTPUTS_DIR, "test_predictions.csv")
    if os.path.exists(preds_path):
        state["test_predictions"] = pd.read_csv(preds_path)
    else:
        state["test_predictions"] = pd.DataFrame()

    state["model_loaded"] = "siren_xgb" in state["models"]
    state["trained_at"] = state["metadata"].get("trained_at", "unknown")

    print("[OK] SIREN server ready!")
    yield
    # Cleanup
    state.clear()
    print("[STOP] SIREN server stopped.")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# APP SETUP
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

app = FastAPI(
    title="SIREN — Signal-Informed Restaurant ETA Network",
    description="Zomato Kitchen Prep Time (KPT) Prediction Backend",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ENDPOINT IMPLEMENTATIONS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Return server health status including model loading state."""
    return HealthResponse(
        status="ok",
        model_loaded=state.get("model_loaded", False),
        dataset_rows=len(state.get("orders", [])),
        trained_at=state.get("trained_at", "unknown"),
    )


@app.get("/api/dashboard/summary", response_model=DashboardSummary)
async def dashboard_summary():
    """Return dashboard summary with KPT, rider wait, and city rush index."""
    df = state.get("orders", pd.DataFrame())
    preds = state.get("test_predictions", pd.DataFrame())
    results = state.get("results", pd.DataFrame())

    if preds.empty or results.empty:
        raise HTTPException(status_code=503, detail="Pipeline outputs not available. Run the pipeline first.")

    # Compute from test predictions (simulates "today")
    siren_row = results[results["model"] == "SIREN_XGBoost"]
    baseline_row = results[results["model"] == "Baseline_XGBoost"]

    avg_kpt = float(preds["true_kpt"].mean())
    rider_wait = float(siren_row["rider_wait"].iloc[0]) if len(siren_row) > 0 else 0
    p50_error = float(siren_row["p50_ae"].iloc[0]) if len(siren_row) > 0 else 0
    p90_error = float(siren_row["p90_ae"].iloc[0]) if len(siren_row) > 0 else 0

    # Approximate on-time rate: orders where |error| < 5 min
    if not preds.empty:
        errors = np.abs(preds["pred_siren_xgb"] - preds["true_kpt"])
        on_time = float((errors < 5).mean())
    else:
        on_time = 0.0

    # City rush index from the dataset
    city_rush = {}
    if not df.empty and "google_busyness_index" in df.columns:
        city_rush = df.groupby("city")["google_busyness_index"].mean().round(1).to_dict()

    return DashboardSummary(
        avg_kpt_today=round(avg_kpt, 2),
        rider_wait_avg=round(rider_wait, 2),
        p50_eta_error=round(p50_error, 2),
        p90_eta_error=round(p90_error, 2),
        total_orders_today=len(preds),
        on_time_rate=round(on_time, 4),
        city_rush_index=city_rush,
    )


@app.get("/api/orders/live", response_model=List[LiveOrder])
async def live_orders(
    limit: int = Query(50, ge=1, le=500),
    city: str = Query("all"),
    status: str = Query("all"),
):
    """Return recent orders with predictions. Simulates live order feed."""
    df = state.get("orders", pd.DataFrame())
    merchants = state.get("merchants", pd.DataFrame())
    preds = state.get("test_predictions", pd.DataFrame())

    if df.empty:
        raise HTTPException(status_code=503, detail="Dataset not loaded.")

    # Use test predictions as "live" orders
    if preds.empty:
        raise HTTPException(status_code=503, detail="Test predictions not available.")

    merged = preds.copy()

    # Filter by city
    if city != "all":
        merged = merged[merged["city"] == city]

    # Add merchant names
    name_map = dict(zip(merchants["merchant_id"], merchants["merchant_name"])) if not merchants.empty else {}

    # Get recent orders
    sample = merged.tail(limit)

    result = []
    for _, row in sample.iterrows():
        oid = row["order_id"]
        # Lookup from orders df
        order_row = df[df["order_id"] == oid]
        mid = order_row["merchant_id"].iloc[0] if len(order_row) > 0 else "unknown"
        cuisine = row.get("cuisine", "unknown")
        error = abs(row["pred_siren_xgb"] - row["true_kpt"])

        if error < 3:
            quality = "excellent"
        elif error < 6:
            quality = "good"
        elif error < 10:
            quality = "fair"
        else:
            quality = "poor"

        order_status = "delivered"
        if status != "all" and order_status != status:
            continue

        result.append(LiveOrder(
            order_id=oid,
            merchant_name=name_map.get(mid, mid),
            cuisine=cuisine,
            predicted_kpt=round(float(row["pred_siren_xgb"]), 2),
            actual_kpt=round(float(row["true_kpt"]), 2),
            status=order_status,
            rider_assigned=True,
            for_signal_quality=quality,
        ))

    return result[:limit]


@app.post("/api/predict", response_model=PredictResponse)
async def predict_kpt(request: PredictRequest):
    """Run real-time KPT prediction using the SIREN XGBoost model."""
    model = state.get("models", {}).get("siren_xgb")
    if model is None:
        raise HTTPException(status_code=503, detail="SIREN XGBoost model not loaded.")

    merchants = state.get("merchants", pd.DataFrame())
    orders = state.get("orders", pd.DataFrame())

    # Look up merchant
    m_row = merchants[merchants["merchant_id"] == request.merchant_id]
    if m_row.empty:
        bias_mean = 0.0
        bias_offset = 0.0
    else:
        bias_mean = float(m_row["bias_mean"].iloc[0])
        # Compute bias offset from orders if available
        m_orders = orders[orders["merchant_id"] == request.merchant_id]
        if "bias_offset" in m_orders.columns and len(m_orders) > 0:
            bias_offset = float(m_orders["bias_offset"].iloc[-1])
        else:
            bias_offset = bias_mean

    # Build feature vector
    is_lunch = 1 if 12 <= request.hour <= 14 else 0
    is_dinner = 1 if 19 <= request.hour <= 21 else 0
    is_weekend = 1 if request.day_of_week >= 5 else 0
    rush_mult = 1.0 + 1.5 * (request.google_busyness_index / 100.0)
    rain_impact = request.rain_flag * request.rain_severity * 3.5
    complexity_penalty = (request.order_complexity - 1) * 2.5
    cuisine_base = CUISINE_BASE_KPT.get(request.cuisine, 14)
    hour_sin = float(np.sin(2 * np.pi * request.hour / 24))
    hour_cos = float(np.cos(2 * np.pi * request.hour / 24))
    dow_sin = float(np.sin(2 * np.pi * request.day_of_week / 7))
    dow_cos = float(np.cos(2 * np.pi * request.day_of_week / 7))

    # Estimate concurrent orders
    concurrent = max(1, int(rush_mult * 3))

    # Relative rush (approximate)
    relative_rush = rush_mult / 1.5 if rush_mult > 0 else 1.0

    features = {
        "hour": request.hour,
        "day_of_week": request.day_of_week,
        "is_lunch_rush": is_lunch,
        "is_dinner_rush": is_dinner,
        "is_weekend": is_weekend,
        "order_complexity": request.order_complexity,
        "zomato_concurrent_orders": concurrent,
        "rush_multiplier": rush_mult,
        "relative_rush": relative_rush,
        "rain_flag": request.rain_flag,
        "rain_kpt_impact": rain_impact,
        "complexity_kpt_penalty": complexity_penalty,
        "cuisine_base_kpt": cuisine_base,
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "dow_sin": dow_sin,
        "dow_cos": dow_cos,
        "mean_marking_bias": bias_mean,
        "bias_offset": bias_offset,
        "google_busyness_index": request.google_busyness_index,
    }

    X = np.array([[features[f] for f in SIREN_FEATURES]])
    raw_pred = float(model.predict(X)[0])

    # Apply bias correction
    adjusted_pred = raw_pred - bias_offset
    adjusted_pred = max(2.0, adjusted_pred)

    # Confidence interval (approximate via ±model residual stats)
    siren_row = state.get("results", pd.DataFrame())
    if not siren_row.empty:
        siren_row = siren_row[siren_row["model"] == "SIREN_XGBoost"].iloc[0]
        rmse = float(siren_row["rmse"])
    else:
        rmse = 5.0
    ci_low = max(1.0, adjusted_pred - 1.28 * rmse)
    ci_high = adjusted_pred + 1.28 * rmse

    # FOR signal warning if merchant has high bias
    for_warning = abs(bias_mean) > 2.0

    # Feature importance
    fi_df = state.get("feature_importance", pd.DataFrame())
    contributions = []
    if not fi_df.empty:
        fi_xgb = fi_df[fi_df["model"] == "SIREN_XGBoost"].sort_values("importance", ascending=False).head(8)
        for _, row in fi_xgb.iterrows():
            fname = row["feature"]
            contributions.append(FeatureContribution(
                feature=fname,
                value=round(features.get(fname, 0), 4),
                importance=round(float(row["importance"]), 4),
            ))

    return PredictResponse(
        predicted_kpt_min=round(adjusted_pred, 2),
        confidence_interval=[round(ci_low, 2), round(ci_high, 2)],
        for_signal_warning=for_warning,
        recommended_dispatch_offset_min=round(max(1, adjusted_pred - 3), 1),
        feature_contributions=contributions,
    )


@app.get("/api/merchants", response_model=List[MerchantSummary])
async def list_merchants(
    city: str = Query("all"),
    cuisine: str = Query("all"),
    tier: str = Query("all"),
    behavior: str = Query("all"),
    search: str = Query(""),
    page: int = Query(1, ge=1),
    per_page: int = Query(50, ge=1, le=200),
):
    """Return paginated merchant list with filtering."""
    merchants = state.get("merchants", pd.DataFrame())
    if merchants.empty:
        raise HTTPException(status_code=503, detail="Merchants data not loaded.")

    filtered = merchants.copy()

    if city != "all":
        filtered = filtered[filtered["city"] == city]
    if cuisine != "all":
        filtered = filtered[filtered["cuisine"] == cuisine]
    if tier != "all":
        filtered = filtered[filtered["merchant_tier"] == tier]
    if behavior != "all":
        filtered = filtered[filtered["behavior_class"] == behavior]
    if search:
        mask = filtered["merchant_name"].str.contains(search, case=False, na=False)
        mask |= filtered["merchant_id"].str.contains(search, case=False, na=False)
        filtered = filtered[mask]

    # Paginate
    start = (page - 1) * per_page
    end = start + per_page
    page_data = filtered.iloc[start:end]

    return [
        MerchantSummary(
            merchant_id=str(row["merchant_id"]),
            merchant_name=str(row["merchant_name"]),
            merchant_tier=str(row["merchant_tier"]),
            cuisine=str(row["cuisine"]),
            city=str(row["city"]),
            behavior_class=str(row["behavior_class"]),
            avg_daily_orders=int(row["avg_daily_orders"]),
            has_iot_beacon=bool(row["has_iot_beacon"]),
            bias_mean=round(float(row["bias_mean"]), 3),
        )
        for _, row in page_data.iterrows()
    ]


@app.get("/api/merchants/{merchant_id}", response_model=MerchantDetail)
async def get_merchant(merchant_id: str):
    """Return detailed merchant info including order history and bias trend."""
    merchants = state.get("merchants", pd.DataFrame())
    orders = state.get("orders", pd.DataFrame())

    m_row = merchants[merchants["merchant_id"] == merchant_id]
    if m_row.empty:
        raise HTTPException(status_code=404, detail=f"Merchant {merchant_id} not found.")

    m = m_row.iloc[0]

    # Get last 100 orders
    m_orders = orders[orders["merchant_id"] == merchant_id].tail(100)

    recent = []
    for _, o in m_orders.iterrows():
        recent.append({
            "order_id": str(o.get("order_id", "")),
            "order_timestamp": str(o.get("order_timestamp", "")),
            "measured_kpt_min": round(float(o.get("measured_kpt_min", 0)), 2),
            "true_kpt_min": round(float(o.get("true_kpt_min", 0)), 2),
            "cuisine": str(o.get("cuisine", "")),
            "order_complexity": int(o.get("order_complexity", 1)),
        })

    # KPT distribution
    kpt_vals = m_orders["true_kpt_min"] if not m_orders.empty else pd.Series([0])
    kpt_dist = {
        "mean": round(float(kpt_vals.mean()), 2),
        "median": round(float(kpt_vals.median()), 2),
        "std": round(float(kpt_vals.std()), 2),
        "min": round(float(kpt_vals.min()), 2),
        "max": round(float(kpt_vals.max()), 2),
        "p25": round(float(kpt_vals.quantile(0.25)), 2),
        "p75": round(float(kpt_vals.quantile(0.75)), 2),
    }

    # Bias trend (rolling bias from last orders)
    bias_trend = []
    if "bias_offset" in m_orders.columns and not m_orders.empty:
        bias_trend = m_orders["bias_offset"].dropna().tolist()[-20:]
        bias_trend = [round(float(b), 3) for b in bias_trend]

    return MerchantDetail(
        merchant_id=str(m["merchant_id"]),
        merchant_name=str(m["merchant_name"]),
        merchant_tier=str(m["merchant_tier"]),
        cuisine=str(m["cuisine"]),
        city=str(m["city"]),
        behavior_class=str(m["behavior_class"]),
        avg_daily_orders=int(m["avg_daily_orders"]),
        has_iot_beacon=bool(m["has_iot_beacon"]),
        bias_mean=round(float(m["bias_mean"]), 3),
        recent_orders=recent,
        kpt_distribution=kpt_dist,
        bias_trend=bias_trend,
    )


@app.get("/api/signal-quality", response_model=SignalQualityResponse)
async def signal_quality(
    date_from: str = Query("", description="Start date (YYYY-MM-DD)"),
    date_to: str = Query("", description="End date (YYYY-MM-DD)"),
):
    """Return Layer 1/2/3 signal quality stats."""
    df = state.get("orders", pd.DataFrame())
    merchants = state.get("merchants", pd.DataFrame())

    if df.empty:
        raise HTTPException(status_code=503, detail="Dataset not loaded.")

    # Optional date filtering
    if date_from:
        df = df[pd.to_datetime(df["order_timestamp"]) >= date_from]
    if date_to:
        df = df[pd.to_datetime(df["order_timestamp"]) <= date_to]

    # Layer 1
    layer1 = {}
    if "flag_rider_delta" in df.columns:
        predicted = df["flag_rider_delta"].astype(bool) | df["flag_percentile"].astype(bool)
        actual = df["is_rider_triggered_FOR"].astype(bool)
        tp = int((predicted & actual).sum())
        fp = int((predicted & ~actual).sum())
        fn = int((~predicted & actual).sum())
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        bias_before = float((df["measured_kpt_min"] - df["true_kpt_min"]).mean())
        clean_mask = df["is_clean_label"] == True
        bias_after = float((df.loc[clean_mask, "clean_kpt_min"] - df.loc[clean_mask, "true_kpt_min"]).mean()) if "clean_kpt_min" in df.columns and clean_mask.sum() > 0 else 0
        layer1 = {
            "filter_rate": round(float(predicted.mean()), 4),
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f1, 4),
            "bias_before": round(bias_before, 4),
            "bias_after": round(bias_after, 4),
        }
    else:
        layer1 = {"status": "Layer 1 not yet applied"}

    # Layer 2
    layer2 = {}
    if "rush_multiplier" in df.columns:
        layer2 = {
            "rush_mean": round(float(df["rush_multiplier"].mean()), 4),
            "rain_count": int(df["rain_flag"].sum()),
            "weather_impact_mean": round(float(df["rain_kpt_impact"].mean()), 4) if "rain_kpt_impact" in df.columns else 0,
        }
    else:
        layer2 = {"status": "Layer 2 not yet applied"}

    # Layer 3
    layer3 = {}
    if "behavior_class_predicted" in df.columns or "behavior_class" in merchants.columns:
        beh_col = "behavior_class"
        counts = merchants[beh_col].value_counts().to_dict() if beh_col in merchants.columns else {}
        mean_bias = round(float(df["bias_offset"].mean()), 4) if "bias_offset" in df.columns else 0
        layer3 = {
            "behavior_class_counts": counts,
            "mean_bias_offset": mean_bias,
        }
    else:
        layer3 = {"status": "Layer 3 not yet applied"}

    return SignalQualityResponse(layer1=layer1, layer2=layer2, layer3=layer3)


@app.get("/api/model-results", response_model=ModelResultsResponse)
async def model_results():
    """Return model comparison results, feature importance, tier breakdown."""
    results = state.get("results", pd.DataFrame())
    fi = state.get("feature_importance", pd.DataFrame())
    metadata = state.get("metadata", {})

    if results.empty:
        raise HTTPException(status_code=503, detail="Model results not available. Run the pipeline first.")

    models_list = results.to_dict("records")

    # Feature importance (SIREN XGBoost top features)
    fi_list = []
    if not fi.empty:
        fi_xgb = fi[fi["model"] == "SIREN_XGBoost"].sort_values("importance", ascending=False)
        fi_list = fi_xgb[["feature", "importance", "layer"]].to_dict("records")

    # Tier breakdown
    tier_breakdown = []
    baseline_tier = metadata.get("baseline_tier_mae", {})
    siren_tier = metadata.get("siren_xgb_tier_mae", {})
    for tier_name in set(list(baseline_tier.keys()) + list(siren_tier.keys())):
        tier_breakdown.append({
            "tier": tier_name,
            "baseline_mae": baseline_tier.get(tier_name, 0),
            "siren_mae": siren_tier.get(tier_name, 0),
        })

    return ModelResultsResponse(
        models=models_list,
        feature_importance=fi_list,
        tier_breakdown=tier_breakdown,
    )


@app.get("/api/simulation", response_model=SimulationResponse)
async def simulation(
    contam: int = Query(40, ge=0, le=100, description="Contamination %"),
    theta: int = Query(90, ge=30, le=300, description="Rider-delta threshold (seconds)"),
    dispatch_offset: float = Query(3.0, ge=0, le=15, description="Dispatch offset (min)"),
):
    """Run a what-if simulation with different contamination and theta settings."""
    results = state.get("results", pd.DataFrame())
    preds = state.get("test_predictions", pd.DataFrame())

    if results.empty or preds.empty:
        raise HTTPException(status_code=503, detail="Pipeline outputs not available.")

    baseline_row = results[results["model"] == "Baseline_XGBoost"].iloc[0]
    siren_row = results[results["model"] == "SIREN_XGBoost"].iloc[0]

    # Scale metrics based on contamination level (simulation approximation)
    contam_factor = contam / 38.0  # 38% is the default contamination rate
    noise_factor = 1.0 + 0.15 * (contam_factor - 1.0)

    # Simulate different theta by scaling filter rate
    theta_factor = 1.0 + 0.005 * (theta - 90)

    baseline_mae = float(baseline_row["mae"]) * noise_factor
    siren_mae = float(siren_row["mae"]) * max(0.8, 1.0 - 0.002 * (theta - 60))

    improvement = ((baseline_mae - siren_mae) / baseline_mae) * 100

    # Rider wait
    baseline_wait = float(baseline_row["rider_wait"]) * noise_factor
    siren_wait = float(siren_row["rider_wait"]) * max(0.7, 1.0 - 0.001 * (theta - 60))

    # Labels filtered
    filtered_pct = min(60, max(10, contam * theta / 100))

    return SimulationResponse(
        baseline_mae=round(baseline_mae, 3),
        siren_mae=round(siren_mae, 3),
        labels_filtered_pct=round(filtered_pct, 1),
        rider_wait_baseline=round(baseline_wait, 3),
        rider_wait_siren=round(siren_wait, 3),
        improvement_pct=round(improvement, 1),
    )


@app.get("/api/merchants/{merchant_id}/export")
async def export_merchant_orders(merchant_id: str):
    """Export merchant's order history as a CSV download."""
    orders = state.get("orders", pd.DataFrame())
    if orders.empty:
        raise HTTPException(status_code=503, detail="Dataset not loaded.")

    m_orders = orders[orders["merchant_id"] == merchant_id]
    if m_orders.empty:
        raise HTTPException(status_code=404, detail=f"No orders for merchant {merchant_id}.")

    csv_buffer = io.StringIO()
    m_orders.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)

    return StreamingResponse(
        iter([csv_buffer.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={merchant_id}_orders.csv"},
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
