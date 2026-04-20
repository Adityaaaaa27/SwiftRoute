# SIREN — Signal-Informed Restaurant ETA Network

> **Zomato Kitchen Prep Time (KPT) Prediction System**
> A 3-layer ML pipeline that de-noises biased FOR (Food Order Ready) signals, enriches predictions with external data, and corrects per-merchant marking biases.

---

## 🏗 Architecture

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  LAYER 1         │     │  LAYER 2         │     │  LAYER 3         │
│  FOR De-noising  │────▶│  Signal          │────▶│  Bias Correction │
│                  │     │  Enrichment      │     │                  │
│  • Rider-delta   │     │  • Rush mult     │     │  • Behavior      │
│    filter        │     │  • Weather       │     │    classifier    │
│  • Percentile    │     │  • Time cyclical │     │  • KMeans        │
│    check         │     │  • Cuisine prior │     │    clustering    │
│  • IoT override  │     │  • Complexity    │     │  • Rolling bias  │
└──────────────────┘     └──────────────────┘     └──────────────────┘
         │                                                 │
         ▼                                                 ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    MODEL TRAINING (3 Models)                        │
│  A: Baseline XGBoost (biased labels, no enrichment)                 │
│  B: SIREN XGBoost (clean labels + enrichment + bias correction)     │
│  C: SIREN LightGBM (same pipeline, different algorithm)             │
└──────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    FastAPI Backend Server                            │
│  10 API endpoints · Real-time prediction · Dashboard metrics        │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
siren_backend/
├── data/
│   ├── orders_100k.csv          # 100K synthetic orders
│   └── merchants_1k.csv         # 1K merchant profiles
├── models/
│   ├── baseline_xgb.pkl         # Baseline XGBoost model
│   ├── siren_xgb.pkl            # SIREN XGBoost model
│   ├── siren_lgbm.pkl           # SIREN LightGBM model
│   ├── layer3_kmeans.pkl        # Merchant clustering model
│   ├── layer3_scaler.pkl        # Feature scaler for clustering
│   └── training_metadata.json   # Training configuration & metrics
├── outputs/
│   ├── model_results.csv        # All model metrics
│   ├── feature_importance.csv   # Feature importance by model
│   ├── test_predictions.csv     # Test set predictions
│   ├── accuracy_report.txt      # Full accuracy report
│   └── *.png                    # 6 visualization charts
├── data_generator.py            # Synthetic dataset generator
├── layer1_denoising.py          # Layer 1: FOR de-noising
├── layer2_enrichment.py         # Layer 2: Signal enrichment
├── layer3_bias_correction.py    # Layer 3: Merchant bias correction
├── model_training.py            # 3-model training pipeline
├── visualizations.py            # 6 publication-quality charts
├── accuracy_report.py           # Comprehensive accuracy report
├── main.py                      # FastAPI backend server
├── run_pipeline.py              # Full pipeline runner
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
cd siren_backend
pip install -r requirements.txt
```

### 2. Run the full pipeline

```bash
python run_pipeline.py
```

This will:
1. Generate 100K synthetic orders + 1K merchants
2. Apply Layer 1 de-noising (FOR signal filtering)
3. Apply Layer 2 enrichment (rush, weather, time features)
4. Apply Layer 3 bias correction (per-merchant bias offsets)
5. Train 3 ML models (Baseline XGB, SIREN XGB, SIREN LGBM)
6. Generate 6 visualization charts
7. Produce the accuracy report

### 3. Start the API server

```bash
python run_pipeline.py --serve
# or directly:
python main.py
```

Server runs at: `http://localhost:8000`  
API docs at: `http://localhost:8000/docs`

---

## 📊 Expected Accuracy Numbers

| Metric | Baseline XGBoost | SIREN XGBoost | Improvement |
|--------|-----------------|---------------|-------------|
| **MAE** (min) | 6.5 – 8.5 | 4.5 – 6.0 | 25 – 40% |
| **RMSE** (min) | 8.5 – 11.0 | 5.5 – 7.5 | 28 – 38% |
| **P50 AE** (min) | 5.0 – 7.0 | 3.5 – 5.0 | 20 – 35% |
| **P90 AE** (min) | 14 – 18 | 9 – 13 | 30 – 42% |
| **Rider Wait** (min) | 5.5 – 8.0 | 3.0 – 4.5 | 38 – 50% |
| **Layer 1 F1** | — | > 0.88 | — |

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | Health check & model status |
| `GET` | `/api/dashboard/summary` | Dashboard metrics summary |
| `GET` | `/api/orders/live` | Recent orders with predictions |
| `POST` | `/api/predict` | Real-time KPT prediction |
| `GET` | `/api/merchants` | Paginated merchant list |
| `GET` | `/api/merchants/{id}` | Merchant detail + history |
| `GET` | `/api/signal-quality` | Layer 1/2/3 quality stats |
| `GET` | `/api/model-results` | Model comparison results |
| `GET` | `/api/simulation` | What-if simulation |
| `GET` | `/api/merchants/{id}/export` | CSV download of orders |

---

## 🧪 Dataset Details

- **100,000 orders** across **1,000 merchants**, **10 cities**, **10 cuisines**
- **38% contamination rate** (rider-triggered FORs)
- **25% early markers**, 55% accurate, 20% late markers
- **IoT beacon coverage (realistic, India 2024)**:
  - Large chain: **40%** (only flagship outlets with shelf sensors)
  - Mid-independent: **3%** (rare early adopters)
  - Small kiosk: **0%** (not economically viable)
  - **Effective coverage: ~4–5% of merchants, ~7–8% of orders**
- Date range: Jan 2023 – Dec 2024

---

## ⚙️ Requirements

- Python 3.10+
- XGBoost ≥ 1.7
- LightGBM ≥ 4.0
- FastAPI ≥ 0.100
- scikit-learn ≥ 1.3
- See `requirements.txt` for full list

---

## 📝 License

Research project — Zomato SIREN System Specialization.
