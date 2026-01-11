# Football Match Prediction ML Pipeline

A machine learning pipeline for predicting football (soccer) match outcomes using historical data from Sofascore.

## 🎯 Project Overview

This project predicts match results (Home Win / Draw / Away Win) using:
- **1,087 engineered features** (rolling stats, H2H, odds)
- **XGBoost classifier** with tuned hyperparameters
- **Time-leak-free** feature engineering

### Key Results (80/20 Temporal Split)

| Metric | Value |
|--------|-------|
| Test Accuracy | **51.3%** |
| Draw Accuracy | **29.9%** |
| Best ROI Zone | Odds 1.5-2.0: **+6.2%** |
| High Confidence (>60%) | **+4.9% to +12.6% ROI** |

## 📁 Project Structure

```
├── src/                          # Source code
│   ├── process_data.py           # Basic data processing (59 features)
│   ├── process_data_extended.py  # Extended processing (1,087 features)
│   ├── train_model.py            # Initial XGBoost training
│   ├── train_model_v2.py         # Improved model (reduced overfitting)
│   └── sofascore_colab.ipynb     # Scraper notebook (run in Colab)
│
├── data/
│   ├── raw/                      # Raw scraped CSV files
│   ├── processed/                # Feature-engineered datasets
│   └── predictions/              # Model predictions
│
├── models/                       # Saved models and scalers
│   ├── xgboost_model_v2.json     # Best model
│   └── scaler_v2.pkl             # Feature scaler
│
├── results/                      # Analysis outputs
│   ├── feature_importance.png
│   └── model_results_v2.csv
│
├── analysis_odds.py              # Odds range analysis script
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Process Data

```bash
# Basic processing (59 features)
python src/process_data.py --input data/raw/matches.csv --output data/processed/features.csv

# Extended processing (1,087 features)
python src/process_data_extended.py --input data/raw/matches.csv --output data/processed/features_extended.csv
```

### 3. Train Model

```bash
# Train improved model (v2)
python src/train_model_v2.py
```

### 4. Analyze Performance

```bash
python analysis_odds.py
```

## 📊 Model Performance

### Accuracy by Odds Range

| Odds Range | Accuracy | ROI |
|------------|----------|-----|
| 1.0 - 1.5 | 70.5% | -5.8% |
| **1.5 - 2.0** | **61.4%** | **+6.2%** |
| 2.0 - 3.0 | 40.9% | -1.5% |
| 3.0 - 5.0 | 32.7% | +10.5% |

### Accuracy by Model Confidence

| Confidence | Accuracy | ROI |
|------------|----------|-----|
| > 70% | 87.5% | +12.6% |
| 60-70% | 72.9% | +4.9% |
| < 60% | ~42% | Negative |

## 🔧 Features

### Rolling Statistics (Windows: 5, 10 matches)
- Goals for/against
- Expected goals (xG)
- Shots, passes, tackles
- Ball possession
- First half / Second half stats

### Head-to-Head (H2H)
- Total encounters
- Win percentages
- Draw rate

### Odds Features
- 1X2 odds (Home/Draw/Away)
- Implied probabilities
- Log odds
- Overround (margin)

## ⚠️ Important Notes

1. **Time Leakage Prevention**: All rolling features use `shift(1)` to exclude current match
2. **Temporal Split**: Always split chronologically, never randomly
3. **Draw Prediction**: Hardest class - consider two-stage classification
4. **Value Betting**: Filter for confidence >60% and odds 1.5-2.0

## 📦 Requirements

- Python 3.9+
- pandas
- numpy
- scikit-learn
- xgboost
- matplotlib

## 📝 License

MIT License

## 🙏 Acknowledgments

- Data source: Sofascore (via scraping)
- Model: XGBoost
