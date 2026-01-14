#!/usr/bin/env python3
"""
Football Match Prediction - IMPROVED Model Training Pipeline
=============================================================

This improved version addresses the overfitting issues identified in the evaluation:
1. Stronger regularization (reduced tree depth, slower learning)
2. Feature selection (top 200 features instead of 1,087)
3. Class weighting (boost draw prediction)
4. Conservative value betting (15% EV threshold)
5. Better model calibration

Changes from v1:
- max_depth: 6 → 4
- learning_rate: 0.05 → 0.01  
- reg_alpha: 0.1 → 1.0
- reg_lambda: 1.0 → 10.0
- subsample: 0.8 → 0.5
- colsample_bytree: 0.8 → 0.3
- Feature reduction: 1,087 → 200
- Class weights: Added
- EV threshold: 5% → 15%

Author: Claude (Anthropic)
Version: 2.0.0 (Improved)

Usage:
    python train_model_v2.py
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, log_loss, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION - IMPROVED PARAMETERS
# =============================================================================

def find_latest_file(directory, pattern='*.csv'):
    """Find the most recent CSV file in a directory."""
    import glob
    files = glob.glob(os.path.join(directory, pattern))
    if not files:
        return None
    return sorted(files)[-1]

# File paths - auto-find latest files
PROCESSED_DATA = 'data/processed/processed_features_extended.csv'
_future_file = find_latest_file('data/predictions', 'sofascore_future_*.csv')
FUTURE_DATA = _future_file if _future_file else 'data/predictions/sofascore_future.csv'

# Temporal split dates
TRAIN_CUTOFF = '2024-08-01'
VALID_CUTOFF = '2025-01-01'

# ============ KEY IMPROVEMENT: STRONGER REGULARIZATION ============
# These parameters reduce overfitting significantly
XGB_PARAMS = {
    'max_depth': 4,              # Was 6 → Shallower trees = less overfitting
    'learning_rate': 0.01,       # Was 0.05 → Slower learning = harder to fit noise
    'objective': 'multi:softprob',
    'num_class': 3,
    'eval_metric': 'mlogloss',
    'subsample': 0.5,            # Was 0.8 → Use only 50% of rows per tree
    'colsample_bytree': 0.3,     # Was 0.8 → Use only 30% of columns per tree
    'min_child_weight': 10,      # Was 3 → Require more samples per leaf
    'reg_alpha': 1.0,            # Was 0.1 → 10x L1 regularization
    'reg_lambda': 10.0,          # Was 1.0 → 10x L2 regularization
    'random_state': 42,
    'verbosity': 0
}

NUM_BOOST_ROUNDS = 1000  # More rounds since learning rate is lower
EARLY_STOPPING_ROUNDS = 100

# ============ KEY IMPROVEMENT: FEATURE SELECTION ============
# Keep only top N features to reduce curse of dimensionality
TOP_N_FEATURES = 200  # Was 1,087 → Reduces overfitting

# ============ KEY IMPROVEMENT: CONSERVATIVE VALUE BETTING ============
VALUE_BET_THRESHOLD = 0.15  # Was 0.05 → Only bet on high-confidence edges

# Label encoding
LABEL_ENCODE = {-1: 0, 0: 1, 1: 2}
LABEL_DECODE = {0: -1, 1: 0, 2: 1}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def print_separator(title=""):
    if title:
        print(f"\n{'='*70}")
        print(f" {title}")
        print(f"{'='*70}")
    else:
        print("=" * 70)


def encode_labels(y):
    """Encode labels from -1/0/1 to 0/1/2 for XGBoost."""
    return np.array([LABEL_ENCODE[label] for label in y])


def decode_labels(y_encoded):
    """Decode labels from 0/1/2 back to -1/0/1."""
    return np.array([LABEL_DECODE[label] for label in y_encoded])


def map_predictions_to_classes(pred_probs):
    """Map XGBoost prediction probabilities to original class labels."""
    pred_classes = np.argmax(pred_probs, axis=1)
    return decode_labels(pred_classes)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_processed_data():
    """Load the processed features dataset."""
    print_separator("STEP 1: LOADING DATA")
    
    df = pd.read_csv(PROCESSED_DATA)
    print(f"Loaded: {df.shape[0]:,} matches, {df.shape[1]:,} columns")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    
    # Show target distribution
    print(f"\nTarget distribution:")
    target_counts = df['target'].value_counts().sort_index()
    for target, count in target_counts.items():
        label = {-1: 'Away Win', 0: 'Draw', 1: 'Home Win'}[target]
        print(f"  {target:2d} ({label:9s}): {count:,} ({count/len(df)*100:.1f}%)")
    
    # Identify feature columns
    exclude_cols = ['match_id', 'date', 'timestamp', 'home_team_id', 'away_team_id',
                    'home_team', 'away_team', 'tournament_id', 'league', 
                    'target', 'home_score', 'away_score']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    print(f"\nTotal features available: {len(feature_cols)}")
    
    # Handle nulls
    null_count = df[feature_cols].isnull().sum().sum()
    if null_count > 0:
        print(f"Filling {null_count:,} null values with 0...")
        df[feature_cols] = df[feature_cols].fillna(0)
    
    return df, feature_cols


# =============================================================================
# FEATURE SELECTION - KEY IMPROVEMENT
# =============================================================================

def select_top_features(X_train, y_train, feature_cols, top_n=200):
    """
    Select top N features using a quick XGBoost model.
    
    WHY: Reduces curse of dimensionality from 1,087 → 200 features.
    With 235 training samples, having 1,087 features causes severe overfitting.
    Rule of thumb: 1 feature per 10-20 samples = 12-24 features ideal.
    We use 200 as a compromise for predictive power.
    """
    print_separator("STEP 3: FEATURE SELECTION")
    print(f"Original features: {len(feature_cols)}")
    print(f"Target: {top_n} features")
    
    # Train a quick model to get feature importance
    y_encoded = encode_labels(y_train)
    
    quick_params = {
        'max_depth': 4,
        'learning_rate': 0.1,
        'objective': 'multi:softprob',
        'num_class': 3,
        'eval_metric': 'mlogloss',
        'subsample': 0.8,
        'colsample_bytree': 0.5,
        'random_state': 42,
        'verbosity': 0
    }
    
    dtrain = xgb.DMatrix(X_train, label=y_encoded, feature_names=feature_cols)
    quick_model = xgb.train(quick_params, dtrain, num_boost_round=50, verbose_eval=False)
    
    # Get feature importance
    importance = quick_model.get_score(importance_type='weight')
    
    # Map f0, f1, f2... to actual names
    importance_named = {}
    for feat_key, score in importance.items():
        try:
            feat_idx = int(feat_key[1:])
            feat_name = feature_cols[feat_idx]
            importance_named[feat_name] = score
        except (ValueError, IndexError):
            continue
    
    # Select top N
    sorted_features = sorted(importance_named.items(), key=lambda x: x[1], reverse=True)
    selected_features = [f[0] for f in sorted_features[:top_n]]
    
    # If we got fewer than top_n, add remaining features
    if len(selected_features) < top_n:
        remaining = [f for f in feature_cols if f not in selected_features]
        selected_features.extend(remaining[:top_n - len(selected_features)])
    
    print(f"Selected: {len(selected_features)} features")
    
    # Show top 10 features
    print(f"\nTop 10 most important features:")
    for i, (feat, score) in enumerate(sorted_features[:10], 1):
        print(f"  {i:2d}. {feat[:50]:50s} : {score:.0f}")
    
    # Categorize selected features
    rolling_count = sum(1 for f in selected_features if 'rolling' in f)
    odds_count = sum(1 for f in selected_features if 'odds_' in f or 'implied_' in f)
    h2h_count = sum(1 for f in selected_features if 'h2h' in f)
    streak_count = sum(1 for f in selected_features if 'streak_' in f)
    
    print(f"\nSelected features by category:")
    print(f"  Rolling: {rolling_count}")
    print(f"  Odds:    {odds_count}")
    print(f"  H2H:     {h2h_count}")
    print(f"  Streak:  {streak_count}")
    
    return selected_features


# =============================================================================
# TEMPORAL SPLIT
# =============================================================================

def create_temporal_split(df, feature_cols):
    """Create train/validation/test split based on dates."""
    print_separator("STEP 2: TEMPORAL SPLIT")
    
    train_cutoff = pd.to_datetime(TRAIN_CUTOFF)
    valid_cutoff = pd.to_datetime(VALID_CUTOFF)
    
    df['date'] = pd.to_datetime(df['date'])
    
    train_mask = df['date'] < train_cutoff
    valid_mask = (df['date'] >= train_cutoff) & (df['date'] < valid_cutoff)
    test_mask = df['date'] >= valid_cutoff
    
    train_data = df[train_mask].copy()
    valid_data = df[valid_mask].copy()
    test_data = df[test_mask].copy()
    
    print(f"Train: {len(train_data):,} matches ({train_data['date'].min().date()} to {train_data['date'].max().date()})")
    print(f"Valid: {len(valid_data):,} matches ({valid_data['date'].min().date()} to {valid_data['date'].max().date()})")
    print(f"Test:  {len(test_data):,} matches ({test_data['date'].min().date()} to {test_data['date'].max().date()})")
    
    X_train = train_data[feature_cols].values
    y_train = train_data['target'].values
    X_valid = valid_data[feature_cols].values
    y_valid = valid_data['target'].values
    X_test = test_data[feature_cols].values
    y_test = test_data['target'].values
    
    # Baseline accuracy
    most_frequent = np.bincount(y_train.astype(int) + 1).argmax() - 1
    baseline_acc = (y_train == most_frequent).mean()
    print(f"\nBaseline accuracy (always predict {most_frequent}): {baseline_acc:.2%}")
    
    return (X_train, y_train, train_data), (X_valid, y_valid, valid_data), (X_test, y_test, test_data)


# =============================================================================
# MODEL TRAINING - WITH CLASS WEIGHTS
# =============================================================================

def train_xgboost_improved(X_train, y_train, X_valid, y_valid, feature_cols):
    """
    Train XGBoost model with improved regularization and class weights.
    
    KEY IMPROVEMENTS:
    1. Stronger regularization parameters
    2. Class weights to handle imbalance
    3. More rounds with slower learning rate
    """
    print_separator("STEP 4: MODEL TRAINING (IMPROVED)")
    
    # ============ KEY IMPROVEMENT: CLASS WEIGHTS ============
    # Calculate sample weights to boost underrepresented classes (especially draws)
    y_train_encoded = encode_labels(y_train)
    y_valid_encoded = encode_labels(y_valid)
    
    sample_weights = compute_sample_weight('balanced', y_train_encoded)
    
    # Show class distribution and weights
    print("Class distribution and weights:")
    for label in [-1, 0, 1]:
        count = (y_train == label).sum()
        encoded = LABEL_ENCODE[label]
        avg_weight = sample_weights[y_train == label].mean()
        name = {-1: 'Away', 0: 'Draw', 1: 'Home'}[label]
        print(f"  {name:5s}: {count:3d} samples, avg weight: {avg_weight:.2f}")
    
    print("\nXGBoost Parameters (IMPROVED):")
    for k, v in XGB_PARAMS.items():
        print(f"  {k}: {v}")
    
    # Create DMatrix with sample weights
    dtrain = xgb.DMatrix(X_train, label=y_train_encoded, 
                         weight=sample_weights, feature_names=feature_cols)
    dvalid = xgb.DMatrix(X_valid, label=y_valid_encoded, feature_names=feature_cols)
    
    # Train with early stopping
    print(f"\nTraining (max {NUM_BOOST_ROUNDS} rounds, early stopping after {EARLY_STOPPING_ROUNDS})...")
    
    evals = [(dtrain, 'train'), (dvalid, 'valid')]
    evals_result = {}
    
    model = xgb.train(
        XGB_PARAMS,
        dtrain,
        num_boost_round=NUM_BOOST_ROUNDS,
        evals=evals,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        evals_result=evals_result,
        verbose_eval=100
    )
    
    print(f"\nModel trained with {model.best_iteration} rounds")
    print(f"Best validation logloss: {model.best_score:.4f}")
    
    return model, evals_result


# =============================================================================
# MODEL EVALUATION
# =============================================================================

def evaluate_model(model, X_train, y_train, X_valid, y_valid, X_test, y_test, feature_cols):
    """Comprehensive model evaluation."""
    print_separator("STEP 5: MODEL EVALUATION")
    
    dtrain = xgb.DMatrix(X_train, feature_names=feature_cols)
    dvalid = xgb.DMatrix(X_valid, feature_names=feature_cols)
    dtest = xgb.DMatrix(X_test, feature_names=feature_cols)
    
    train_probs = model.predict(dtrain)
    valid_probs = model.predict(dvalid)
    test_probs = model.predict(dtest)
    
    train_pred = map_predictions_to_classes(train_probs)
    valid_pred = map_predictions_to_classes(valid_probs)
    test_pred = map_predictions_to_classes(test_probs)
    
    results = {}
    for name, y_true, y_pred, y_probs in [
        ('Train', y_train, train_pred, train_probs),
        ('Valid', y_valid, valid_pred, valid_probs),
        ('Test', y_test, test_pred, test_probs)
    ]:
        acc = accuracy_score(y_true, y_pred)
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        results[name] = {'accuracy': acc, 'f1_macro': f1_macro}
        
        print(f"\n{name} Set:")
        print(f"  Accuracy:    {acc:.4f}")
        print(f"  F1 (macro):  {f1_macro:.4f}")
    
    # Overfitting check - KEY METRIC
    overfit_gap = results['Train']['accuracy'] - results['Valid']['accuracy']
    print(f"\n*** OVERFITTING CHECK ***")
    print(f"Overfitting gap (Train Acc - Valid Acc): {overfit_gap:.4f}")
    if overfit_gap > 0.10:
        print("  WARNING: Still overfitting!")
    elif overfit_gap > 0.05:
        print("  CAUTION: Some overfitting remains.")
    else:
        print("  GOOD: Model is well-regularized!")
    
    # Confusion matrix
    print(f"\nTest Set Confusion Matrix:")
    cm = confusion_matrix(y_test, test_pred, labels=[-1, 0, 1])
    print("              Predicted")
    print("           -1    0    1")
    print("Actual -1: {:4d} {:4d} {:4d}".format(cm[0, 0], cm[0, 1], cm[0, 2]))
    print("       0:  {:4d} {:4d} {:4d}".format(cm[1, 0], cm[1, 1], cm[1, 2]))
    print("       1:  {:4d} {:4d} {:4d}".format(cm[2, 0], cm[2, 1], cm[2, 2]))
    
    # Per-class accuracy - KEY METRIC FOR DRAWS
    print(f"\nPer-class accuracy (Test):")
    label_names = {-1: 'Away', 0: 'Draw', 1: 'Home'}
    for i, label in enumerate([-1, 0, 1]):
        class_total = cm[i].sum()
        class_correct = cm[i, i]
        class_acc = class_correct / class_total if class_total > 0 else 0
        print(f"  {label:2d} ({label_names[label]:5s}): {class_acc:.2%} ({class_correct}/{class_total})")
    
    return results, test_probs, test_pred


# =============================================================================
# FUTURE PREDICTIONS
# =============================================================================

def predict_future_matches(model, scaler, feature_cols, df_historical):
    """Predict outcomes for future matches with FULL feature engineering."""
    print_separator("STEP 7: PREDICTING FUTURE MATCHES")
    
    if not os.path.exists(FUTURE_DATA):
        print(f"Future data file not found: {FUTURE_DATA}")
        return None, None
    
    df_future = pd.read_csv(FUTURE_DATA)
    print(f"Loaded {len(df_future)} future matches")
    
    # Load processed historical data
    df_hist = pd.read_csv(PROCESSED_DATA)
    df_hist['date'] = pd.to_datetime(df_hist['date'])
    
    # =========================================================================
    # STEP 1: Calculate derived odds features
    # =========================================================================
    print("\n  Calculating derived odds features...")
    
    for col in ['odds_1x2_home', 'odds_1x2_draw', 'odds_1x2_away']:
        if col not in df_future.columns:
            df_future[col] = 2.0  # Default if missing
    
    # Implied probabilities
    df_future['implied_prob_home'] = 1.0 / df_future['odds_1x2_home']
    df_future['implied_prob_draw'] = 1.0 / df_future['odds_1x2_draw']
    df_future['implied_prob_away'] = 1.0 / df_future['odds_1x2_away']
    
    # Log odds
    df_future['log_odds_home'] = np.log(df_future['odds_1x2_home'].clip(lower=1.01))
    df_future['log_odds_draw'] = np.log(df_future['odds_1x2_draw'].clip(lower=1.01))
    df_future['log_odds_away'] = np.log(df_future['odds_1x2_away'].clip(lower=1.01))
    
    # Overround
    df_future['overround'] = df_future['implied_prob_home'] + df_future['implied_prob_draw'] + df_future['implied_prob_away']
    
    # BTTS odds (use defaults if not present)
    if 'odds_btts_yes' not in df_future.columns:
        df_future['odds_btts_yes'] = 1.85
    if 'odds_btts_no' not in df_future.columns:
        df_future['odds_btts_no'] = 1.95
    
    # =========================================================================
    # STEP 2: Build feature matrix with proper imputation
    # =========================================================================
    print("  Building feature matrix with team-specific imputation...")
    
    X_future = np.zeros((len(df_future), len(feature_cols)))
    feature_sources = {'direct': 0, 'team_hist': 0, 'global': 0}
    
    for idx, future_row in df_future.iterrows():
        home_team_id = future_row.get('home_team_id')
        away_team_id = future_row.get('away_team_id')
        home_team = future_row.get('home_team')
        away_team = future_row.get('away_team')
        
        # Get historical matches for EACH team (both home and away appearances)
        if pd.notna(home_team_id):
            home_team_matches = df_hist[
                (df_hist['home_team_id'] == home_team_id) | 
                (df_hist['away_team_id'] == home_team_id)
            ].tail(10)
        else:
            home_team_matches = df_hist[
                (df_hist['home_team'] == home_team) | 
                (df_hist['away_team'] == home_team)
            ].tail(10)
        
        if pd.notna(away_team_id):
            away_team_matches = df_hist[
                (df_hist['home_team_id'] == away_team_id) | 
                (df_hist['away_team_id'] == away_team_id)
            ].tail(10)
        else:
            away_team_matches = df_hist[
                (df_hist['home_team'] == away_team) | 
                (df_hist['away_team'] == away_team)
            ].tail(10)
        
        for j, feat in enumerate(feature_cols):
            # Priority 1: Direct from future data (odds, derived odds)
            if feat in df_future.columns and pd.notna(future_row.get(feat)):
                X_future[idx, j] = future_row[feat]
                if idx == 0:
                    feature_sources['direct'] += 1
            
            # Priority 2: Team-specific historical imputation
            elif feat in df_hist.columns:
                value = None
                
                # For "home_" prefixed features, use the home team's history
                if feat.startswith('home_') and len(home_team_matches) > 0:
                    # Get the team's stats when they played at home
                    home_as_home = home_team_matches[home_team_matches['home_team'] == home_team]
                    if len(home_as_home) > 0 and feat in home_as_home.columns:
                        value = home_as_home[feat].mean()
                    elif feat in home_team_matches.columns:
                        value = home_team_matches[feat].mean()
                
                # For "away_" prefixed features, use the away team's history  
                elif feat.startswith('away_') and len(away_team_matches) > 0:
                    # Get the team's stats when they played away
                    away_as_away = away_team_matches[away_team_matches['away_team'] == away_team]
                    if len(away_as_away) > 0 and feat in away_as_away.columns:
                        value = away_as_away[feat].mean()
                    elif feat in away_team_matches.columns:
                        value = away_team_matches[feat].mean()
                
                # For streak features
                elif 'streak_home_' in feat and len(home_team_matches) > 0:
                    value = home_team_matches[feat].iloc[-1] if feat in home_team_matches.columns else None
                elif 'streak_away_' in feat and len(away_team_matches) > 0:
                    value = away_team_matches[feat].iloc[-1] if feat in away_team_matches.columns else None
                elif 'streak_both_' in feat:
                    # Use average of both teams
                    vals = []
                    if len(home_team_matches) > 0 and feat in home_team_matches.columns:
                        vals.append(home_team_matches[feat].iloc[-1])
                    if len(away_team_matches) > 0 and feat in away_team_matches.columns:
                        vals.append(away_team_matches[feat].iloc[-1])
                    value = np.mean(vals) if vals else None
                
                # For H2H features
                elif feat.startswith('h2h_'):
                    # Find matches between these two teams
                    if pd.notna(home_team_id) and pd.notna(away_team_id):
                        h2h = df_hist[
                            ((df_hist['home_team_id'] == home_team_id) & (df_hist['away_team_id'] == away_team_id)) |
                            ((df_hist['home_team_id'] == away_team_id) & (df_hist['away_team_id'] == home_team_id))
                        ]
                    else:
                        h2h = df_hist[
                            ((df_hist['home_team'] == home_team) & (df_hist['away_team'] == away_team)) |
                            ((df_hist['home_team'] == away_team) & (df_hist['away_team'] == home_team))
                        ]
                    if len(h2h) > 0 and feat in h2h.columns:
                        value = h2h[feat].iloc[-1]
                
                # For high_claims features
                elif feat.startswith('high_claims_') and len(home_team_matches) > 0:
                    value = home_team_matches[feat].mean() if feat in home_team_matches.columns else None
                
                if value is not None and pd.notna(value):
                    X_future[idx, j] = value
                    if idx == 0:
                        feature_sources['team_hist'] += 1
                else:
                    # Priority 3: Global average from all historical data
                    X_future[idx, j] = df_hist[feat].mean() if pd.notna(df_hist[feat].mean()) else 0
                    if idx == 0:
                        feature_sources['global'] += 1
            else:
                X_future[idx, j] = 0
    
    print(f"  Feature sources (first match): direct={feature_sources['direct']}, team_hist={feature_sources['team_hist']}, global={feature_sources['global']}")
    
    # =========================================================================
    # STEP 3: Scale and predict
    # =========================================================================
    X_future = np.nan_to_num(X_future, nan=0.0)
    X_future_scaled = scaler.transform(X_future)
    
    dfuture = xgb.DMatrix(X_future_scaled, feature_names=feature_cols)
    future_probs = model.predict(dfuture)
    future_pred = map_predictions_to_classes(future_probs)
    
    # =========================================================================
    # STEP 4: Build output
    # =========================================================================
    predictions = df_future[['date', 'home_team', 'away_team', 'league']].copy()
    for col in ['odds_1x2_home', 'odds_1x2_draw', 'odds_1x2_away']:
        if col in df_future.columns:
            predictions[col] = df_future[col]
    
    predictions['pred_away_prob'] = future_probs[:, 0]
    predictions['pred_draw_prob'] = future_probs[:, 1]
    predictions['pred_home_prob'] = future_probs[:, 2]
    predictions['predicted_result'] = future_pred
    predictions['confidence'] = future_probs.max(axis=1)
    
    print(f"\nPredictions summary:")
    print(f"  Home wins predicted: {(future_pred == 1).sum()}")
    print(f"  Draws predicted:     {(future_pred == 0).sum()}")
    print(f"  Away wins predicted: {(future_pred == -1).sum()}")
    print(f"\nConfidence stats:")
    print(f"  Min: {predictions['confidence'].min()*100:.1f}%")
    print(f"  Max: {predictions['confidence'].max()*100:.1f}%")
    print(f"  Mean: {predictions['confidence'].mean()*100:.1f}%")
    print(f"  >= 50%: {(predictions['confidence'] >= 0.50).sum()} matches")
    print(f"  >= 60%: {(predictions['confidence'] >= 0.60).sum()} matches")
    
    return predictions, future_probs


# =============================================================================
# VALUE BETTING - CONSERVATIVE THRESHOLD
# =============================================================================

def analyze_value_bets_conservative(predictions):
    """
    Identify value betting opportunities with CONSERVATIVE threshold.
    
    KEY IMPROVEMENT: EV threshold raised from 5% to 15%
    This filters out false positives from model overconfidence.
    """
    print_separator("STEP 8: VALUE BETTING (CONSERVATIVE)")
    
    if predictions is None or 'odds_1x2_home' not in predictions.columns:
        print("Predictions or odds not available.")
        return None
    
    predictions = predictions.copy()
    
    # Calculate implied probabilities
    predictions['implied_prob_home'] = 1 / predictions['odds_1x2_home']
    predictions['implied_prob_draw'] = 1 / predictions['odds_1x2_draw']
    predictions['implied_prob_away'] = 1 / predictions['odds_1x2_away']
    
    # Calculate expected value
    predictions['ev_home'] = (predictions['pred_home_prob'] * predictions['odds_1x2_home']) - 1
    predictions['ev_draw'] = (predictions['pred_draw_prob'] * predictions['odds_1x2_draw']) - 1
    predictions['ev_away'] = (predictions['pred_away_prob'] * predictions['odds_1x2_away']) - 1
    
    # Find best bet per match
    predictions['best_bet'] = predictions[['ev_home', 'ev_draw', 'ev_away']].idxmax(axis=1)
    predictions['best_bet'] = predictions['best_bet'].map({
        'ev_home': 'HOME', 'ev_draw': 'DRAW', 'ev_away': 'AWAY'
    })
    predictions['best_ev'] = predictions[['ev_home', 'ev_draw', 'ev_away']].max(axis=1)
    
    # ============ KEY IMPROVEMENT: CONSERVATIVE THRESHOLD ============
    print(f"EV Threshold: {VALUE_BET_THRESHOLD*100:.0f}% (was 5%)")
    
    value_bets = predictions[predictions['best_ev'] > VALUE_BET_THRESHOLD].copy()
    value_bets = value_bets.sort_values('best_ev', ascending=False)
    
    print(f"\nValue Bets Found: {len(value_bets)} / {len(predictions)}")
    
    if len(value_bets) > 0:
        print(f"\nConservative Value Betting Opportunities:")
        print("-" * 80)
        
        for idx, row in value_bets.head(10).iterrows():
            bet_type = row['best_bet']
            
            if bet_type == 'HOME':
                odds, model_prob, implied = row['odds_1x2_home'], row['pred_home_prob'], row['implied_prob_home']
            elif bet_type == 'DRAW':
                odds, model_prob, implied = row['odds_1x2_draw'], row['pred_draw_prob'], row['implied_prob_draw']
            else:
                odds, model_prob, implied = row['odds_1x2_away'], row['pred_away_prob'], row['implied_prob_away']
            
            ev_pct = row['best_ev'] * 100
            edge = (model_prob - implied) * 100
            
            print(f"\n{row['date']} | {row['home_team']} vs {row['away_team']}")
            print(f"  Bet: {bet_type} @ {odds:.2f}")
            print(f"  Model: {model_prob:.1%} | Bookmaker: {implied:.1%} | Edge: +{edge:.1f}%")
            print(f"  Expected Value: +{ev_pct:.1f}%")
    
    # Summary
    print(f"\n{'='*50}")
    print("CONSERVATIVE VALUE BETTING SUMMARY")
    print(f"{'='*50}")
    print(f"Threshold: EV > {VALUE_BET_THRESHOLD*100:.0f}%")
    print(f"Value bets: {len(value_bets)} (was 50 with 5% threshold)")
    
    if len(value_bets) > 0:
        print(f"Average EV: {value_bets['best_ev'].mean()*100:.1f}%")
        print(f"Bet types: {value_bets['best_bet'].value_counts().to_dict()}")
    
    return value_bets


# =============================================================================
# SAVE OUTPUTS
# =============================================================================

def save_outputs(model, scaler, feature_cols, predictions, value_bets, results):
    """Save model, predictions, and results."""
    print_separator("STEP 9: SAVING OUTPUTS")
    
    model.save_model('models/xgboost_model_v2.json')
    print("Saved: models/xgboost_model_v2.json")
    
    joblib.dump(scaler, 'models/scaler_v2.pkl')
    print("Saved: models/scaler_v2.pkl")
    
    with open('results/selected_features.txt', 'w') as f:
        for feat in feature_cols:
            f.write(feat + '\n')
    print(f"Saved: results/selected_features.txt ({len(feature_cols)} features)")
    
    if predictions is not None:
        predictions.to_csv('data/predictions/future_predictions_v2.csv', index=False)
        print("Saved: data/predictions/future_predictions_v2.csv")
    
    if value_bets is not None and len(value_bets) > 0:
        value_bets.to_csv('data/predictions/value_bets_v2.csv', index=False)
        print(f"Saved: data/predictions/value_bets_v2.csv ({len(value_bets)} bets)")
    
    results_df = pd.DataFrame(results).T
    results_df.to_csv('results/model_results_v2.csv')
    print("Saved: results/model_results_v2.csv")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    print("\n" + "=" * 70)
    print(" FOOTBALL ML PREDICTION - IMPROVED PIPELINE (v2)")
    print("=" * 70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    start_time = datetime.now()
    
    # Step 1: Load data
    df, all_feature_cols = load_processed_data()
    
    # Step 2: Temporal split (with all features first)
    train_data, valid_data, test_data = create_temporal_split(df, all_feature_cols)
    X_train_full, y_train, train_df = train_data
    X_valid_full, y_valid, valid_df = valid_data
    X_test_full, y_test, test_df = test_data
    
    # Step 3: Feature selection (reduce 1,087 → 200)
    selected_features = select_top_features(X_train_full, y_train, all_feature_cols, TOP_N_FEATURES)
    
    # Re-extract with selected features only
    X_train = train_df[selected_features].values
    X_valid = valid_df[selected_features].values
    X_test = test_df[selected_features].values
    
    # Step 4: Scale features
    print_separator("STEP 3.5: FEATURE SCALING")
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)
    X_test_scaled = scaler.transform(X_test)
    print(f"Scaler fitted on {len(selected_features)} features")
    
    # Step 5: Train improved model
    model, evals_result = train_xgboost_improved(X_train_scaled, y_train, X_valid_scaled, y_valid, selected_features)
    
    # Step 6: Evaluate
    results, test_probs, test_pred = evaluate_model(
        model, X_train_scaled, y_train, X_valid_scaled, y_valid, X_test_scaled, y_test, selected_features
    )
    
    # Step 7: Feature importance
    print_separator("STEP 6: FEATURE IMPORTANCE")
    importance = model.get_score(importance_type='weight')
    importance_named = {}
    for feat_key, score in importance.items():
        try:
            feat_idx = int(feat_key[1:])
            importance_named[selected_features[feat_idx]] = score
        except:
            continue
    sorted_imp = sorted(importance_named.items(), key=lambda x: x[1], reverse=True)
    print("Top 10 features:")
    for i, (f, s) in enumerate(sorted_imp[:10], 1):
        print(f"  {i}. {f[:50]}")
    
    # Step 8: Future predictions
    predictions, future_probs = predict_future_matches(model, scaler, selected_features, df)
    
    # Step 9: Conservative value betting
    value_bets = analyze_value_bets_conservative(predictions)
    
    # Step 10: Save
    save_outputs(model, scaler, selected_features, predictions, value_bets, results)
    
    # Final summary
    elapsed = (datetime.now() - start_time).total_seconds()
    
    print_separator("PIPELINE COMPLETE")
    print(f"Total time: {elapsed:.1f} seconds")
    print(f"\n*** IMPROVEMENTS SUMMARY ***")
    print(f"Features: 1,087 -> {len(selected_features)} (reduced)")
    print(f"Regularization: Increased 10x")
    print(f"Class weights: Applied")
    print(f"EV threshold: 5% -> {VALUE_BET_THRESHOLD*100:.0f}%")
    
    print(f"\nModel Performance:")
    print(f"  Train Accuracy: {results['Train']['accuracy']:.2%}")
    print(f"  Valid Accuracy: {results['Valid']['accuracy']:.2%}")
    print(f"  Test Accuracy:  {results['Test']['accuracy']:.2%}")
    print(f"  Overfitting gap: {results['Train']['accuracy'] - results['Valid']['accuracy']:.2%}")
    
    if value_bets is not None:
        print(f"\nConservative Value Bets: {len(value_bets)} (was 50)")
    
    return model, scaler, predictions, value_bets


if __name__ == "__main__":
    model, scaler, predictions, value_bets = main()
