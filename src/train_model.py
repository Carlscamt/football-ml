#!/usr/bin/env python3
"""
Football Match Prediction - ML Training & Value Betting Pipeline
=================================================================

XGBoost multiclass classifier for match outcome prediction:
- 1  = Home Win
- 0  = Draw  
- -1 = Away Win

Features:
- Temporal train/valid/test split (no data leakage)
- 1,087 features from processed dataset
- Feature importance analysis
- Future match predictions
- Value betting opportunity identification

Author: Claude (Anthropic)
Version: 1.0.0

Usage:
    python train_model.py
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, log_loss, confusion_matrix, classification_report
import xgboost as xgb
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION
# =============================================================================

# File paths
PROCESSED_DATA = 'data/processed/processed_features_extended.csv'
FUTURE_DATA = 'data/predictions/sofascore_future_6lg_58matches_20260111.csv'
HISTORICAL_DATA = 'data/raw/sofascore_parallel_6lg_3yr_2434matches.csv'

# Temporal split dates
TRAIN_CUTOFF = '2024-08-01'    # Train: before this
VALID_CUTOFF = '2025-01-01'    # Valid: between train_cutoff and this, Test: after

# XGBoost hyperparameters
XGB_PARAMS = {
    'max_depth': 6,
    'learning_rate': 0.05,
    'objective': 'multi:softprob',
    'num_class': 3,
    'eval_metric': 'mlogloss',
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': 42,
    'verbosity': 0
}

NUM_BOOST_ROUNDS = 500
EARLY_STOPPING_ROUNDS = 50

# Value betting threshold
VALUE_BET_THRESHOLD = 0.05  # 5% minimum expected value


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def print_separator(title=""):
    """Print a separator line with optional title."""
    if title:
        print(f"\n{'='*70}")
        print(f" {title}")
        print(f"{'='*70}")
    else:
        print("=" * 70)


# Label encoding for XGBoost (requires 0 to num_class-1)
# Original: -1 (away), 0 (draw), 1 (home)
# Encoded:  0 (away), 1 (draw), 2 (home)
LABEL_ENCODE = {-1: 0, 0: 1, 1: 2}
LABEL_DECODE = {0: -1, 1: 0, 2: 1}


def encode_labels(y):
    """Encode labels from -1/0/1 to 0/1/2 for XGBoost."""
    return np.array([LABEL_ENCODE[label] for label in y])


def decode_labels(y_encoded):
    """Decode labels from 0/1/2 back to -1/0/1."""
    return np.array([LABEL_DECODE[label] for label in y_encoded])


def map_predictions_to_classes(pred_probs):
    """
    Map XGBoost prediction probabilities to original class labels.
    
    XGBoost outputs: [P(class 0), P(class 1), P(class 2)]
    Encoded as:      [P(away), P(draw), P(home)]
    Original labels: -1 (away), 0 (draw), 1 (home)
    
    Returns predictions in original label space (-1/0/1).
    """
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
    
    print(f"\nFeature breakdown:")
    rolling_cols = [c for c in feature_cols if 'rolling' in c]
    odds_cols = [c for c in feature_cols if 'odds_' in c or 'implied_' in c or 'log_odds' in c or 'overround' in c]
    h2h_cols = [c for c in feature_cols if 'h2h' in c]
    streak_cols = [c for c in feature_cols if 'streak_' in c]
    
    print(f"  Rolling features: {len(rolling_cols)}")
    print(f"  Odds features:    {len(odds_cols)}")
    print(f"  H2H features:     {len(h2h_cols)}")
    print(f"  Streak features:  {len(streak_cols)}")
    print(f"  TOTAL:            {len(feature_cols)}")
    
    # Handle nulls
    null_count = df[feature_cols].isnull().sum().sum()
    if null_count > 0:
        print(f"\nFilling {null_count:,} null values with 0...")
        df[feature_cols] = df[feature_cols].fillna(0)
    
    return df, feature_cols


# =============================================================================
# TEMPORAL SPLIT
# =============================================================================

def create_temporal_split(df, feature_cols):
    """
    Create train/validation/test split based on dates.
    
    CRITICAL: This is time-series data, so we CANNOT randomly shuffle.
    Train < Validation < Test (chronologically)
    """
    print_separator("STEP 2: TEMPORAL SPLIT")
    
    train_cutoff = pd.to_datetime(TRAIN_CUTOFF)
    valid_cutoff = pd.to_datetime(VALID_CUTOFF)
    
    # Convert date column
    df['date'] = pd.to_datetime(df['date'])
    
    # Split
    train_mask = df['date'] < train_cutoff
    valid_mask = (df['date'] >= train_cutoff) & (df['date'] < valid_cutoff)
    test_mask = df['date'] >= valid_cutoff
    
    train_data = df[train_mask].copy()
    valid_data = df[valid_mask].copy()
    test_data = df[test_mask].copy()
    
    print(f"Train: {len(train_data):,} matches ({train_data['date'].min().date()} to {train_data['date'].max().date()})")
    print(f"Valid: {len(valid_data):,} matches ({valid_data['date'].min().date()} to {valid_data['date'].max().date()})")
    print(f"Test:  {len(test_data):,} matches ({test_data['date'].min().date()} to {test_data['date'].max().date()})")
    
    # Extract X and y
    X_train = train_data[feature_cols].values
    y_train = train_data['target'].values
    
    X_valid = valid_data[feature_cols].values
    y_valid = valid_data['target'].values
    
    X_test = test_data[feature_cols].values
    y_test = test_data['target'].values
    
    # Print target distribution per split
    print(f"\nTarget distribution per split:")
    for name, y in [('Train', y_train), ('Valid', y_valid), ('Test', y_test)]:
        away = (y == -1).sum() / len(y) * 100
        draw = (y == 0).sum() / len(y) * 100
        home = (y == 1).sum() / len(y) * 100
        print(f"  {name}: Away {away:.1f}% | Draw {draw:.1f}% | Home {home:.1f}%")
    
    # Baseline accuracy (always predict most frequent class)
    most_frequent = np.bincount(y_train + 1).argmax() - 1  # Shift for -1/0/1
    baseline_acc = (y_train == most_frequent).mean()
    print(f"\nBaseline accuracy (always predict {most_frequent}): {baseline_acc:.2%}")
    
    return (X_train, y_train, train_data), (X_valid, y_valid, valid_data), (X_test, y_test, test_data)


# =============================================================================
# FEATURE SCALING
# =============================================================================

def scale_features(X_train, X_valid, X_test, feature_cols):
    """
    Scale features using StandardScaler.
    
    CRITICAL: Fit scaler ONLY on training data to prevent data leakage.
    """
    print_separator("STEP 3: FEATURE SCALING")
    
    scaler = StandardScaler()
    scaler.fit(X_train)
    
    X_train_scaled = scaler.transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Scaler fitted on training data")
    print(f"Train - Mean: {X_train_scaled.mean():.6f}, Std: {X_train_scaled.std():.6f}")
    print(f"Valid - Mean: {X_valid_scaled.mean():.6f}, Std: {X_valid_scaled.std():.6f}")
    print(f"Test  - Mean: {X_test_scaled.mean():.6f}, Std: {X_test_scaled.std():.6f}")
    
    # Create DMatrix objects for XGBoost
    dtrain = xgb.DMatrix(X_train_scaled, label=X_train, feature_names=feature_cols)
    dvalid = xgb.DMatrix(X_valid_scaled, label=X_valid, feature_names=feature_cols)
    dtest = xgb.DMatrix(X_test_scaled, label=X_test, feature_names=feature_cols)
    
    return scaler, X_train_scaled, X_valid_scaled, X_test_scaled


# =============================================================================
# MODEL TRAINING
# =============================================================================

def train_xgboost(X_train, y_train, X_valid, y_valid, feature_cols):
    """Train XGBoost model with early stopping."""
    print_separator("STEP 4: MODEL TRAINING")
    
    # Encode labels for XGBoost (requires 0 to num_class-1)
    y_train_encoded = encode_labels(y_train)
    y_valid_encoded = encode_labels(y_valid)
    
    print("XGBoost Parameters:")
    for k, v in XGB_PARAMS.items():
        print(f"  {k}: {v}")
    
    # Create DMatrix with encoded labels
    dtrain = xgb.DMatrix(X_train, label=y_train_encoded, feature_names=feature_cols)
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
        verbose_eval=50
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
    
    # Create DMatrix for prediction
    dtrain = xgb.DMatrix(X_train, feature_names=feature_cols)
    dvalid = xgb.DMatrix(X_valid, feature_names=feature_cols)
    dtest = xgb.DMatrix(X_test, feature_names=feature_cols)
    
    # Get predictions
    train_probs = model.predict(dtrain)
    valid_probs = model.predict(dvalid)
    test_probs = model.predict(dtest)
    
    train_pred = map_predictions_to_classes(train_probs)
    valid_pred = map_predictions_to_classes(valid_probs)
    test_pred = map_predictions_to_classes(test_probs)
    
    # Calculate metrics
    results = {}
    for name, y_true, y_pred, y_probs in [
        ('Train', y_train, train_pred, train_probs),
        ('Valid', y_valid, valid_pred, valid_probs),
        ('Test', y_test, test_pred, test_probs)
    ]:
        acc = accuracy_score(y_true, y_pred)
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Log loss (need to align probabilities with label order -1, 0, 1)
        try:
            ll = log_loss(y_true, y_probs, labels=[-1, 0, 1])
        except:
            ll = np.nan
        
        results[name] = {
            'accuracy': acc,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'log_loss': ll
        }
        
        print(f"\n{name} Set:")
        print(f"  Accuracy:    {acc:.4f}")
        print(f"  F1 (macro):  {f1_macro:.4f}")
        print(f"  F1 (weighted): {f1_weighted:.4f}")
        print(f"  Log Loss:    {ll:.4f}")
    
    # Overfitting check
    overfit_gap = results['Train']['accuracy'] - results['Valid']['accuracy']
    print(f"\nOverfitting gap (Train Acc - Valid Acc): {overfit_gap:.4f}")
    if overfit_gap > 0.10:
        print("  WARNING: Model may be significantly overfitting!")
    elif overfit_gap > 0.05:
        print("  CAUTION: Some overfitting detected.")
    else:
        print("  OK: Model appears well-regularized.")
    
    # Confusion matrix for test set
    print(f"\nTest Set Confusion Matrix:")
    cm = confusion_matrix(y_test, test_pred, labels=[-1, 0, 1])
    print("              Predicted")
    print("           -1    0    1")
    print("Actual -1: {:4d} {:4d} {:4d}".format(cm[0, 0], cm[0, 1], cm[0, 2]))
    print("       0:  {:4d} {:4d} {:4d}".format(cm[1, 0], cm[1, 1], cm[1, 2]))
    print("       1:  {:4d} {:4d} {:4d}".format(cm[2, 0], cm[2, 1], cm[2, 2]))
    
    # Per-class accuracy
    print(f"\nPer-class accuracy (Test):")
    label_names = {-1: 'Away', 0: 'Draw', 1: 'Home'}
    for i, label in enumerate([-1, 0, 1]):
        class_total = cm[i].sum()
        class_correct = cm[i, i]
        class_acc = class_correct / class_total if class_total > 0 else 0
        print(f"  {label:2d} ({label_names[label]:5s}): {class_acc:.2%} ({class_correct}/{class_total})")
    
    return results, test_probs, test_pred


# =============================================================================
# FEATURE IMPORTANCE
# =============================================================================

def analyze_feature_importance(model, feature_cols):
    """Analyze and visualize feature importance."""
    print_separator("STEP 6: FEATURE IMPORTANCE")
    
    # Get importance scores
    importance = model.get_score(importance_type='weight')
    
    if not importance:
        print("No feature importance available.")
        return None
    
    # Map f0, f1, f2... to actual feature names
    importance_named = {}
    for feat_key, score in importance.items():
        try:
            feat_idx = int(feat_key[1:])  # Remove 'f' prefix
            feat_name = feature_cols[feat_idx]
            importance_named[feat_name] = score
        except (ValueError, IndexError):
            continue
    
    # Sort by importance
    sorted_importance = sorted(importance_named.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\nTop 20 Most Important Features:")
    for i, (feat, score) in enumerate(sorted_importance[:20], 1):
        print(f"  {i:2d}. {feat:55s} : {score:5.0f}")
    
    # Categorize top features
    top_50 = [f[0] for f in sorted_importance[:50]]
    rolling_count = sum(1 for f in top_50 if 'rolling' in f)
    odds_count = sum(1 for f in top_50 if 'odds_' in f or 'implied_' in f)
    h2h_count = sum(1 for f in top_50 if 'h2h' in f)
    streak_count = sum(1 for f in top_50 if 'streak_' in f)
    
    print(f"\nTop 50 features by category:")
    print(f"  Rolling: {rolling_count}")
    print(f"  Odds:    {odds_count}")
    print(f"  H2H:     {h2h_count}")
    print(f"  Streak:  {streak_count}")
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    top_n = min(20, len(sorted_importance))
    features = [f[0][:40] for f in sorted_importance[:top_n]]  # Truncate names
    scores = [f[1] for f in sorted_importance[:top_n]]
    
    plt.barh(range(top_n), scores, color='steelblue')
    plt.yticks(range(top_n), features, fontsize=9)
    plt.xlabel('Feature Importance (Usage Frequency)', fontsize=11)
    plt.title('Top 20 Features for Match Prediction', fontsize=13, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('results/feature_importance.png', dpi=120, bbox_inches='tight')
    print(f"\nSaved: results/feature_importance.png")
    
    return sorted_importance


# =============================================================================
# FUTURE MATCH PREDICTIONS
# =============================================================================

def predict_future_matches(model, scaler, feature_cols, df_historical):
    """
    Predict outcomes for future matches.
    
    Since future matches don't have rolling features pre-calculated,
    we need to compute them from historical data.
    """
    print_separator("STEP 7: PREDICTING FUTURE MATCHES")
    
    if not os.path.exists(FUTURE_DATA):
        print(f"Future data file not found: {FUTURE_DATA}")
        return None
    
    df_future = pd.read_csv(FUTURE_DATA)
    print(f"Loaded {len(df_future)} future matches")
    print(f"Date range: {df_future['date'].min()} to {df_future['date'].max()}")
    
    # Check which features are available
    available_features = [c for c in feature_cols if c in df_future.columns]
    missing_features = [c for c in feature_cols if c not in df_future.columns]
    
    print(f"\nFeature availability:")
    print(f"  Available: {len(available_features)}")
    print(f"  Missing:   {len(missing_features)}")
    
    # For missing features, we'll compute simple rolling averages from historical data
    # This is a simplified approach - for production, use the full feature pipeline
    
    if len(missing_features) > 0:
        print("\nComputing rolling features for future matches from historical data...")
        
        # Get the processed historical data to extract team averages
        df_hist = pd.read_csv(PROCESSED_DATA)
        
        # Create team average features from recent matches
        # Get last 3 months of data for team averages
        recent_cutoff = df_hist['date'].max()  # Use most recent data
        
        # For each future match, get the team's average feature values
        X_future = np.zeros((len(df_future), len(feature_cols)))
        
        for idx, future_row in df_future.iterrows():
            home_team_id = future_row['home_team_id']
            away_team_id = future_row['away_team_id']
            
            # Get average features for home team (from their home matches)
            home_matches = df_hist[df_hist['home_team_id'] == home_team_id].tail(10)
            
            # Get average features for away team (from their away matches)
            away_matches = df_hist[df_hist['away_team_id'] == away_team_id].tail(10)
            
            for j, feat in enumerate(feature_cols):
                if feat in df_future.columns and pd.notna(future_row.get(feat)):
                    X_future[idx, j] = future_row[feat]
                elif 'home_' in feat and feat in df_hist.columns:
                    # Use home team's historical average
                    if len(home_matches) > 0:
                        X_future[idx, j] = home_matches[feat].mean()
                elif 'away_' in feat and feat in df_hist.columns:
                    # Use away team's historical average
                    if len(away_matches) > 0:
                        X_future[idx, j] = away_matches[feat].mean()
                else:
                    # Use overall average from training data
                    if feat in df_hist.columns:
                        X_future[idx, j] = df_hist[feat].mean()
                    else:
                        X_future[idx, j] = 0
        
        print(f"  Computed features for {len(df_future)} matches")
    else:
        X_future = df_future[feature_cols].values
    
    # Fill any remaining NaN
    X_future = np.nan_to_num(X_future, nan=0.0)
    
    # Scale features
    X_future_scaled = scaler.transform(X_future)
    
    # Predict
    dfuture = xgb.DMatrix(X_future_scaled, feature_names=feature_cols)
    future_probs = model.predict(dfuture)
    future_pred = map_predictions_to_classes(future_probs)
    
    # Create predictions dataframe
    predictions = df_future[['date', 'home_team', 'away_team', 'league']].copy()
    
    # Add odds if available
    for col in ['odds_1x2_home', 'odds_1x2_draw', 'odds_1x2_away']:
        if col in df_future.columns:
            predictions[col] = df_future[col]
    
    # Add predictions
    predictions['pred_away_prob'] = future_probs[:, 0]
    predictions['pred_draw_prob'] = future_probs[:, 1]
    predictions['pred_home_prob'] = future_probs[:, 2]
    predictions['predicted_result'] = future_pred
    predictions['confidence'] = future_probs.max(axis=1)
    
    # Display predictions
    print(f"\nFuture Match Predictions:")
    print(predictions[['date', 'home_team', 'away_team', 
                       'pred_home_prob', 'pred_draw_prob', 'pred_away_prob', 
                       'confidence']].head(15).to_string(index=False))
    
    return predictions, future_probs


# =============================================================================
# VALUE BETTING ANALYSIS
# =============================================================================

def analyze_value_bets(predictions):
    """Identify value betting opportunities."""
    print_separator("STEP 8: VALUE BETTING ANALYSIS")
    
    if predictions is None:
        print("No predictions available for value betting analysis.")
        return None
    
    # Check if odds are available
    if 'odds_1x2_home' not in predictions.columns:
        print("Odds data not available. Skipping value betting analysis.")
        return None
    
    # Calculate implied probabilities from bookmaker odds
    predictions['implied_prob_home'] = 1 / predictions['odds_1x2_home']
    predictions['implied_prob_draw'] = 1 / predictions['odds_1x2_draw']
    predictions['implied_prob_away'] = 1 / predictions['odds_1x2_away']
    
    # Calculate expected value for each outcome
    # EV = (Model Probability × Odds) - 1
    predictions['ev_home'] = (predictions['pred_home_prob'] * predictions['odds_1x2_home']) - 1
    predictions['ev_draw'] = (predictions['pred_draw_prob'] * predictions['odds_1x2_draw']) - 1
    predictions['ev_away'] = (predictions['pred_away_prob'] * predictions['odds_1x2_away']) - 1
    
    # Find best bet per match
    predictions['best_bet'] = predictions[['ev_home', 'ev_draw', 'ev_away']].idxmax(axis=1)
    predictions['best_bet'] = predictions['best_bet'].map({
        'ev_home': 'HOME',
        'ev_draw': 'DRAW', 
        'ev_away': 'AWAY'
    })
    predictions['best_ev'] = predictions[['ev_home', 'ev_draw', 'ev_away']].max(axis=1)
    
    # Identify value bets (positive EV above threshold)
    value_bets = predictions[predictions['best_ev'] > VALUE_BET_THRESHOLD].copy()
    value_bets = value_bets.sort_values('best_ev', ascending=False)
    
    print(f"Value Bets Found (EV > {VALUE_BET_THRESHOLD*100:.0f}%): {len(value_bets)} / {len(predictions)}")
    
    if len(value_bets) > 0:
        print(f"\nTop Value Betting Opportunities:")
        print("-" * 80)
        
        for idx, row in value_bets.head(10).iterrows():
            bet_type = row['best_bet']
            
            if bet_type == 'HOME':
                odds = row['odds_1x2_home']
                model_prob = row['pred_home_prob']
                implied_prob = row['implied_prob_home']
            elif bet_type == 'DRAW':
                odds = row['odds_1x2_draw']
                model_prob = row['pred_draw_prob']
                implied_prob = row['implied_prob_draw']
            else:  # AWAY
                odds = row['odds_1x2_away']
                model_prob = row['pred_away_prob']
                implied_prob = row['implied_prob_away']
            
            ev_pct = row['best_ev'] * 100
            edge = (model_prob - implied_prob) * 100
            
            print(f"\n{row['date']} | {row['home_team']} vs {row['away_team']}")
            print(f"  Bet: {bet_type} @ {odds:.2f}")
            print(f"  Model: {model_prob:.1%} | Bookmaker: {implied_prob:.1%} | Edge: +{edge:.1f}%")
            print(f"  Expected Value: +{ev_pct:.1f}%")
    
    # Summary statistics
    print(f"\n{'='*40}")
    print("VALUE BETTING SUMMARY")
    print(f"{'='*40}")
    print(f"Total matches analyzed: {len(predictions)}")
    print(f"Value bets found: {len(value_bets)}")
    
    if len(value_bets) > 0:
        print(f"Average EV of value bets: {value_bets['best_ev'].mean()*100:.1f}%")
        print(f"Best EV: {value_bets['best_ev'].max()*100:.1f}%")
        
        bet_types = value_bets['best_bet'].value_counts()
        print(f"\nValue bets by type:")
        for bet_type, count in bet_types.items():
            print(f"  {bet_type}: {count}")
    
    return value_bets


# =============================================================================
# SAVE OUTPUTS
# =============================================================================

def save_outputs(model, scaler, feature_cols, predictions, value_bets, results):
    """Save model, predictions, and results."""
    print_separator("STEP 9: SAVING OUTPUTS")
    
    # Save model
    model.save_model('models/xgboost_model.json')
    print("Saved: models/xgboost_model.json")
    
    # Save scaler
    joblib.dump(scaler, 'models/scaler.pkl')
    print("Saved: models/scaler.pkl")
    
    # Save feature list
    with open('results/feature_columns.txt', 'w') as f:
        for feat in feature_cols:
            f.write(feat + '\n')
    print("Saved: results/feature_columns.txt")
    
    # Save predictions
    if predictions is not None:
        predictions.to_csv('data/predictions/future_predictions.csv', index=False)
        print("Saved: data/predictions/future_predictions.csv")
    
    # Save value bets
    if value_bets is not None and len(value_bets) > 0:
        value_bets.to_csv('data/predictions/value_bets.csv', index=False)
        print("Saved: data/predictions/value_bets.csv")
    
    # Save results summary
    results_df = pd.DataFrame(results).T
    results_df.to_csv('results/model_results.csv')
    print("Saved: results/model_results.csv")
    
    print("\nAll outputs saved successfully!")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution pipeline."""
    print("\n" + "=" * 70)
    print(" FOOTBALL MATCH PREDICTION - ML TRAINING PIPELINE")
    print("=" * 70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    start_time = datetime.now()
    
    # Step 1: Load data
    df, feature_cols = load_processed_data()
    
    # Step 2: Temporal split
    train_data, valid_data, test_data = create_temporal_split(df, feature_cols)
    X_train, y_train, train_df = train_data
    X_valid, y_valid, valid_df = valid_data
    X_test, y_test, test_df = test_data
    
    # Step 3: Scale features
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)
    X_test_scaled = scaler.transform(X_test)
    
    # Step 4: Train model
    model, evals_result = train_xgboost(X_train_scaled, y_train, X_valid_scaled, y_valid, feature_cols)
    
    # Step 5: Evaluate
    results, test_probs, test_pred = evaluate_model(
        model, X_train_scaled, y_train, X_valid_scaled, y_valid, X_test_scaled, y_test, feature_cols
    )
    
    # Step 6: Feature importance
    importance = analyze_feature_importance(model, feature_cols)
    
    # Step 7: Future predictions
    predictions, future_probs = predict_future_matches(model, scaler, feature_cols, df)
    
    # Step 8: Value betting
    value_bets = analyze_value_bets(predictions)
    
    # Step 9: Save outputs
    save_outputs(model, scaler, feature_cols, predictions, value_bets, results)
    
    # Final summary
    elapsed = (datetime.now() - start_time).total_seconds()
    
    print_separator("PIPELINE COMPLETE")
    print(f"Total time: {elapsed:.1f} seconds")
    print(f"\nModel Performance:")
    print(f"  Train Accuracy: {results['Train']['accuracy']:.2%}")
    print(f"  Valid Accuracy: {results['Valid']['accuracy']:.2%}")
    print(f"  Test Accuracy:  {results['Test']['accuracy']:.2%}")
    
    if value_bets is not None and len(value_bets) > 0:
        print(f"\nValue Bets: {len(value_bets)} opportunities found")
    
    print(f"\nOutputs:")
    print(f"  - xgboost_model.json")
    print(f"  - scaler.pkl")
    print(f"  - feature_importance.png")
    print(f"  - future_predictions.csv")
    print(f"  - value_bets.csv")
    print(f"  - model_results.csv")
    
    return model, scaler, predictions, value_bets


if __name__ == "__main__":
    model, scaler, predictions, value_bets = main()
