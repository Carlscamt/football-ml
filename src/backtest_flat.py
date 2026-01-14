#!/usr/bin/env python3
"""
Flat Betting Strategy Backtest
==============================
Simulates betting 1 unit on each model prediction.
Tests on the holdout test set (newest 20% of matches).
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import os
from datetime import datetime
from sklearn.preprocessing import StandardScaler

# =============================================================================
# CONFIGURATION
# =============================================================================

PROCESSED_DATA = 'data/processed/processed_features_extended.csv'
MODEL_PATH = 'models/xgboost_model_v2.json'
FEATURES_PATH = 'results/selected_features.txt'

# Betting settings
UNIT_SIZE = 1.0  # Bet 1 unit per match
MIN_CONFIDENCE = 0.0  # Bet on all predictions (no filter)
MIN_ODDS = 1.0  # No minimum odds filter
MAX_ODDS = 100.0  # No maximum odds filter

# Temporal split
TRAIN_CUTOFF = '2024-08-01'
VALID_CUTOFF = '2025-01-01'
TEST_CUTOFF = '2025-01-01'


def load_model_and_data():
    """Load trained model and data, recreate scaler."""
    print("=" * 60)
    print("FLAT BETTING BACKTEST")
    print("=" * 60)
    
    # Load model
    model = xgb.Booster()
    model.load_model(MODEL_PATH)
    print(f"[+] Model loaded: {MODEL_PATH}")
    
    # Load features
    with open(FEATURES_PATH, 'r') as f:
        features = [line.strip() for line in f if line.strip()]
    print(f"[+] Features loaded: {len(features)}")
    
    # Load data
    df = pd.read_csv(PROCESSED_DATA)
    df['date'] = pd.to_datetime(df['date'])
    print(f"[+] Data loaded: {len(df):,} matches")
    
    # Recreate scaler from training data (same as training script)
    df_train = df[df['date'] < VALID_CUTOFF]
    X_train = df_train[features].fillna(0)
    scaler = StandardScaler()
    scaler.fit(X_train)
    print(f"[+] Scaler fitted on {len(df_train):,} training samples")
    
    return model, scaler, features, df


def run_backtest(model, scaler, features, df, 
                 min_confidence=0.0, min_odds=1.0, max_odds=100.0,
                 strategy_name="All Predictions"):
    """Run flat betting backtest on test data."""
    
    # Split into test set
    df_test = df[df['date'] >= TEST_CUTOFF].copy()
    print(f"\nTest set: {len(df_test):,} matches ({TEST_CUTOFF} onwards)")
    
    # Get features for prediction
    X_test = df_test[features].fillna(0)
    X_test_scaled = scaler.transform(X_test)
    dtest = xgb.DMatrix(X_test_scaled, feature_names=features)
    
    # Predict
    probs = model.predict(dtest)
    df_test['pred_away_prob'] = probs[:, 0]
    df_test['pred_draw_prob'] = probs[:, 1]
    df_test['pred_home_prob'] = probs[:, 2]
    df_test['predicted'] = np.argmax(probs, axis=1) - 1  # -1, 0, 1
    df_test['confidence'] = probs.max(axis=1)
    
    # Calculate actual result
    df_test['actual'] = np.sign(df_test['home_score'] - df_test['away_score']).astype(int)
    
    # ==========================================================================
    # SIMULATE FLAT BETTING
    # ==========================================================================
    
    results = []
    
    for _, row in df_test.iterrows():
        pred = row['predicted']
        actual = row['actual']
        conf = row['confidence']
        
        # Get odds for predicted outcome
        if pred == 1:  # Home
            odds = row.get('odds_1x2_home', np.nan)
            prob = row['pred_home_prob']
        elif pred == 0:  # Draw
            odds = row.get('odds_1x2_draw', np.nan)
            prob = row['pred_draw_prob']
        else:  # Away
            odds = row.get('odds_1x2_away', np.nan)
            prob = row['pred_away_prob']
        
        # Skip if no odds
        if pd.isna(odds) or odds <= 0:
            continue
        
        # Apply filters
        if conf < min_confidence:
            continue
        if odds < min_odds or odds > max_odds:
            continue
        
        # Calculate result
        stake = UNIT_SIZE
        if pred == actual:
            profit = stake * (odds - 1)  # Win
            won = True
        else:
            profit = -stake  # Loss
            won = False
        
        results.append({
            'date': row['date'],
            'match': f"{row['home_team']} vs {row['away_team']}",
            'predicted': {1: 'Home', 0: 'Draw', -1: 'Away'}[pred],
            'actual': {1: 'Home', 0: 'Draw', -1: 'Away'}[actual],
            'odds': odds,
            'confidence': conf,
            'stake': stake,
            'profit': profit,
            'won': won
        })
    
    df_results = pd.DataFrame(results)
    
    if len(df_results) == 0:
        print("No bets placed with current filters!")
        return None
    
    # ==========================================================================
    # CALCULATE METRICS
    # ==========================================================================
    
    total_bets = len(df_results)
    wins = df_results['won'].sum()
    losses = total_bets - wins
    total_staked = df_results['stake'].sum()
    total_profit = df_results['profit'].sum()
    roi = (total_profit / total_staked) * 100
    win_rate = (wins / total_bets) * 100
    
    # Cumulative profit for plotting
    df_results['cumulative_profit'] = df_results['profit'].cumsum()
    
    # Best/worst streaks
    df_results['streak'] = (df_results['won'] != df_results['won'].shift()).cumsum()
    win_streaks = df_results[df_results['won']].groupby('streak').size()
    loss_streaks = df_results[~df_results['won']].groupby('streak').size()
    best_streak = win_streaks.max() if len(win_streaks) > 0 else 0
    worst_streak = loss_streaks.max() if len(loss_streaks) > 0 else 0
    
    # Average odds
    avg_odds = df_results['odds'].mean()
    avg_winning_odds = df_results[df_results['won']]['odds'].mean() if wins > 0 else 0
    
    # ==========================================================================
    # PRINT RESULTS
    # ==========================================================================
    
    print(f"\n{'='*60}")
    print(f"STRATEGY: {strategy_name}")
    print(f"{'='*60}")
    print(f"\nBetting Configuration:")
    print(f"  Unit size:       ${UNIT_SIZE:.2f}")
    print(f"  Min confidence:  {min_confidence*100:.0f}%")
    print(f"  Odds range:      {min_odds:.1f} - {max_odds:.1f}")
    
    print(f"\nResults:")
    print(f"  Total bets:      {total_bets}")
    print(f"  Wins:            {wins} ({win_rate:.1f}%)")
    print(f"  Losses:          {losses}")
    print(f"  Total staked:    ${total_staked:.2f}")
    print(f"  Total profit:    ${total_profit:+.2f}")
    print(f"  ROI:             {roi:+.1f}%")
    
    print(f"\nOdds Analysis:")
    print(f"  Average odds:    {avg_odds:.2f}")
    print(f"  Avg winning odds:{avg_winning_odds:.2f}")
    
    print(f"\nStreaks:")
    print(f"  Best win streak: {best_streak}")
    print(f"  Worst loss streak: {worst_streak}")
    
    # Profit by prediction type
    print(f"\nBy Prediction Type:")
    for pred_type in ['Home', 'Draw', 'Away']:
        subset = df_results[df_results['predicted'] == pred_type]
        if len(subset) > 0:
            p = subset['profit'].sum()
            w = subset['won'].mean() * 100
            print(f"  {pred_type:5}: {len(subset):3} bets, {w:.0f}% win, ${p:+.2f} profit")
    
    return df_results


def run_multiple_strategies(model, scaler, features, df):
    """Test multiple betting strategies."""
    
    strategies = [
        {"name": "All Predictions (No Filter)", "min_conf": 0.0, "min_odds": 1.0, "max_odds": 100},
        {"name": "Confidence > 40%", "min_conf": 0.40, "min_odds": 1.0, "max_odds": 100},
        {"name": "Confidence > 50%", "min_conf": 0.50, "min_odds": 1.0, "max_odds": 100},
        {"name": "Confidence > 60%", "min_conf": 0.60, "min_odds": 1.0, "max_odds": 100},
        {"name": "Odds 1.5-2.5 Only", "min_conf": 0.0, "min_odds": 1.5, "max_odds": 2.5},
        {"name": "Odds 2.0-3.0 Only", "min_conf": 0.0, "min_odds": 2.0, "max_odds": 3.0},
        {"name": "Conf>50% + Odds 1.5-3.0", "min_conf": 0.50, "min_odds": 1.5, "max_odds": 3.0},
    ]
    
    summary = []
    
    for strat in strategies:
        result = run_backtest(
            model, scaler, features, df,
            min_confidence=strat['min_conf'],
            min_odds=strat['min_odds'],
            max_odds=strat['max_odds'],
            strategy_name=strat['name']
        )
        
        if result is not None:
            total_bets = len(result)
            profit = result['profit'].sum()
            roi = (profit / result['stake'].sum()) * 100
            win_rate = result['won'].mean() * 100
            
            summary.append({
                'Strategy': strat['name'],
                'Bets': total_bets,
                'Win Rate': f"{win_rate:.1f}%",
                'Profit': f"${profit:+.2f}",
                'ROI': f"{roi:+.1f}%"
            })
    
    # Print summary table
    print("\n" + "=" * 80)
    print("STRATEGY COMPARISON SUMMARY")
    print("=" * 80)
    
    df_summary = pd.DataFrame(summary)
    print(df_summary.to_string(index=False))
    
    # Save summary
    df_summary.to_csv('results/betting_strategy_comparison.csv', index=False)
    print(f"\n[+] Summary saved: results/betting_strategy_comparison.csv")
    
    return df_summary


if __name__ == "__main__":
    model, scaler, features, df = load_model_and_data()
    
    print("\n" + "=" * 60)
    print("RUNNING FLAT BETTING BACKTEST")
    print("Strategy: 1 unit per bet")
    print("=" * 60)
    
    # Run multiple strategies
    summary = run_multiple_strategies(model, scaler, features, df)
