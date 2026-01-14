#!/usr/bin/env python3
"""Quick analysis of Confidence > 60% strategy"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

# Load everything
model = xgb.Booster()
model.load_model('models/xgboost_model_v2.json')

with open('results/selected_features.txt', 'r') as f:
    features = [line.strip() for line in f if line.strip()]

df = pd.read_csv('data/processed/processed_features_extended.csv')
df['date'] = pd.to_datetime(df['date'])

# Recreate scaler
df_train = df[df['date'] < '2025-01-01']
scaler = StandardScaler()
scaler.fit(df_train[features].fillna(0))

# Test set
df_test = df[df['date'] >= '2025-01-01'].copy()
X_test = df_test[features].fillna(0)
X_test_scaled = scaler.transform(X_test)
dtest = xgb.DMatrix(X_test_scaled, feature_names=features)

probs = model.predict(dtest)
df_test['pred_away_prob'] = probs[:, 0]
df_test['pred_draw_prob'] = probs[:, 1]
df_test['pred_home_prob'] = probs[:, 2]
df_test['predicted'] = np.argmax(probs, axis=1) - 1
df_test['confidence'] = probs.max(axis=1)
df_test['actual'] = np.sign(df_test['home_score'] - df_test['away_score']).astype(int)

# Filter for Confidence > 60%
high_conf = df_test[df_test['confidence'] > 0.60].copy()

# Get odds and results
results = []
for _, row in high_conf.iterrows():
    pred = row['predicted']
    actual = row['actual']
    if pred == 1:
        odds = row.get('odds_1x2_home', np.nan)
        bet_type = 'Home'
    elif pred == 0:
        odds = row.get('odds_1x2_draw', np.nan)
        bet_type = 'Draw'
    else:
        odds = row.get('odds_1x2_away', np.nan)
        bet_type = 'Away'
    
    if pd.isna(odds) or odds <= 0:
        continue
    
    won = pred == actual
    profit = (odds - 1) if won else -1
    
    results.append({
        'date': row['date'],
        'league': row['league'],
        'home_team': row['home_team'],
        'away_team': row['away_team'],
        'bet_type': bet_type,
        'odds': odds,
        'confidence': row['confidence'],
        'won': won,
        'profit': profit,
        'actual_result': {1: 'Home', 0: 'Draw', -1: 'Away'}[actual]
    })

df_r = pd.DataFrame(results)

print('='*70)
print('CONFIDENCE > 60% STRATEGY - DETAILED ANALYSIS')
print('='*70)
print(f'Total bets: {len(df_r)}')
win_rate = df_r['won'].mean() * 100
print(f'Win rate: {win_rate:.1f}%')
total_profit = df_r['profit'].sum()
print(f'Total profit: ${total_profit:+.2f}')
roi = total_profit / len(df_r) * 100
print(f'ROI: {roi:+.1f}%')

print('\n' + '='*70)
print('BY BET TYPE')
print('='*70)
for bt in ['Home', 'Draw', 'Away']:
    sub = df_r[df_r['bet_type'] == bt]
    if len(sub) > 0:
        wr = sub['won'].mean() * 100
        avg_odds = sub['odds'].mean()
        profit = sub['profit'].sum()
        print(f'{bt:6}: {len(sub):3} bets | Win: {wr:5.1f}% | Avg Odds: {avg_odds:.2f} | Profit: ${profit:+.2f}')
    else:
        print(f'{bt:6}:   0 bets')

print('\n' + '='*70)
print('BY LEAGUE')
print('='*70)
for league in sorted(df_r['league'].unique()):
    sub = df_r[df_r['league'] == league]
    wr = sub['won'].mean() * 100
    profit = sub['profit'].sum()
    print(f'{league:15}: {len(sub):3} bets | Win: {wr:5.1f}% | Profit: ${profit:+.2f}')

print('\n' + '='*70)
print('BY ODDS RANGE')
print('='*70)
ranges = [(1.0, 1.5), (1.5, 2.0), (2.0, 2.5), (2.5, 3.0), (3.0, 10.0)]
for lo, hi in ranges:
    sub = df_r[(df_r['odds'] >= lo) & (df_r['odds'] < hi)]
    if len(sub) > 0:
        wr = sub['won'].mean() * 100
        avg = sub['odds'].mean()
        profit = sub['profit'].sum()
        print(f'Odds {lo:.1f}-{hi:.1f}: {len(sub):3} bets | Win: {wr:5.1f}% | Avg: {avg:.2f} | Profit: ${profit:+.2f}')

print('\n' + '='*70)
print('SAMPLE BETS (Last 15)')
print('='*70)
sample = df_r.tail(15).copy()
sample['date'] = sample['date'].dt.strftime('%Y-%m-%d')
sample['result'] = sample['won'].map({True: 'WIN', False: 'LOSS'})
sample['profit_str'] = sample['profit'].map(lambda x: f'${x:+.2f}')
print(sample[['date', 'league', 'home_team', 'away_team', 'bet_type', 'odds', 'result', 'profit_str']].to_string(index=False))
