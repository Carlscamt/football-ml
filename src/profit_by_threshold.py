import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

# Load model
model = xgb.Booster()
model.load_model('models/xgboost_model_v2.json')

with open('results/selected_features.txt', 'r') as f:
    features = [line.strip() for line in f if line.strip()]

df = pd.read_csv('data/processed/processed_features_extended.csv')
df['date'] = pd.to_datetime(df['date'])

# Scaler
df_train = df[df['date'] < '2025-01-01']
scaler = StandardScaler()
scaler.fit(df_train[features].fillna(0))

# Test set
df_test = df[df['date'] >= '2025-01-01'].copy()
X_test = df_test[features].fillna(0)
X_test_scaled = scaler.transform(X_test)
dtest = xgb.DMatrix(X_test_scaled, feature_names=features)

probs = model.predict(dtest)
df_test['pred_away'] = probs[:, 0]
df_test['pred_draw'] = probs[:, 1]
df_test['pred_home'] = probs[:, 2]
df_test['predicted'] = np.argmax(probs, axis=1) - 1
df_test['confidence'] = probs.max(axis=1)
df_test['actual'] = np.sign(df_test['home_score'] - df_test['away_score']).astype(int)

print('='*80)
print('PROFITABILITY BY CONFIDENCE THRESHOLD (1 Unit Flat Betting)')
print('='*80)
print(f"{'Threshold':>10} | {'Bets':>6} | {'Wins':>6} | {'Win%':>6} | {'Profit':>10} | {'ROI':>8}")
print('-'*80)

thresholds = [0.34, 0.36, 0.38, 0.40, 0.42, 0.44, 0.46, 0.48, 0.50, 0.52, 0.54, 0.56, 0.58, 0.60]

results = []
for thresh in thresholds:
    subset = df_test[df_test['confidence'] >= thresh]
    
    profit = 0
    wins = 0
    total_bets = 0
    
    for _, row in subset.iterrows():
        pred = row['predicted']
        actual = row['actual']
        
        if pred == 1:
            odds = row.get('odds_1x2_home', np.nan)
        elif pred == 0:
            odds = row.get('odds_1x2_draw', np.nan)
        else:
            odds = row.get('odds_1x2_away', np.nan)
        
        if pd.isna(odds) or odds <= 0:
            continue
        
        total_bets += 1
        if pred == actual:
            profit += (odds - 1)
            wins += 1
        else:
            profit -= 1
    
    if total_bets > 0:
        win_rate = wins / total_bets * 100
        roi = profit / total_bets * 100
        results.append({'thresh': thresh, 'bets': total_bets, 'wins': wins, 'win_rate': win_rate, 'profit': profit, 'roi': roi})
        
        status = '[PROFIT]' if roi > 0 else '[LOSS]'
        print(f"{thresh*100:>9.0f}% | {total_bets:>6} | {wins:>6} | {win_rate:>5.1f}% | ${profit:>+9.2f} | {roi:>+7.1f}% {status}")

print('='*80)

# Find breakeven point
profitable = [r for r in results if r['roi'] > 0]
if profitable:
    lowest_profitable = min(profitable, key=lambda x: x['thresh'])
    print('')
    print(f"LOWEST PROFITABLE THRESHOLD: {lowest_profitable['thresh']*100:.0f}%")
    print(f"  Bets: {lowest_profitable['bets']}")
    print(f"  Win Rate: {lowest_profitable['win_rate']:.1f}%")
    print(f"  ROI: {lowest_profitable['roi']:+.1f}%")
    print(f"  Total Profit: ${lowest_profitable['profit']:+.2f}")
else:
    print('No profitable thresholds found!')

# Best ROI
best = max(results, key=lambda x: x['roi'])
print('')
print(f"BEST ROI THRESHOLD: {best['thresh']*100:.0f}%")
print(f"  Bets: {best['bets']}")
print(f"  Win Rate: {best['win_rate']:.1f}%")
print(f"  ROI: {best['roi']:+.1f}%")
print(f"  Total Profit: ${best['profit']:+.2f}")
