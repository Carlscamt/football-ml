import pandas as pd
df = pd.read_csv('data/predictions/future_predictions_v2.csv')
print('Columns:', df.columns.tolist())
print()
if 'confidence' in df.columns:
    print('Confidence stats:')
    print(df['confidence'].describe())
    print()
    print('Matches by confidence threshold:')
    c60 = len(df[df['confidence'] >= 0.60])
    c50 = len(df[df['confidence'] >= 0.50])
    c40 = len(df[df['confidence'] >= 0.40])
    print(f'  >= 60%: {c60}')
    print(f'  >= 50%: {c50}')
    print(f'  >= 40%: {c40}')
    print(f'  Total:  {len(df)}')
    
    if c60 > 0:
        print('\nHigh confidence matches:')
        hc = df[df['confidence'] >= 0.60]
        print(hc[['home_team', 'away_team', 'predicted_result', 'confidence']].head(10))
else:
    print('No confidence column found!')
    print('Available columns:', df.columns.tolist())
