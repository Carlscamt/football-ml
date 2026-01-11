import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Config
DATA_PATH = 'data/processed/processed_features_extended.csv'
TOP_N_FEATURES = 200

def load_and_split():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # Identify cols
    exclude_cols = ['match_id', 'date', 'timestamp', 'home_team_id', 'away_team_id',
                    'home_team', 'away_team', 'tournament_id', 'league', 
                    'target', 'home_score', 'away_score']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    # Fill nulls
    df[feature_cols] = df[feature_cols].fillna(0)
    
    # 80/20 Split (Temporal)
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    
    print(f"Train: {len(train_df)}, Test: {len(test_df)}")
    
    return train_df, test_df, feature_cols

def select_features(X_train, y_train, feature_cols, top_n=200):
    print("Selecting features...")
    # Quick model for importance
    label_map = {-1:0, 0:1, 1:2}
    y_enc = np.array([label_map[y] for y in y_train])
    
    dtrain = xgb.DMatrix(X_train, label=y_enc, feature_names=feature_cols)
    params = {
        'max_depth': 4, 'learning_rate': 0.1, 'objective': 'multi:softprob', 
        'num_class': 3, 'verbosity': 0
    }
    model = xgb.train(params, dtrain, num_boost_round=50)
    
    scores = model.get_score(importance_type='weight')
    sorted_feats = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    # Map back to names (XGBoost returns f0, f1...)
    # Actually get_score returns feature names if provided in DMatrix
    # But sometimes it returns f0 if feature_names arg is tricky. 
    # Let's assume it returns names since we passed feature_names.
    
    selected = [f for f, s in sorted_feats[:top_n]]
    # Ensure we fill up to top_n if fewer satisfied
    if len(selected) < top_n:
        remaining = [f for f in feature_cols if f not in selected]
        selected.extend(remaining[:top_n - len(selected)])
        
    return selected

def train_model(train_df, test_df, features):
    print("Training XGBoost (v2 params)...")
    X_train = train_df[features].values
    y_train = train_df['target'].values
    X_test = test_df[features].values
    y_test = test_df['target'].values
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Encode
    label_map = {-1:0, 0:1, 1:2}
    y_train_enc = np.array([label_map[y] for y in y_train])
    y_test_enc = np.array([label_map[y] for y in y_test])
    
    # Weights
    weights = compute_sample_weight('balanced', y_train_enc)
    
    dtrain = xgb.DMatrix(X_train_scaled, label=y_train_enc, weight=weights)
    dtest = xgb.DMatrix(X_test_scaled, label=y_test_enc)
    
    params = {
        'max_depth': 4,
        'learning_rate': 0.01,
        'objective': 'multi:softprob',
        'num_class': 3,
        'eval_metric': 'mlogloss',
        'subsample': 0.5,
        'colsample_bytree': 0.3,
        'min_child_weight': 10,
        'reg_alpha': 1.0,
        'reg_lambda': 10.0,
        'random_state': 42,
        'verbosity': 0
    }
    
    model = xgb.train(params, dtrain, num_boost_round=500, 
                      evals=[(dtrain,'train'),(dtest,'test')],
                      early_stopping_rounds=50, verbose_eval=False)
    
    preds = model.predict(dtest)
    pred_labels_enc = np.argmax(preds, axis=1)
    rev_map = {0:-1, 1:0, 2:1}
    pred_labels = np.array([rev_map[x] for x in pred_labels_enc])
    
    acc = accuracy_score(y_test, pred_labels)
    print(f"Test Accuracy: {acc:.2%}")
    
    return pred_labels, preds

def analyze_odds(test_df, pred_labels, pred_probs):
    print("\n" + "="*60)
    print("ODDS ANALYSIS (Test Set)")
    print("="*60)
    
    df = test_df.copy()
    df['pred'] = pred_labels
    
    # Determine predicted probability for the predicted class
    # pred_probs is (N, 3) for [-1, 0, 1] mapped to [0, 1, 2]
    # We need to map back carefully. pred_probs[:,0] -> -1, etc.
    # Actually checking label_map: {-1:0, 0:1, 1:2}
    
    max_probs = np.max(pred_probs, axis=1)
    df['pred_conf'] = max_probs
    
    # Add correctness
    df['correct'] = (df['pred'] == df['target'])
    
    # Define odds bins
    bins = [1.0, 1.5, 2.0, 3.0, 5.0, 100.0]
    labels = ['1.0-1.5', '1.5-2.0', '2.0-3.0', '3.0-5.0', '5.0+']
    
    # We want to analyze based on the Odds of the PREDICTED outcome?
    # Or based on the "Favorite" status?
    # Usually: "When predicted outcome odds are X, accuracy is Y"
    
    # Get odds for predicted outcome
    odds_col = []
    for idx, row in df.iterrows():
        p = row['pred']
        if p == 1: col = 'odds_1x2_home'
        elif p == 0: col = 'odds_1x2_draw'
        else: col = 'odds_1x2_away'
        
        if col in row and pd.notna(row[col]):
            odds_col.append(row[col])
        else:
            odds_col.append(np.nan)
    
    df['pred_odds'] = odds_col
    df['odds_bin'] = pd.cut(df['pred_odds'], bins=bins, labels=labels)
    
    print("\nAccuracy by Predicted Odds Range:")
    print(f"{'Range':<10} | {'Count':<6} | {'Acc':<7} | {'Roi (flat)':<10}")
    print("-" * 45)
    
    for label in labels:
        subset = df[df['odds_bin'] == label]
        if len(subset) == 0:
            continue
            
        acc = subset['correct'].mean()
        count = len(subset)
        
        # Calculate ROI if we bet 1 unit on every prediction in this range
        # Profit = (Odds - 1) for correct, -1 for incorrect
        # Net = sum(correct * (odds-1) - incorrect)
        #     = sum(correct * odds) - total_bets
        
        total_bets = len(subset)
        total_return = subset[subset['correct']]['pred_odds'].sum()
        roi = (total_return - total_bets) / total_bets
        
        print(f"{label:<10} | {count:<6} | {acc:<7.1%} | {roi:<+10.1%}")

    # Also analyze by Confidence
    print("\nAccuracy by Model Confidence:")
    conf_bins = [0.33, 0.4, 0.5, 0.6, 0.7, 1.0]
    conf_labels = ['33-40%', '40-50%', '50-60%', '60-70%', '70%+']
    df['conf_bin'] = pd.cut(df['pred_conf'], bins=conf_bins, labels=conf_labels)
    
    print(f"{'Conf':<10} | {'Count':<6} | {'Acc':<7} | {'ROI':<10}")
    print("-" * 45)
    for label in conf_labels:
        subset = df[df['conf_bin'] == label]
        if len(subset) == 0: continue
        
        acc = subset['correct'].mean()
        count = len(subset)
        total_return = subset[subset['correct']]['pred_odds'].sum()
        roi = (total_return - len(subset)) / len(subset)
        print(f"{label:<10} | {count:<6} | {acc:<7.1%} | {roi:<+10.1%}")

if __name__ == '__main__':
    train, test, feats = load_and_split()
    sel_feats = select_features(train[feats].values, train['target'].values, feats)
    preds, probs = train_model(train, test, sel_feats)
    analyze_odds(test, preds, probs)
