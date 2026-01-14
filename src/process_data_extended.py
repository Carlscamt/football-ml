#!/usr/bin/env python3
"""
Football Match Prediction - ENHANCED Data Processing Pipeline
==============================================================

This ENHANCED version processes ALL ~300 statistical features with:
- Intelligent mixed-format cleaning (percentages, fractions, numbers)
- Vectorized rolling averages for ALL features
- Strict time-leakage prevention
- Comprehensive data quality reporting

Author: Claude (Anthropic)
Version: 2.0.0 (Extended)

Usage:
    python process_data_extended.py --input raw.csv --output processed_extended.csv
"""

import argparse
import time
import re
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION
# =============================================================================

ROLLING_WINDOWS = [5, 10]  # Match history windows

# Columns to EXCLUDE from rolling (keep as-is or handle separately)
EXCLUDE_PREFIXES = ['et1_', 'et2_']  # Extra time - unreliable, skip entirely
PRESERVE_COLS = ['odds_', 'streak_', 'high_claims_']  # Keep but don't roll

# Identifier columns
ID_COLUMNS = ['match_id', 'date', 'timestamp', 'home_team_id', 'away_team_id', 
              'home_team', 'away_team', 'tournament_id', 'league']

# Odds columns
ODDS_COLUMNS = ['odds_1x2_home', 'odds_1x2_draw', 'odds_1x2_away', 
                'odds_btts_yes', 'odds_btts_no']


# =============================================================================
# DATA CLEANING FUNCTIONS
# =============================================================================

def clean_feature_value(value) -> Optional[float]:
    """
    Convert mixed formats to single numeric value.
    
    WHY: Raw data contains multiple formats that need standardization:
    
    Handles:
    - "34%"           → 34.0     (percentage string)
    - "43/70 (61%)"   → 61.0     (fraction with percentage - extract %)
    - "127/158 (80%)" → 80.0     (fraction with percentage - extract %)
    - 1.43            → 1.43     (already numeric float)
    - 66              → 66.0     (already numeric int)
    - ""              → NaN      (empty string)
    - None/NaN        → NaN      (null values)
    
    WHY extract percentage from fractions?
    - The percentage is the meaningful stat (e.g., 61% pass accuracy)
    - Raw counts (43/70) vary by match duration, percentage normalizes
    
    Returns:
        float value or NaN
    """
    # Handle null/empty values
    if pd.isna(value):
        return np.nan
    
    # Convert to string for parsing
    value_str = str(value).strip()
    
    if value_str == '' or value_str.lower() == 'none' or value_str.lower() == 'nan':
        return np.nan
    
    # -------------------------------------------------------------------------
    # Case 1: Fraction with percentage format "43/70 (61%)"
    # WHY extract percentage: The % is the normalized, comparable value
    # -------------------------------------------------------------------------
    if '(' in value_str and '%' in value_str:
        try:
            # Extract the percentage inside parentheses: "43/70 (61%)" → "61"
            pct_match = re.search(r'\((\d+(?:\.\d+)?)\s*%\)', value_str)
            if pct_match:
                return float(pct_match.group(1))
        except:
            pass
        return np.nan
    
    # -------------------------------------------------------------------------
    # Case 2: Simple percentage format "34%"
    # WHY: Remove % symbol, keep the numeric value
    # -------------------------------------------------------------------------
    if value_str.endswith('%'):
        try:
            return float(value_str.rstrip('%').strip())
        except:
            return np.nan
    
    # -------------------------------------------------------------------------
    # Case 3: Plain number (int or float)
    # WHY: Already in correct format, just ensure it's a float
    # -------------------------------------------------------------------------
    try:
        return float(value_str)
    except:
        return np.nan


def clean_all_features(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """
    Apply cleaning to all feature columns.
    
    WHY: Standardizes all columns to numeric format for rolling calculations.
    
    Args:
        df: DataFrame with raw data
        feature_cols: List of columns to clean
        
    Returns:
        DataFrame with cleaned numeric columns
    """
    df_clean = df.copy()
    cleaned_count = 0
    
    for col in feature_cols:
        if col not in df_clean.columns:
            continue
            
        original_dtype = df_clean[col].dtype
        
        # Only clean if not already numeric
        if original_dtype == 'object' or df_clean[col].astype(str).str.contains('%', na=False).any():
            df_clean[col] = df_clean[col].apply(clean_feature_value)
            cleaned_count += 1
        else:
            # Already numeric, just ensure float type
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    return df_clean, cleaned_count


# =============================================================================
# FEATURE IDENTIFICATION
# =============================================================================

def identify_feature_columns(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Categorize columns into feature types.
    
    WHY: We need to know which columns to roll, which to preserve, which to skip.
    
    Returns:
        Dictionary with column categories
    """
    all_cols = df.columns.tolist()
    
    categories = {
        'identifiers': [],      # match_id, date, team names, etc.
        'target_cols': [],      # home_score, away_score
        'odds': [],             # Pre-match odds (preserve, don't roll)
        'streaks': [],          # Streak data (preserve, don't roll)
        'extra_time': [],       # et1_, et2_ (EXCLUDE entirely)
        'full_match': [],       # Main stats (ROLL these)
        'first_half': [],       # 1st_ prefix (ROLL these)
        'second_half': [],      # 2nd_ prefix (ROLL these)
        'other': []             # Anything else
    }
    
    for col in all_cols:
        if col in ID_COLUMNS:
            categories['identifiers'].append(col)
        elif col in ['home_score', 'away_score', 'target']:
            categories['target_cols'].append(col)
        elif col.startswith('odds_'):
            categories['odds'].append(col)
        elif col.startswith('streak_') or col.startswith('high_claims_'):
            categories['streaks'].append(col)
        elif col.startswith('et1_') or col.startswith('et2_'):
            # EXCLUDED: Extra time stats don't exist for most matches
            categories['extra_time'].append(col)
        elif col.startswith('1st_'):
            categories['first_half'].append(col)
        elif col.startswith('2nd_'):
            categories['second_half'].append(col)
        else:
            # Full match stat (no prefix)
            categories['full_match'].append(col)
    
    return categories


# =============================================================================
# ROLLING FEATURES (VECTORIZED, NO LEAKAGE)
# =============================================================================

def create_rolling_features_vectorized(
    df: pd.DataFrame, 
    feature_cols: List[str],
    windows: List[int] = [5, 10]
) -> pd.DataFrame:
    """
    Create rolling averages for ALL features using vectorized operations.
    
    HOW LEAKAGE IS PREVENTED:
    =========================
    1. Data is sorted by date BEFORE processing
    2. Rolling calculation uses .shift(1) AFTER the rolling mean
    3. This ensures the current match's values are NEVER included
    
    The order is CRITICAL:
        .rolling(window).mean().shift(1)  ✓ CORRECT
        .shift(1).rolling(window).mean()  ✗ WRONG (would include current)
    
    WHY shift(1)?
    - shift(1) moves all values down by 1 position
    - The current row gets the value from the row ABOVE it
    - This means the current match's data is excluded from its own features
    
    Args:
        df: DataFrame sorted by date with cleaned numeric features
        feature_cols: Columns to create rolling features for
        windows: Rolling window sizes
        
    Returns:
        DataFrame with rolling features only
    """
    # Ensure sorted by date (critical for time-series)
    df_sorted = df.sort_values('date').reset_index(drop=True)
    
    rolling_data = {}
    total_features = len(feature_cols) * len(windows) * 2  # *2 for home/away
    created = 0
    
    for feature in feature_cols:
        if feature not in df_sorted.columns:
            continue
        
        for window in windows:
            # -----------------------------------------------------------------
            # HOME TEAM ROLLING
            # -----------------------------------------------------------------
            # For each match, calculate the home team's average in their
            # PREVIOUS matches (as home team in this tournament)
            #
            # WHY groupby('home_team_id')?
            # - Each team has their own history
            # - We want: "What was Team X's average in their last N HOME games?"
            #
            # WHY .shift(1)?
            # - After calculating rolling mean, shift down 1 position
            # - This excludes the CURRENT match from its own calculation
            # - Example: Match 5's feature = mean of matches 1-4 (not 1-5)
            # -----------------------------------------------------------------
            home_col_name = f'home_rolling_{window}_{feature}'
            
            rolling_data[home_col_name] = (
                df_sorted.groupby('home_team_id')[feature]
                .transform(lambda x: x.rolling(window=window, min_periods=1).mean().shift(1))
            )
            
            # -----------------------------------------------------------------
            # AWAY TEAM ROLLING
            # -----------------------------------------------------------------
            # Same logic but for away team's previous away games
            # -----------------------------------------------------------------
            away_col_name = f'away_rolling_{window}_{feature}'
            
            rolling_data[away_col_name] = (
                df_sorted.groupby('away_team_id')[feature]
                .transform(lambda x: x.rolling(window=window, min_periods=1).mean().shift(1))
            )
            
            created += 2
    
    return pd.DataFrame(rolling_data, index=df_sorted.index)


# =============================================================================
# H2H FEATURES (FROM PREVIOUS SCRIPT)
# =============================================================================

def compute_h2h_features(
    df: pd.DataFrame,
    home_team_id: int,
    away_team_id: int,
    match_date: str
) -> Dict[str, float]:
    """
    Compute head-to-head statistics using ONLY matches BEFORE the given date.
    
    HOW LEAKAGE IS PREVENTED:
    - Filter: date < match_date (strictly before)
    - The current match is NEVER included
    """
    # Get all PAST H2H matches
    h2h_matches = df[
        (
            ((df['home_team_id'] == home_team_id) & (df['away_team_id'] == away_team_id)) |
            ((df['home_team_id'] == away_team_id) & (df['away_team_id'] == home_team_id))
        ) &
        (df['date'] < match_date)
    ]
    
    total = len(h2h_matches)
    
    if total == 0:
        return {
            'h2h_total': 0,
            'h2h_home_wins': 0,
            'h2h_away_wins': 0,
            'h2h_draws': 0,
            'h2h_home_win_pct': 0.5,
            'h2h_away_win_pct': 0.5,
            'h2h_draw_pct': 0.0
        }
    
    home_wins = away_wins = draws = 0
    
    for _, match in h2h_matches.iterrows():
        if match['home_score'] > match['away_score']:
            if match['home_team_id'] == home_team_id:
                home_wins += 1
            else:
                away_wins += 1
        elif match['home_score'] < match['away_score']:
            if match['away_team_id'] == home_team_id:
                home_wins += 1
            else:
                away_wins += 1
        else:
            draws += 1
    
    return {
        'h2h_total': total,
        'h2h_home_wins': home_wins,
        'h2h_away_wins': away_wins,
        'h2h_draws': draws,
        'h2h_home_win_pct': home_wins / total,
        'h2h_away_win_pct': away_wins / total,
        'h2h_draw_pct': draws / total
    }


# =============================================================================
# ODDS FEATURES
# =============================================================================

def compute_odds_features(row: pd.Series) -> Dict[str, float]:
    """
    Compute derived odds features.
    
    WHY safe to use?
    - Pre-match odds are set BEFORE the match
    - They don't leak any information about the match outcome
    - They represent bookmaker's assessment of probabilities
    """
    features = {}
    
    # Keep original odds
    for col in ODDS_COLUMNS:
        if col in row.index:
            features[col] = row[col]
    
    # Implied probabilities
    for outcome in ['home', 'draw', 'away']:
        odds_col = f'odds_1x2_{outcome}'
        if pd.notna(row.get(odds_col)) and row[odds_col] > 0:
            features[f'implied_prob_{outcome}'] = 1 / row[odds_col]
            features[f'log_odds_{outcome}'] = np.log(row[odds_col])
        else:
            features[f'implied_prob_{outcome}'] = np.nan
            features[f'log_odds_{outcome}'] = np.nan
    
    # Overround (bookmaker margin)
    probs = [features.get(f'implied_prob_{x}', np.nan) for x in ['home', 'draw', 'away']]
    if all(pd.notna(p) for p in probs):
        features['overround'] = sum(probs) - 1
    else:
        features['overround'] = np.nan
    
    return features


# =============================================================================
# MAIN PROCESSING PIPELINE
# =============================================================================

def process_data_extended(input_path: str, output_path: str) -> pd.DataFrame:
    """
    ENHANCED data processing pipeline with ALL features.
    
    Transforms raw match data into ML-ready features with NO time leakage.
    """
    print("=" * 70)
    print("ENHANCED FOOTBALL ML DATA PROCESSING PIPELINE")
    print("=" * 70)
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print()
    
    start_time = time.time()
    
    # =========================================================================
    # STEP 1: Load Data
    # =========================================================================
    print("[1/7] Loading raw data...")
    df_raw = pd.read_csv(input_path)
    print(f"  Loaded: {len(df_raw):,} matches, {len(df_raw.columns)} columns")
    
    # Parse and sort dates (CRITICAL for time-series)
    df_raw['date'] = pd.to_datetime(df_raw['date']).dt.strftime('%Y-%m-%d')
    df_raw = df_raw.sort_values('date').reset_index(drop=True)
    print(f"  Date range: {df_raw['date'].min()} to {df_raw['date'].max()}")
    print()
    
    # =========================================================================
    # STEP 2: Identify Column Categories
    # =========================================================================
    print("[2/7] Categorizing columns...")
    categories = identify_feature_columns(df_raw)
    
    print(f"  Identifiers:   {len(categories['identifiers'])}")
    print(f"  Full match:    {len(categories['full_match'])}")
    print(f"  First half:    {len(categories['first_half'])}")
    print(f"  Second half:   {len(categories['second_half'])}")
    print(f"  Odds:          {len(categories['odds'])}")
    print(f"  Streaks:       {len(categories['streaks'])}")
    print(f"  Extra time:    {len(categories['extra_time'])} (EXCLUDED)")
    
    # Features to roll = full match + 1st half + 2nd half
    features_to_roll = (
        categories['full_match'] + 
        categories['first_half'] + 
        categories['second_half']
    )
    # Remove score columns from rolling
    features_to_roll = [c for c in features_to_roll if c not in ['home_score', 'away_score']]
    
    print(f"\n  Total features for rolling: {len(features_to_roll)}")
    print()
    
    # =========================================================================
    # STEP 3: Create Target Variable
    # =========================================================================
    print("[3/7] Creating target variable...")
    df_raw['target'] = np.sign(df_raw['home_score'] - df_raw['away_score']).astype(int)
    
    target_dist = df_raw['target'].value_counts().sort_index()
    print(f"  Home wins (1):  {target_dist.get(1, 0):,} ({target_dist.get(1, 0)/len(df_raw)*100:.1f}%)")
    print(f"  Draws (0):      {target_dist.get(0, 0):,} ({target_dist.get(0, 0)/len(df_raw)*100:.1f}%)")
    print(f"  Away wins (-1): {target_dist.get(-1, 0):,} ({target_dist.get(-1, 0)/len(df_raw)*100:.1f}%)")
    print()
    
    # =========================================================================
    # STEP 4: Clean All Feature Columns
    # =========================================================================
    print("[4/7] Cleaning mixed formats...")
    null_before = df_raw[features_to_roll].isnull().sum().sum()
    
    df_clean, cleaned_count = clean_all_features(df_raw, features_to_roll)
    
    null_after = df_clean[features_to_roll].isnull().sum().sum()
    print(f"  Columns cleaned: {cleaned_count}")
    print(f"  Nulls before cleaning: {null_before:,}")
    print(f"  Nulls after cleaning:  {null_after:,}")
    print()
    
    # =========================================================================
    # STEP 5: Create Rolling Features (VECTORIZED)
    # =========================================================================
    print("[5/7] Creating rolling features (vectorized)...")
    print(f"  Features: {len(features_to_roll)}")
    print(f"  Windows:  {ROLLING_WINDOWS}")
    print(f"  Expected columns: {len(features_to_roll) * len(ROLLING_WINDOWS) * 2}")
    
    rolling_start = time.time()
    rolling_df = create_rolling_features_vectorized(df_clean, features_to_roll, ROLLING_WINDOWS)
    rolling_time = time.time() - rolling_start
    
    print(f"  Created: {len(rolling_df.columns)} rolling features in {rolling_time:.1f}s")
    print()
    
    # =========================================================================
    # STEP 6: Compute H2H Features
    # =========================================================================
    print("[6/7] Computing H2H features...")
    h2h_list = []
    n_matches = len(df_clean)
    
    for idx, row in df_clean.iterrows():
        if (idx + 1) % 200 == 0 or idx == n_matches - 1:
            pct = (idx + 1) / n_matches * 100
            print(f"\r  Progress: {idx + 1}/{n_matches} ({pct:.1f}%)", end='', flush=True)
        
        h2h = compute_h2h_features(
            df_clean, row['home_team_id'], row['away_team_id'], row['date']
        )
        h2h_list.append(h2h)
    
    df_h2h = pd.DataFrame(h2h_list)
    print()
    print()
    
    # =========================================================================
    # STEP 7: Compute Odds Features & Assemble Final Dataset
    # =========================================================================
    print("[7/7] Assembling final dataset...")
    
    # Odds features
    odds_list = [compute_odds_features(row) for _, row in df_clean.iterrows()]
    df_odds = pd.DataFrame(odds_list)
    
    # Get streak columns if they exist
    streak_cols = [c for c in categories['streaks'] if c in df_clean.columns]
    
    # Assemble final dataset
    # Order: Identifiers -> Target -> Scores -> Odds -> H2H -> Streaks -> Rolling
    df_final = pd.concat([
        df_clean[ID_COLUMNS].reset_index(drop=True),
        df_clean[['target', 'home_score', 'away_score']].reset_index(drop=True),
        df_odds.reset_index(drop=True),
        df_h2h.reset_index(drop=True),
        df_clean[streak_cols].reset_index(drop=True) if streak_cols else pd.DataFrame(),
        rolling_df.reset_index(drop=True)
    ], axis=1)
    
    # Handle remaining nulls
    # WHERE nulls come from:
    # 1. First few matches per team (no rolling history) - fill with 0 or team average
    # 2. Missing odds (~1% of matches) - keep as NaN for now
    # 3. Some stats missing for certain matches - forward fill
    
    null_before_fill = df_final.isnull().sum().sum()
    
    # Fill rolling features with 0 (teams start with no history)
    rolling_cols = [c for c in df_final.columns if 'rolling' in c]
    df_final[rolling_cols] = df_final[rolling_cols].fillna(0)
    
    # Fill H2H with neutral values (already set in function, but just in case)
    h2h_cols = [c for c in df_final.columns if 'h2h' in c]
    df_final[h2h_cols] = df_final[h2h_cols].fillna(0)
    
    null_after_fill = df_final.isnull().sum().sum()
    
    print(f"  Nulls before filling: {null_before_fill:,}")
    print(f"  Nulls after filling:  {null_after_fill:,}")
    
    # Save
    df_final.to_csv(output_path, index=False)
    print(f"\n  Saved: {output_path}")
    
    # =========================================================================
    # DATA QUALITY REPORT
    # =========================================================================
    elapsed = time.time() - start_time
    
    print()
    print("=" * 70)
    print("ENHANCED DATA PROCESSING REPORT")
    print("=" * 70)
    print()
    
    print("Pipeline Statistics:")
    print(f"  Processing time:  {elapsed:.1f} seconds")
    print(f"  Raw columns:      {len(df_raw.columns)}")
    print(f"  Features rolled:  {len(features_to_roll)}")
    print(f"  Rolling features: {len(rolling_cols)}")
    print(f"  H2H features:     {len(h2h_cols)}")
    print()
    
    print("Final Shape:")
    print(f"  Rows:    {len(df_final):,}")
    print(f"  Columns: {len(df_final.columns)}")
    print()
    
    print("Null Values Remaining:")
    null_summary = df_final.isnull().sum()
    cols_with_nulls = null_summary[null_summary > 0].sort_values(ascending=False)
    if len(cols_with_nulls) > 0:
        for col in cols_with_nulls.head(5).index:
            print(f"  {col}: {null_summary[col]} ({null_summary[col]/len(df_final)*100:.1f}%)")
        if len(cols_with_nulls) > 5:
            print(f"  ... and {len(cols_with_nulls) - 5} more")
    else:
        print("  None!")
    print()
    
    print("Sample Rolling Feature Ranges:")
    sample_features = [
        'home_rolling_5_ball_possession_home',
        'home_rolling_5_expected_goals_home', 
        'away_rolling_10_shots_on_target_away',
        'home_rolling_5_passes_home'
    ]
    for feat in sample_features:
        if feat in df_final.columns:
            col_data = df_final[feat].dropna()
            if len(col_data) > 0:
                print(f"  {feat}:")
                print(f"    Range: {col_data.min():.2f} - {col_data.max():.2f}")
                print(f"    Mean:  {col_data.mean():.2f}")
    print()
    
    print("Matches per League:")
    print(df_final['league'].value_counts().to_string())
    print()
    
    print("Date Range:")
    print(f"  Start: {df_final['date'].min()}")
    print(f"  End:   {df_final['date'].max()}")
    print()
    
    print("=" * 70)
    print("PIPELINE COMPLETE!")
    print("=" * 70)
    
    return df_final


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

def find_latest_raw_file():
    """Find the most recent CSV file in data/raw/"""
    raw_dir = 'data/raw'
    if not os.path.exists(raw_dir):
        return None
    files = [f for f in os.listdir(raw_dir) if f.endswith('.csv')]
    if not files:
        return None
    # Sort by filename (which includes date) and return newest
    return os.path.join(raw_dir, sorted(files)[-1])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Enhanced Football ML Data Pipeline (ALL Features)'
    )
    
    # Auto-find latest file
    default_input = find_latest_raw_file()
    if default_input is None:
        default_input = 'data/raw/sofascore_data.csv'
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        default=default_input,
        help='Input CSV path (default: latest file in data/raw/)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='data/processed/processed_features_extended.csv',
        help='Output CSV path'
    )
    
    args = parser.parse_args()
    
    print(f"Using input file: {args.input}")
    df = process_data_extended(args.input, args.output)
