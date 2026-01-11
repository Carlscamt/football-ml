#!/usr/bin/env python3
"""
Football Match Prediction - Time-Leak-Free Data Processing Pipeline
====================================================================

This script transforms raw Sofascore match data into a clean feature dataset
ready for ML modeling. All features are computed using ONLY historical data
to prevent time leakage.

Author: Claude (Anthropic)
Version: 1.0.0

Usage:
    python process_data.py --input raw_matches.csv --output processed_features.csv
"""

import argparse
import time
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION
# =============================================================================

ROLLING_WINDOWS = [5, 10]  # Match history windows

# Odds columns to keep (these are pre-match, so safe)
ODDS_COLUMNS = [
    'odds_1x2_home', 'odds_1x2_draw', 'odds_1x2_away',
    'odds_btts_yes', 'odds_btts_no'
]

# Identifier columns
ID_COLUMNS = ['match_id', 'date', 'timestamp', 'home_team_id', 'away_team_id', 
              'home_team', 'away_team', 'tournament_id', 'league']


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def print_progress(current: int, total: int, prefix: str = '', width: int = 40):
    """Display a progress bar."""
    pct = current / total
    filled = int(width * pct)
    bar = '#' * filled + '-' * (width - filled)
    print(f'\r{prefix} [{bar}] {current}/{total} ({pct*100:.1f}%)', end='', flush=True)
    if current == total:
        print()


def create_target(home_score: int, away_score: int) -> int:
    """
    Create target variable from scores.
    
    Returns:
        1  = Home win
        0  = Draw
        -1 = Away win
    """
    if home_score > away_score:
        return 1
    elif home_score < away_score:
        return -1
    else:
        return 0


# =============================================================================
# ROLLING FEATURES (TIME-LEAK FREE)
# =============================================================================

def compute_team_rolling_features(
    df: pd.DataFrame,
    team_id: int,
    match_date: str,
    windows: List[int] = [5, 10]
) -> Dict[str, float]:
    """
    Compute rolling statistics for a team using ONLY matches BEFORE the given date.
    
    This is the core time-leak prevention: we filter to date < match_date
    so no future information is used.
    
    Args:
        df: Full dataframe of all matches
        team_id: The team to compute features for
        match_date: Current match date (features use data BEFORE this)
        windows: List of window sizes for rolling calculations
    
    Returns:
        Dictionary of rolling features
    """
    features = {}
    
    # Get all PAST matches for this team (CRITICAL: strictly less than)
    # Team could be home or away
    team_matches = df[
        ((df['home_team_id'] == team_id) | (df['away_team_id'] == team_id)) &
        (df['date'] < match_date)
    ].sort_values('date', ascending=False)
    
    if len(team_matches) == 0:
        # No history - return neutral/zero features
        for w in windows:
            features[f'rolling_{w}_goals_for'] = np.nan
            features[f'rolling_{w}_goals_against'] = np.nan
            features[f'rolling_{w}_wins'] = 0
            features[f'rolling_{w}_draws'] = 0
            features[f'rolling_{w}_losses'] = 0
            features[f'rolling_{w}_win_pct'] = 0.5  # Neutral
            features[f'rolling_{w}_points_per_game'] = 1.0  # Neutral (1 point = draw avg)
        return features
    
    # For each window size
    for w in windows:
        recent = team_matches.head(w)
        n_matches = len(recent)
        
        if n_matches == 0:
            features[f'rolling_{w}_goals_for'] = np.nan
            features[f'rolling_{w}_goals_against'] = np.nan
            features[f'rolling_{w}_wins'] = 0
            features[f'rolling_{w}_draws'] = 0
            features[f'rolling_{w}_losses'] = 0
            features[f'rolling_{w}_win_pct'] = 0.5
            features[f'rolling_{w}_points_per_game'] = 1.0
            continue
        
        # Calculate goals for/against based on whether team was home or away
        goals_for = 0
        goals_against = 0
        wins = 0
        draws = 0
        losses = 0
        
        for _, match in recent.iterrows():
            if match['home_team_id'] == team_id:
                # Team was HOME
                gf = match['home_score']
                ga = match['away_score']
            else:
                # Team was AWAY
                gf = match['away_score']
                ga = match['home_score']
            
            goals_for += gf
            goals_against += ga
            
            if gf > ga:
                wins += 1
            elif gf < ga:
                losses += 1
            else:
                draws += 1
        
        # Store features
        features[f'rolling_{w}_goals_for'] = goals_for / n_matches
        features[f'rolling_{w}_goals_against'] = goals_against / n_matches
        features[f'rolling_{w}_wins'] = wins
        features[f'rolling_{w}_draws'] = draws
        features[f'rolling_{w}_losses'] = losses
        features[f'rolling_{w}_win_pct'] = wins / n_matches
        features[f'rolling_{w}_points_per_game'] = (3 * wins + draws) / n_matches
    
    return features


# =============================================================================
# H2H FEATURES (TIME-LEAK FREE)
# =============================================================================

def compute_h2h_features(
    df: pd.DataFrame,
    home_team_id: int,
    away_team_id: int,
    match_date: str
) -> Dict[str, float]:
    """
    Compute head-to-head statistics using ONLY matches BEFORE the given date.
    
    Args:
        df: Full dataframe of all matches
        home_team_id: Current match home team
        away_team_id: Current match away team
        match_date: Current match date (H2H uses matches BEFORE this)
    
    Returns:
        Dictionary of H2H features
    """
    features = {}
    
    # Get all PAST H2H matches (either team could be home/away)
    h2h_matches = df[
        (
            ((df['home_team_id'] == home_team_id) & (df['away_team_id'] == away_team_id)) |
            ((df['home_team_id'] == away_team_id) & (df['away_team_id'] == home_team_id))
        ) &
        (df['date'] < match_date)  # CRITICAL: strictly before
    ]
    
    total = len(h2h_matches)
    
    if total == 0:
        # No previous encounters - return neutral
        features['h2h_total'] = 0
        features['h2h_home_wins'] = 0
        features['h2h_away_wins'] = 0
        features['h2h_draws'] = 0
        features['h2h_home_win_pct'] = 0.5  # Neutral
        features['h2h_away_win_pct'] = 0.5  # Neutral
        features['h2h_draw_pct'] = 0.0
        return features
    
    # Count results relative to CURRENT match's home/away teams
    home_wins = 0
    away_wins = 0
    draws = 0
    
    for _, match in h2h_matches.iterrows():
        # Determine who won this historical match
        if match['home_score'] > match['away_score']:
            # Home team of THAT match won
            if match['home_team_id'] == home_team_id:
                home_wins += 1  # Current home team won
            else:
                away_wins += 1  # Current away team won
        elif match['home_score'] < match['away_score']:
            # Away team of THAT match won
            if match['away_team_id'] == home_team_id:
                home_wins += 1  # Current home team won
            else:
                away_wins += 1  # Current away team won
        else:
            draws += 1
    
    features['h2h_total'] = total
    features['h2h_home_wins'] = home_wins
    features['h2h_away_wins'] = away_wins
    features['h2h_draws'] = draws
    features['h2h_home_win_pct'] = home_wins / total
    features['h2h_away_win_pct'] = away_wins / total
    features['h2h_draw_pct'] = draws / total
    
    return features


# =============================================================================
# ODDS FEATURES
# =============================================================================

def compute_odds_features(row: pd.Series) -> Dict[str, float]:
    """
    Compute derived odds features.
    
    These are PRE-MATCH odds, so they are safe to use (no leakage).
    
    Args:
        row: Single match row
    
    Returns:
        Dictionary of odds features
    """
    features = {}
    
    # Keep original odds
    for col in ODDS_COLUMNS:
        if col in row.index:
            features[col] = row[col]
    
    # Implied probabilities (1 / odds)
    if pd.notna(row.get('odds_1x2_home')):
        features['implied_prob_home'] = 1 / row['odds_1x2_home']
    else:
        features['implied_prob_home'] = np.nan
        
    if pd.notna(row.get('odds_1x2_draw')):
        features['implied_prob_draw'] = 1 / row['odds_1x2_draw']
    else:
        features['implied_prob_draw'] = np.nan
        
    if pd.notna(row.get('odds_1x2_away')):
        features['implied_prob_away'] = 1 / row['odds_1x2_away']
    else:
        features['implied_prob_away'] = np.nan
    
    # Log odds (useful for some models)
    if pd.notna(row.get('odds_1x2_home')) and row['odds_1x2_home'] > 0:
        features['log_odds_home'] = np.log(row['odds_1x2_home'])
    else:
        features['log_odds_home'] = np.nan
        
    if pd.notna(row.get('odds_1x2_draw')) and row['odds_1x2_draw'] > 0:
        features['log_odds_draw'] = np.log(row['odds_1x2_draw'])
    else:
        features['log_odds_draw'] = np.nan
        
    if pd.notna(row.get('odds_1x2_away')) and row['odds_1x2_away'] > 0:
        features['log_odds_away'] = np.log(row['odds_1x2_away'])
    else:
        features['log_odds_away'] = np.nan
    
    # Overround (bookmaker margin) - sum of implied probs - 1
    if all(pd.notna(features.get(f'implied_prob_{x}')) for x in ['home', 'draw', 'away']):
        features['overround'] = (features['implied_prob_home'] + 
                                 features['implied_prob_draw'] + 
                                 features['implied_prob_away'] - 1)
    else:
        features['overround'] = np.nan
    
    return features


# =============================================================================
# MAIN PROCESSING PIPELINE
# =============================================================================

def process_data(input_path: str, output_path: str) -> pd.DataFrame:
    """
    Main data processing pipeline.
    
    Transforms raw match data into ML-ready features with NO time leakage.
    
    Args:
        input_path: Path to raw CSV
        output_path: Path for output CSV
    
    Returns:
        Processed DataFrame
    """
    print("=" * 60)
    print("FOOTBALL ML DATA PROCESSING PIPELINE")
    print("=" * 60)
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print()
    
    start_time = time.time()
    
    # -------------------------------------------------------------------------
    # STEP 1: Load and Clean Data
    # -------------------------------------------------------------------------
    print("[1/6] Loading raw data...")
    df_raw = pd.read_csv(input_path)
    print(f"  Loaded: {len(df_raw):,} matches, {len(df_raw.columns)} columns")
    
    # Parse dates
    df_raw['date'] = pd.to_datetime(df_raw['date']).dt.strftime('%Y-%m-%d')
    
    # Sort by date (CRITICAL for time-series)
    df_raw = df_raw.sort_values('date').reset_index(drop=True)
    print(f"  Date range: {df_raw['date'].min()} to {df_raw['date'].max()}")
    print(f"  Leagues: {df_raw['league'].nunique()}")
    
    # Check for required columns
    required = ['home_team_id', 'away_team_id', 'home_score', 'away_score', 'date']
    missing = [c for c in required if c not in df_raw.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Drop rows with null scores
    null_scores = df_raw[['home_score', 'away_score']].isnull().any(axis=1).sum()
    if null_scores > 0:
        print(f"  Dropping {null_scores} rows with null scores")
        df_raw = df_raw.dropna(subset=['home_score', 'away_score'])
    
    print()
    
    # -------------------------------------------------------------------------
    # STEP 2: Create Target Variable
    # -------------------------------------------------------------------------
    print("[2/6] Creating target variable...")
    df_raw['target'] = df_raw.apply(
        lambda x: create_target(x['home_score'], x['away_score']), axis=1
    )
    target_dist = df_raw['target'].value_counts().sort_index()
    print(f"  Home wins (1):  {target_dist.get(1, 0):,} ({target_dist.get(1, 0)/len(df_raw)*100:.1f}%)")
    print(f"  Draws (0):      {target_dist.get(0, 0):,} ({target_dist.get(0, 0)/len(df_raw)*100:.1f}%)")
    print(f"  Away wins (-1): {target_dist.get(-1, 0):,} ({target_dist.get(-1, 0)/len(df_raw)*100:.1f}%)")
    print()
    
    # -------------------------------------------------------------------------
    # STEP 3: Compute Rolling Features
    # -------------------------------------------------------------------------
    print("[3/6] Computing rolling features (this may take a few minutes)...")
    
    all_features = []
    n_matches = len(df_raw)
    
    for idx, row in df_raw.iterrows():
        if idx % 100 == 0 or idx == n_matches - 1:
            print_progress(idx + 1, n_matches, '  Progress')
        
        match_features = {}
        
        # Identifiers
        for col in ID_COLUMNS:
            if col in row.index:
                match_features[col] = row[col]
        
        # Target
        match_features['target'] = row['target']
        match_features['home_score'] = row['home_score']
        match_features['away_score'] = row['away_score']
        
        # Rolling features for HOME team
        home_rolling = compute_team_rolling_features(
            df_raw, row['home_team_id'], row['date'], ROLLING_WINDOWS
        )
        for k, v in home_rolling.items():
            match_features[f'home_{k}'] = v
        
        # Rolling features for AWAY team
        away_rolling = compute_team_rolling_features(
            df_raw, row['away_team_id'], row['date'], ROLLING_WINDOWS
        )
        for k, v in away_rolling.items():
            match_features[f'away_{k}'] = v
        
        all_features.append(match_features)
    
    df_features = pd.DataFrame(all_features)
    print()
    
    # -------------------------------------------------------------------------
    # STEP 4: Compute H2H Features
    # -------------------------------------------------------------------------
    print("[4/6] Computing H2H features...")
    
    h2h_features_list = []
    for idx, row in df_raw.iterrows():
        if idx % 100 == 0 or idx == n_matches - 1:
            print_progress(idx + 1, n_matches, '  Progress')
        
        h2h = compute_h2h_features(
            df_raw, row['home_team_id'], row['away_team_id'], row['date']
        )
        h2h_features_list.append(h2h)
    
    df_h2h = pd.DataFrame(h2h_features_list)
    df_features = pd.concat([df_features, df_h2h], axis=1)
    print()
    
    # -------------------------------------------------------------------------
    # STEP 5: Compute Odds Features
    # -------------------------------------------------------------------------
    print("[5/6] Computing odds features...")
    
    odds_features_list = []
    for idx, row in df_raw.iterrows():
        odds = compute_odds_features(row)
        odds_features_list.append(odds)
    
    df_odds = pd.DataFrame(odds_features_list)
    df_features = pd.concat([df_features, df_odds], axis=1)
    
    odds_coverage = df_features['odds_1x2_home'].notna().mean() * 100
    print(f"  Odds coverage: {odds_coverage:.1f}%")
    print()
    
    # -------------------------------------------------------------------------
    # STEP 6: Final Cleanup and Export
    # -------------------------------------------------------------------------
    print("[6/6] Final cleanup and export...")
    
    # Remove duplicates
    initial_rows = len(df_features)
    df_features = df_features.drop_duplicates(subset=['match_id'], keep='first')
    if len(df_features) < initial_rows:
        print(f"  Removed {initial_rows - len(df_features)} duplicate matches")
    
    # Sort by date
    df_features = df_features.sort_values('date').reset_index(drop=True)
    
    # Save
    df_features.to_csv(output_path, index=False)
    print(f"  Saved: {output_path}")
    print()
    
    # -------------------------------------------------------------------------
    # DATA QUALITY REPORT
    # -------------------------------------------------------------------------
    elapsed = time.time() - start_time
    
    print("=" * 60)
    print("DATA QUALITY REPORT")
    print("=" * 60)
    print(f"Processing time: {elapsed:.1f} seconds")
    print()
    
    print("Dataset Shape:")
    print(f"  Rows:    {len(df_features):,}")
    print(f"  Columns: {len(df_features.columns)}")
    print()
    
    print("Null Counts:")
    null_counts = df_features.isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0].sort_values(ascending=False)
    if len(cols_with_nulls) > 0:
        for col, count in cols_with_nulls.head(10).items():
            print(f"  {col}: {count} ({count/len(df_features)*100:.1f}%)")
        if len(cols_with_nulls) > 10:
            print(f"  ... and {len(cols_with_nulls) - 10} more columns with nulls")
    else:
        print("  No null values!")
    print()
    
    print("Feature Columns:")
    feature_cols = [c for c in df_features.columns if c not in ID_COLUMNS + ['target', 'home_score', 'away_score']]
    print(f"  Total features: {len(feature_cols)}")
    print(f"  Rolling: {len([c for c in feature_cols if 'rolling' in c])}")
    print(f"  H2H: {len([c for c in feature_cols if 'h2h' in c])}")
    print(f"  Odds: {len([c for c in feature_cols if 'odds' in c or 'implied' in c or 'log_odds' in c or 'overround' in c])}")
    print()
    
    print("Target Distribution:")
    print(df_features['target'].value_counts().sort_index().to_string())
    print()
    
    print("Sample Rolling Feature Ranges:")
    sample_cols = ['home_rolling_5_goals_for', 'home_rolling_5_win_pct', 'home_rolling_10_points_per_game']
    for col in sample_cols:
        if col in df_features.columns:
            print(f"  {col}: {df_features[col].min():.2f} - {df_features[col].max():.2f} (mean: {df_features[col].mean():.2f})")
    print()
    
    print("First 5 Matches (chronological):")
    preview_cols = ['date', 'home_team', 'away_team', 'target', 'home_rolling_5_goals_for', 'h2h_total', 'odds_1x2_home']
    preview = df_features[[c for c in preview_cols if c in df_features.columns]].head()
    print(preview.to_string(index=False))
    print()
    
    print("=" * 60)
    print("PIPELINE COMPLETE!")
    print("=" * 60)
    
    return df_features


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Football ML Data Processing Pipeline (Time-Leak Free)'
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        default='data/raw/sofascore_parallel_6lg_3yr_2434matches.csv',
        help='Input CSV path'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='data/processed/processed_features.csv',
        help='Output CSV path'
    )
    
    args = parser.parse_args()
    
    df = process_data(args.input, args.output)
