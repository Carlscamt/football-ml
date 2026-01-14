#!/usr/bin/env python3
"""
Incremental Data Updater - Local Version
Finds and scrapes NEW finished matches not in the existing dataset.
"""

import time
import random
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from datetime import datetime
import pandas as pd
import os

try:
    from tls_client import Session
except ImportError:
    print("Installing tls_client...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'tls_client', '-q'])
    from tls_client import Session


# Configuration
LEAGUES = {
    'Premier League': {'ids': [17]},
    'La Liga': {'ids': [8]},
    'Bundesliga': {'ids': [35]},
    'Serie A': {'ids': [23]},
    'Ligue 1': {'ids': [34]},
    'Liga MX': {'ids': [11621, 11620]},
}

BASE_URL = "https://www.sofascore.com/api/v1"
PAGES_TO_CHECK = 20  # Check ~1 week of matches


class SessionPool:
    def __init__(self, size=5):
        self.sessions = [Session(client_identifier="firefox_120") for _ in range(size)]
        self.index = 0
        self.lock = Lock()
    
    def get(self):
        with self.lock:
            s = self.sessions[self.index % len(self.sessions)]
            self.index += 1
            return s


pool = SessionPool(5)


def fetch_json(url, retries=2):
    full_url = f"{BASE_URL}{url}" if url.startswith('/') else url
    session = pool.get()
    
    for attempt in range(retries + 1):
        try:
            time.sleep(random.uniform(0.3, 0.6))
            r = session.get(full_url)
            if r.status_code == 200:
                return r.json()
            elif r.status_code == 403:
                print(f"  [!] 403 Forbidden - may be blocked")
                time.sleep(3)
        except:
            if attempt < retries:
                time.sleep(1)
    return None


def convert_fractional(frac_str):
    try:
        if '/' in str(frac_str):
            n, d = map(int, str(frac_str).split('/'))
            return round(1 + (n / d), 3)
        return float(frac_str)
    except:
        return None


def get_finished_matches(tournament_id, existing_ids, max_pages=20):
    """Get finished matches that are NOT in existing_ids."""
    seasons_data = fetch_json(f"/unique-tournament/{tournament_id}/seasons")
    if not seasons_data or 'seasons' not in seasons_data:
        return []
    
    season_id = seasons_data['seasons'][0]['id']
    new_matches = []
    
    for page in range(max_pages):
        data = fetch_json(f"/unique-tournament/{tournament_id}/season/{season_id}/events/last/{page}")
        if not data or 'events' not in data:
            break
        
        events = data.get('events', [])
        if not events:
            break
        
        page_new = 0
        for e in events:
            match_id = e.get('id')
            status = e.get('status', {}).get('type', '').lower()
            
            if status not in ['finished', 'ended']:
                continue
            
            if match_id in existing_ids:
                continue
            
            ts = e.get('startTimestamp', 0)
            new_matches.append({
                'match_id': match_id,
                'date': datetime.fromtimestamp(ts).strftime('%Y-%m-%d'),
                'timestamp': ts,
                'home_team': e.get('homeTeam', {}).get('name'),
                'home_team_id': e.get('homeTeam', {}).get('id'),
                'away_team': e.get('awayTeam', {}).get('name'),
                'away_team_id': e.get('awayTeam', {}).get('id'),
                'home_score': e.get('homeScore', {}).get('current'),
                'away_score': e.get('awayScore', {}).get('current'),
                'tournament_id': tournament_id,
            })
            page_new += 1
        
        print(f"    Page {page}: +{page_new} new")
    
    return new_matches


def enrich_match(match):
    """Add stats, odds, H2H to a match."""
    match_id = match['match_id']
    
    # Stats
    stats_data = fetch_json(f"/event/{match_id}/statistics")
    if stats_data and 'statistics' in stats_data:
        for period in stats_data.get('statistics', []):
            pname = period.get('period', 'ALL').lower()
            for g in period.get('groups', []):
                for item in g.get('statisticsItems', []):
                    name = item.get('name', '').lower().replace(' ', '_')
                    key = name if pname == 'all' else f'{pname}_{name}'
                    match[f'{key}_home'] = item.get('home')
                    match[f'{key}_away'] = item.get('away')
    
    # Odds
    odds_data = fetch_json(f"/event/{match_id}/odds/1/all")
    if odds_data and 'markets' in odds_data:
        for market in odds_data.get('markets', []):
            mid = market.get('marketId')
            for choice in market.get('choices', []):
                name = choice.get('name', '')
                dec = convert_fractional(choice.get('fractionalValue', ''))
                if not dec:
                    continue
                if mid == 1:
                    if name == '1': match['odds_1x2_home'] = dec
                    elif name == 'X': match['odds_1x2_draw'] = dec
                    elif name == '2': match['odds_1x2_away'] = dec
                elif mid == 5:
                    if name.lower() == 'yes': match['odds_btts_yes'] = dec
                    elif name.lower() == 'no': match['odds_btts_no'] = dec
    
    return match


def update_historical_data(input_file=None):
    """Main update function."""
    print("=" * 60)
    print("INCREMENTAL DATA UPDATER")
    print("=" * 60)
    
    # Find existing file
    if input_file is None:
        raw_dir = "data/raw"
        files = [f for f in os.listdir(raw_dir) if f.endswith('.csv')]
        if not files:
            print("[X] No existing CSV found in data/raw/")
            return None
        input_file = os.path.join(raw_dir, sorted(files)[-1])
    
    print(f"Loading: {input_file}")
    df_existing = pd.read_csv(input_file)
    existing_ids = set(df_existing['match_id'].astype(int).tolist())
    
    print(f"Existing matches: {len(df_existing):,}")
    print(f"Known match IDs: {len(existing_ids):,}")
    
    all_new = []
    
    for name, info in LEAGUES.items():
        print(f"\n[>] {name}")
        
        for tid in info['ids']:
            matches = get_finished_matches(tid, existing_ids, PAGES_TO_CHECK)
            
            if matches:
                print(f"  [+] Found {len(matches)} NEW matches")
                for m in matches:
                    m['league'] = name
                all_new.extend(matches)
                break
            else:
                print(f"  [-] No new matches")
    
    if not all_new:
        print("\n[OK] Data is already up to date!")
        return input_file
    
    # Enrich
    print(f"\n[*] Enriching {len(all_new)} new matches...")
    enriched = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(enrich_match, m) for m in all_new]
        for i, future in enumerate(as_completed(futures)):
            enriched.append(future.result())
            if (i + 1) % 10 == 0:
                print(f"  Progress: {i + 1}/{len(all_new)}")
    
    df_new = pd.DataFrame(enriched)
    
    # Combine
    df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    df_combined = df_combined.drop_duplicates(subset=['match_id'], keep='last')
    df_combined = df_combined.sort_values('date').reset_index(drop=True)
    
    # Save
    today = datetime.now().strftime('%Y%m%d')
    output_file = f"data/raw/sofascore_updated_{len(df_combined)}matches_{today}.csv"
    df_combined.to_csv(output_file, index=False)
    
    print(f"\n[OK] UPDATE COMPLETE!")
    print(f"   Previous: {len(df_existing):,} matches")
    print(f"   Added:    {len(df_new):,} new matches")
    print(f"   Total:    {len(df_combined):,} matches")
    print(f"   Saved:    {output_file}")
    
    return output_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default=None, help='Input CSV file')
    args = parser.parse_args()
    
    update_historical_data(args.input)
