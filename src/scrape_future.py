#!/usr/bin/env python3
"""
Future Matches Scraper - Local Version
Scrapes upcoming matches from Sofascore for the next 7 days.
"""

import time
import random
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from datetime import datetime, timedelta
import pandas as pd

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
    'Liga MX': {'ids': [11621, 11620]},  # Apertura + Clausura
}

BASE_URL = "https://www.sofascore.com/api/v1"


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
                print(f"  [!] 403 Forbidden - may be blocked by Cloudflare")
                time.sleep(3)
        except Exception as e:
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


def get_future_matches(tournament_id, days_ahead=7):
    """Get upcoming matches for a tournament."""
    seasons_data = fetch_json(f"/unique-tournament/{tournament_id}/seasons")
    if not seasons_data or 'seasons' not in seasons_data:
        return []
    
    season_id = seasons_data['seasons'][0]['id']
    matches = []
    
    for page in range(5):
        data = fetch_json(f"/unique-tournament/{tournament_id}/season/{season_id}/events/next/{page}")
        if not data or 'events' not in data:
            break
        
        cutoff = datetime.now() + timedelta(days=days_ahead)
        
        for e in data.get('events', []):
            ts = e.get('startTimestamp', 0)
            match_date = datetime.fromtimestamp(ts)
            
            if match_date > cutoff:
                continue
            
            status = e.get('status', {}).get('type', '')
            if status == 'finished':
                continue
            
            matches.append({
                'match_id': e.get('id'),
                'date': match_date.strftime('%Y-%m-%d'),
                'time': match_date.strftime('%H:%M'),
                'timestamp': ts,
                'home_team': e.get('homeTeam', {}).get('name'),
                'home_team_id': e.get('homeTeam', {}).get('id'),
                'away_team': e.get('awayTeam', {}).get('name'),
                'away_team_id': e.get('awayTeam', {}).get('id'),
                'tournament_id': tournament_id,
            })
    
    return matches


def get_match_odds(match_id):
    """Get pre-match odds."""
    odds = {}
    data = fetch_json(f"/event/{match_id}/odds/1/all")
    
    if data and 'markets' in data:
        for market in data.get('markets', []):
            mid = market.get('marketId')
            for choice in market.get('choices', []):
                name = choice.get('name', '')
                dec = convert_fractional(choice.get('fractionalValue', ''))
                if not dec:
                    continue
                if mid == 1:
                    if name == '1': odds['odds_1x2_home'] = dec
                    elif name == 'X': odds['odds_1x2_draw'] = dec
                    elif name == '2': odds['odds_1x2_away'] = dec
                elif mid == 5:
                    if name.lower() == 'yes': odds['odds_btts_yes'] = dec
                    elif name.lower() == 'no': odds['odds_btts_no'] = dec
    
    return odds


def scrape_future_matches(days_ahead=7):
    """Main scraping function."""
    print("=" * 60)
    print("FUTURE MATCHES SCRAPER")
    print("=" * 60)
    print(f"Days ahead: {days_ahead}")
    print(f"Leagues: {len(LEAGUES)}")
    
    all_matches = []
    
    for name, info in LEAGUES.items():
        print(f"\n[>] {name}...")
        
        for tid in info['ids']:
            matches = get_future_matches(tid, days_ahead)
            
            if matches:
                print(f"  Found {len(matches)} upcoming matches")
                for m in matches:
                    m['league'] = name
                all_matches.extend(matches)
                break
            else:
                print(f"  No matches in ID {tid}")
    
    if not all_matches:
        print("\n[X] No matches found. This might be due to Cloudflare blocking.")
        print("   Try running the Colab notebook instead: sofascore_future_v2.ipynb")
        return None
    
    # Enrich with odds
    print(f"\n[*] Getting odds for {len(all_matches)} matches...")
    for i, match in enumerate(all_matches):
        odds = get_match_odds(match['match_id'])
        match.update(odds)
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i + 1}/{len(all_matches)}")
    
    df = pd.DataFrame(all_matches)
    
    # Save
    today = datetime.now().strftime('%Y%m%d')
    filename = f"data/predictions/sofascore_future_{len(LEAGUES)}lg_{len(df)}matches_{today}.csv"
    df.to_csv(filename, index=False)
    
    print(f"\n[OK] Saved: {filename}")
    print(f"   Total matches: {len(df)}")
    
    return filename


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--days', type=int, default=7, help='Days ahead to scrape')
    args = parser.parse_args()
    
    scrape_future_matches(args.days)
