"""
Tournament Scraper

Fetch tournament and season data from Sofascore API.
"""

import logging
from typing import List, Dict, Any, Optional

from .api_client import SofascoreClient

import sys
sys.path.insert(0, str(__file__).rsplit('\\', 2)[0])
from config import settings

logger = logging.getLogger(__name__)


class TournamentScraper:
    """
    Scraper for tournament-level data (leagues, seasons, standings).
    
    Usage:
        scraper = TournamentScraper()
        seasons = scraper.get_seasons(11621)  # Liga MX
        standings = scraper.get_standings(11621, seasons[0]['id'])
    """
    
    def __init__(self, client: Optional[SofascoreClient] = None):
        """
        Initialize the tournament scraper.
        
        Args:
            client: Optional SofascoreClient instance. Creates new one if not provided.
        """
        self.client = client or SofascoreClient()
    
    def get_tournaments(self, category_id: int = None) -> List[Dict[str, Any]]:
        """
        Get all tournaments for a country/category.
        
        Args:
            category_id: Category ID (default: Mexico = 12)
        
        Returns:
            List of tournament dictionaries with id, name, slug, etc.
        """
        category_id = category_id or settings.MEXICO_CATEGORY_ID
        
        data = self.client.get_endpoint("category_tournaments", category_id=category_id)
        
        if not data or "groups" not in data:
            logger.warning(f"No tournaments found for category {category_id}")
            return []
        
        tournaments = []
        for group in data.get("groups", []):
            for tournament in group.get("uniqueTournaments", []):
                tournaments.append({
                    "id": tournament.get("id"),
                    "name": tournament.get("name"),
                    "slug": tournament.get("slug"),
                    "category": tournament.get("category", {}).get("name"),
                    "has_standings": tournament.get("hasStandings", False),
                })
        
        logger.info(f"Found {len(tournaments)} tournaments for category {category_id}")
        return tournaments
    
    def get_seasons(self, tournament_id: int) -> List[Dict[str, Any]]:
        """
        Get all seasons for a tournament.
        
        Args:
            tournament_id: Tournament ID (e.g., 11621 for Liga MX)
        
        Returns:
            List of season dictionaries with id, name, year.
        """
        data = self.client.get_endpoint("tournament_seasons", tournament_id=tournament_id)
        
        if not data or "seasons" not in data:
            logger.warning(f"No seasons found for tournament {tournament_id}")
            return []
        
        seasons = []
        for season in data.get("seasons", []):
            seasons.append({
                "id": season.get("id"),
                "name": season.get("name"),
                "year": season.get("year"),
            })
        
        logger.info(f"Found {len(seasons)} seasons for tournament {tournament_id}")
        return seasons
    
    def get_standings(self, tournament_id: int, season_id: int) -> List[Dict[str, Any]]:
        """
        Get standings (league table) for a specific season.
        
        Args:
            tournament_id: Tournament ID
            season_id: Season ID
        
        Returns:
            List of team standings with position, team info, stats.
        """
        data = self.client.get_endpoint(
            "standings",
            tournament_id=tournament_id,
            season_id=season_id
        )
        
        if not data or "standings" not in data:
            logger.warning(f"No standings found for tournament {tournament_id}, season {season_id}")
            return []
        
        standings = []
        for standing_group in data.get("standings", []):
            for row in standing_group.get("rows", []):
                team = row.get("team", {})
                standings.append({
                    "position": row.get("position"),
                    "team_id": team.get("id"),
                    "team_name": team.get("name"),
                    "team_slug": team.get("slug"),
                    "matches": row.get("matches"),
                    "wins": row.get("wins"),
                    "draws": row.get("draws"),
                    "losses": row.get("losses"),
                    "goals_for": row.get("scoresFor"),
                    "goals_against": row.get("scoresAgainst"),
                    "goal_difference": row.get("scoresFor", 0) - row.get("scoresAgainst", 0),
                    "points": row.get("points"),
                })
        
        logger.info(f"Found {len(standings)} teams in standings")
        return standings
    
    def get_all_team_ids(self, tournament_id: int, season_id: int) -> List[int]:
        """
        Get all team IDs from a season's standings.
        
        Args:
            tournament_id: Tournament ID
            season_id: Season ID
        
        Returns:
            List of team IDs.
        """
        standings = self.get_standings(tournament_id, season_id)
        return [s["team_id"] for s in standings if s.get("team_id")]
    
    def find_tournament_by_name(self, name: str, category_id: int = None) -> Optional[Dict[str, Any]]:
        """
        Find a tournament by name (case-insensitive partial match).
        
        Args:
            name: Tournament name to search for
            category_id: Category ID (default: Mexico)
        
        Returns:
            Tournament dictionary or None if not found.
        """
        tournaments = self.get_tournaments(category_id)
        name_lower = name.lower()
        
        for tournament in tournaments:
            if name_lower in tournament.get("name", "").lower():
                return tournament
        
        return None
    
    def group_related_tournaments(self, tournaments: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Group related tournaments (e.g., Liga MX Apertura + Clausura).
        
        Uses name similarity to group tournaments that are parts of the same league.
        Examples:
            - "Liga MX, Apertura" + "Liga MX, Clausura" -> "Liga MX"
            - "Liga de Expansión MX, Apertura" + "Liga de Expansión MX, Clausura" -> "Liga de Expansión MX"
        
        Args:
            tournaments: List of tournament dictionaries
        
        Returns:
            Dictionary mapping base name to list of related tournaments.
        """
        groups = {}
        
        for tournament in tournaments:
            name = tournament.get("name", "")
            
            # Extract base name by removing common suffixes
            base_name = name
            for suffix in [", Apertura", ", Clausura", ", Aperture", ", Closure", 
                          " Apertura", " Clausura", " Aperture", " Closure"]:
                if name.endswith(suffix):
                    base_name = name[:-len(suffix)]
                    break
            
            # Also handle ", Women" as a separate category
            is_women = ", Women" in base_name or " Women" in base_name
            if is_women:
                # Keep Women as separate group indicator
                base_name = base_name.replace(", Women", " (Women)").replace(" Women", " (Women)")
            
            if base_name not in groups:
                groups[base_name] = []
            groups[base_name].append(tournament)
        
        return groups
    
    def get_grouped_tournaments(self, category_id: int = None) -> List[Dict[str, Any]]:
        """
        Get tournaments grouped by base name with all related tournament IDs.
        
        Returns a simplified list showing grouped tournaments that includes
        all related tournament IDs for combined scraping.
        
        Args:
            category_id: Category ID (default: Mexico)
        
        Returns:
            List of grouped tournament info with:
                - group_name: Base name of the tournament group
                - tournament_ids: List of all tournament IDs in the group
                - tournaments: List of individual tournament dictionaries
                - description: Human-readable description
        """
        tournaments = self.get_tournaments(category_id)
        groups = self.group_related_tournaments(tournaments)
        
        result = []
        for base_name, group_tournaments in groups.items():
            # Sort by ID to keep consistent order
            group_tournaments.sort(key=lambda t: t.get("id", 0))
            
            tournament_ids = [t["id"] for t in group_tournaments]
            tournament_names = [t["name"] for t in group_tournaments]
            
            if len(group_tournaments) > 1:
                description = f"{base_name} ({', '.join([t['name'].split(', ')[-1] if ', ' in t['name'] else t['name'] for t in group_tournaments])})"
            else:
                description = group_tournaments[0]["name"]
            
            result.append({
                "group_name": base_name,
                "tournament_ids": tournament_ids,
                "tournaments": group_tournaments,
                "description": description,
                "count": len(group_tournaments),
            })
        
        # Sort by name for consistent display
        result.sort(key=lambda g: g["group_name"])
        
        logger.info(f"Grouped {len(tournaments)} tournaments into {len(result)} groups")
        return result


if __name__ == "__main__":
    # Test the tournament scraper
    scraper = TournamentScraper()
    
    print("=== Mexico Tournaments ===")
    tournaments = scraper.get_tournaments()
    for t in tournaments[:5]:
        print(f"  {t['id']}: {t['name']}")
    
    print("\n=== Liga MX Seasons ===")
    seasons = scraper.get_seasons(11621)
    for s in seasons[:5]:
        print(f"  {s['id']}: {s['name']} ({s['year']})")
    
    if seasons:
        print(f"\n=== Standings ({seasons[0]['name']}) ===")
        standings = scraper.get_standings(11621, seasons[0]["id"])
        for s in standings[:5]:
            print(f"  {s['position']}. {s['team_name']} - {s['points']} pts")
