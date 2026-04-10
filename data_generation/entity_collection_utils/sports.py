import os
import json
from tqdm import tqdm
from dotenv import load_dotenv
from sports_client import SportsDBClient, get_baseball_event_details, get_soccer_event_stats
import argparse

load_dotenv()

# ===================== CONFIG =====================
SEARCH_CONFIG = {
    "en": {"keyword": "mlb highlight", "league": "MLB", "sport": "baseball"},
    "zh": {"keyword": "中華職棒精華", "league": "Chinese Professional Baseball League", "sport": "baseball"},
    "ja": {"keyword": "日本プロ野球ハイライト", "league": "Nippon Baseball League", "sport": "baseball"},
    "fr": {"keyword": "Ligue 1 3ème journée highlight", "league": "French Ligue 1", "sport": "soccer"},
    "es": {"keyword": "LaLiga highlights", "league": "Spanish La Liga", "sport": "soccer"}
}

LEAGUE_TIMEZONE = {
    "en": "America/New_York",
    "zh": "Asia/Taipei",
    "ja": "Asia/Tokyo",
    "fr": "Europe/Paris",
    "es": "Europe/Madrid",
}


def fetch_events(lang, start_date, end_date, max_games):
    config = SEARCH_CONFIG[lang]
    sports_client = SportsDBClient(api_key=os.getenv("SPORTSDB_API_KEY", "1"))

    print(f"Fetching events for {config['league']} ({config['sport']}) from {start_date} to {end_date}...")
    events = sports_client.get_events_by_range(start_date, end_date, config["league"], max_games)
    results = []

    for e in tqdm(events):
        match_id = e.get("idEvent")
        match_details = None

        try:
            if config["sport"] == "baseball":
                match_details = get_baseball_event_details(sports_client.api_key, match_id)
            elif config["sport"] == "soccer":
                match_details = get_soccer_event_stats(sports_client.api_key, match_id)
        except Exception as err:
            print(f"[WARN] Failed to fetch details for {match_id}: {err}")
        
        # Skip events with missing critical information
        if any(v is None for v in [
            config["league"],
            config["sport"],
            e.get("dateEvent"),
            e.get("strHomeTeam"),
            e.get("strAwayTeam"),
            e.get("intHomeScore"),
            e.get("intAwayScore"),
            match_id,
            match_details
        ]):
            print(f"[WARN] Skipping event {match_id} due to missing fields.")
            continue
        
        game_info = {
            "league": config["league"],
            "sports": config["sport"],
            "date": e.get("dateEvent"),
            "home_team": e.get("strHomeTeam"),
            "away_team": e.get("strAwayTeam"),
            "score": {"home": e.get("intHomeScore"), "away": e.get("intAwayScore")},
            "match_id": match_id,
            "match_details": match_details
        }
        


        results.append({
            "title": f"{e.get("strEvent")} ({e.get("dateEvent")})",
            "game_info": game_info, 
            "language": lang
        })
        if len(results) >= max_games:
            break

    return results

def get_sport_entity(
    lang: str, 
    start_str: str,
    end_str: str, 
    output_dir: str,
    max_games: int
):

    print(f"\n===== {lang} Search =====")
    results = fetch_events(lang, start_str, end_str, max_games)
    # print(json.dumps(results, indent=2, ensure_ascii=False))
    # save_dir = os.path.join(output_dir, f"{start_str}_{end_str}")
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"{start_str}_{end_str}.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(results)} video entries -> {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Sport Games Snippets")
    parser.add_argument(
        "--lang",
        type=str,
        default="en",
        choices=["en", "zh", "ja", "fr", "es"],
        help="Language code for music data collection."
    )
    parser.add_argument(
        "--start_str",
        type=str,
        default="2025-01-01",
        help="Start date for video collection in YYYY-MM-DD format."
    )
    parser.add_argument(
        "--end_str",
        type=str,
        default="2025-08-31",
        help="End date for video collection in YYYY-MM-DD format."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/sports/sports_pool",
    )
    parser.add_argument(
        "--max_sport_games",
        type=int,
        default=100,
        help="Maximum number of sport games."
    )

    args = parser.parse_args()

    get_sport_entity(
        args.lang,
        args.start_str,
        args.end_str,
        args.output_dir,
        args.max_sport_games
    )
