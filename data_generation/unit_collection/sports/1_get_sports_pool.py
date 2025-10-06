import os
import json
from datetime import datetime
from zoneinfo import ZoneInfo
from tqdm import tqdm
from dotenv import load_dotenv

from youtube_client import YouTubeClient
from openai_client import OpenAIModel_parallel
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


def extract_teams_from_title_with_events(model, title, events):
    events = [
        {
            "team_match": e["strEvent"],
            "sports": e["strSport"],
            "league": e["strLeague"],
        }
        for e in events
    ]
    events_json = json.dumps(events, ensure_ascii=False)
    print(events_json)

    prompt = """
        You are a precise sports assistant.

        Task:
        - You are given a video title and a list of today's matches (events), each with information of team match, sports type, and league.
        - Your job is to identify which match (home vs. away teams) from the events list the video title is about.
        - IMPORTANT: The team names in your output MUST match exactly one of the pairs provided in the events list. Do not modify or shorten names.
        - If the title does not clearly refer to a match in the events list, return empty strings.

        Output format (strict JSON):
        {{
            "home_team": "<exact team name from events or empty string>",
            "away_team": "<exact team name from events or empty string>"
        }}

        Video title: "{title}"
        Today's events: {events}
        """

    try:
        output = model.generate(
            prompt=prompt.format(title=title, events=events_json), 
            response_format={"type": "json_object"}
        )
        print(output.text[0])
        data = json.loads(output.text[0])
        return data.get("home_team") or None, data.get("away_team") or None
    except Exception as e:
        print("JSON parse error:", e)
        return None, None


# ===================== MAIN FLOW =====================
def fetch_videos_and_match_details(lang, start_date, end_date, max_videos=3):
    yt = YouTubeClient()
    llm = OpenAIModel_parallel('gpt-4o-mini', temperature=0.8, max_tokens=9999)
    sports_client = SportsDBClient(api_key=os.getenv("SPORTSDB_API_KEY", ""))
    config = SEARCH_CONFIG[lang]
    videos = yt.search_videos(
        keyword=config["keyword"],
        published_after=f"{start_date}T00:00:00Z",
        published_before=f"{end_date}T23:59:59Z", 
        max_results=max_videos
    )
    print(f"Retrieved {len(videos)} videos from YouTube search.")

    all_results = []
    for vid in videos:
        snippet = yt.fetch_snippet(vid)
        title = snippet["title"]
        # ---------- TIMEZONE TRANSFER ----------
        published_tw = datetime.fromisoformat(snippet["published_time"]).replace(tzinfo=ZoneInfo("Asia/Taipei"))
        league_tz = LEAGUE_TIMEZONE[lang]
        published_local = published_tw.astimezone(ZoneInfo(league_tz))
        pub_date = published_local.date().isoformat()  # YYYY-MM-DD

        events_today = sports_client.get_events_by_date(pub_date, config["league"])
        if not events_today:
            print(f"[WARN] No events found for {pub_date} in {config['league']}")
            continue

        # LLM Judgement
        home_team, away_team = extract_teams_from_title_with_events(llm, title, events_today)
        if not home_team or not away_team:
            print(f"[WARN] No (home_team, away_team) at title: {title}")
            continue

        matched_events = [e for e in events_today if
                          (e["strHomeTeam"].lower() == home_team.lower() and e["strAwayTeam"].lower() == away_team.lower()) or
                          (e["strHomeTeam"].lower() == away_team.lower() and e["strAwayTeam"].lower() == home_team.lower())]
        if not matched_events:
            continue

        e = matched_events[0]
        match_id = e["idEvent"]

        match_details = None
        if config["sport"] == "baseball":
            match_details = get_baseball_event_details(sports_client.api_key, match_id)
        elif config["sport"] == "soccer":
            match_details = get_soccer_event_stats(sports_client.api_key, match_id)

        all_results.append({
            "vid": vid,
            "title": title,
            "published_time": pub_date,
            "game_info":{
                "league": config["league"],
                "sports": config["sport"],
                "home_team": home_team,
                "away_team": away_team,
                "score": {"home": e.get("intHomeScore"), "away": e.get("intAwayScore")},
                "match_id": match_id,
                "match_details": match_details,
            },
            "language": lang
        })

    return all_results


def main(
    lang: str, 
    start_str: str,
    end_str: str, 
    output_dir: str,
    max_snippets: int
):

    print(f"\n===== {lang} Search =====")
    results = fetch_videos_and_match_details(lang, start_str, end_str, max_snippets)
    # print(json.dumps(results, indent=2, ensure_ascii=False))
    save_dir = os.path.join(output_dir, f"{start_str}_{end_str}")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{lang}.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(results)} video entries for {lang} -> {save_path}")


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

    main(
        args.lang,
        args.start_str,
        args.end_str,
        args.output_dir,
        args.max_sport_games
    )
