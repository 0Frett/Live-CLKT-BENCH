import requests
import re
from datetime import datetime, timedelta
import argparse
import os
from dotenv import load_dotenv
load_dotenv()


class SportsDBClient:
    BASE_URL = "https://www.thesportsdb.com/api/v1/json"

    def __init__(self, api_key: str):
        self.api_key = api_key

    def _request(self, endpoint: str, params: dict) -> dict:
        url = f"{self.BASE_URL}/{self.api_key}/{endpoint}"
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"[SportsDBClient] error: {e}")
            return {}

    def get_events_by_date(self, date: str, league: str, window: int = 1) -> list:
        """
        Fetch events for date ± window days (default 1 = yesterday, today, tomorrow).
        date: str in "YYYY-MM-DD"
        league: league name
        """
        base_date = datetime.strptime(date, "%Y-%m-%d").date()
        dates_to_check = [
            base_date + timedelta(days=offset)
            for offset in range(-window, window + 1)
        ]

        all_events = []
        for d in dates_to_check:
            endpoint = "eventsday.php"
            params = {"d": d.isoformat(), "l": league}
            data = self._request(endpoint, params)
            events = data.get("events") or []
            all_events.extend(events)

        # Deduplicate by event ID
        seen = set()
        unique_events = []
        for e in all_events:
            if e["idEvent"] not in seen:
                seen.add(e["idEvent"])
                unique_events.append(e)

        return unique_events
    
    def get_events_by_range(self, start_date: str, end_date: str, league: str, max: int) -> list:
        """
        Fetch all events between start_date and end_date (inclusive).
        start_date, end_date: 'YYYY-MM-DD'
        """
        start = datetime.strptime(start_date, "%Y-%m-%d").date()
        end = datetime.strptime(end_date, "%Y-%m-%d").date()
        current = start
        all_events = []

        while current <= end:
            date_str = current.isoformat()
            daily_events = self.get_events_by_date(date_str, league, window=0)
            if daily_events:
                print(f"[SportsDBClient] got {len(daily_events)} events for {date_str}")
            else:
                print(f"[SportsDBClient] no events for {date_str}")
            all_events.extend(daily_events)
            current += timedelta(days=1)
            if len(all_events) >= (max+20):
                break

        # 去重複
        seen = set()
        unique_events = []
        for e in all_events:
            if e["idEvent"] not in seen:
                seen.add(e["idEvent"])
                unique_events.append(e)

        return unique_events



# ===================== PARSERS =====================

def parse_baseball_result(result_str):
    teams_data = {}
    parts = result_str.split("<br><br>")
    for part in parts:
        lines = part.strip().split("<br>")
        if len(lines) < 3:
            continue
        team_name = lines[0].replace("Innings:", "").strip()
        innings_scores = [int(x) for x in lines[1].split() if x.isdigit()]
        hits_errors = re.findall(r"Hits:\s*(\d+)\s*-\s*Errors:\s*(\d+)", lines[2])
        hits, errors = (map(int, hits_errors[0]) if hits_errors else (None, None))
        teams_data[team_name] = {"innings": innings_scores, "hits": hits, "errors": errors}
    return teams_data


def get_baseball_event_details(api_key, event_id):
    url = f"https://www.thesportsdb.com/api/v1/json/{api_key}/lookupevent.php?id={event_id}"
    res = requests.get(url)
    res.raise_for_status()
    data = res.json()
    events = data.get("events")
    if not events:
        return None
    event = events[0]
    parsed = parse_baseball_result(event.get("strResult", ""))
    return {
        "venue": event.get("strVenue"),
        "home_team": event.get("strHomeTeam"),
        "away_team": event.get("strAwayTeam"),
        "parsed_result": parsed,
    }


def get_soccer_event_stats(api_key, event_id):
    url = f"https://www.thesportsdb.com/api/v1/json/{api_key}/lookupeventstats.php?id={event_id}"
    res = requests.get(url)
    res.raise_for_status()
    data = res.json()
    stats = data.get("eventstats")
    if not stats:
        return None
    result = {}
    for s in stats:
        stat_name = s["strStat"]
        result[stat_name] = {"home": s.get("intHome"), "away": s.get("intAway")}
    return result


if __name__ == "__main__":
    sports_client = SportsDBClient(api_key=os.getenv("SPORTSDB_API_KEY", ""))
    events = sports_client.get_events_by_range("2025-09-01", "2025-09-05", "MLB", max=5)

    print(f"Total events: {len(events)}")
    print(events[:3])
    print(events[:-3])
    print(f"Total events: {len(events)}")