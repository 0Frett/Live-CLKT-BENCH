import os
import requests
from imdb import Cinemagoer
from datetime import datetime
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()


class MovieClient:
    def __init__(self):
        self.tmdb_v3_key = os.getenv("TMDB_V3_API_KEY", "")
        self.tmdb_v4_token = os.getenv("TMDB_V4_API_KEY", "")
        self.tmdb_v3_base_url = "https://api.themoviedb.org/3"
        self.tmdb_v4_headers = {
            "Authorization": f"Bearer {self.tmdb_v4_token}",
            "accept": "application/json"
        }
        self.imdb = Cinemagoer()
        self.RELEASE_TYPE_MAP = {
            1: "Premiere",
            2: "Theatrical (limited)",
            3: "Theatrical (wide)",
            4: "Digital",
            5: "Physical",
            6: "TV"
        }


    def get_movie_info(self, title, lang="en-US"):
        imdb_data = {}

        # IMDb data
        try:
            movies = self.imdb.search_movie(title)

            # print(movies)
            if movies:
                movie = self.imdb.get_movie(movies[0].getID(), info=['main', 'plot'])
                print("movie keys:", movie.keys())
                print("plot:", repr(movie.get("plot")))
                print("synopsis:", repr(movie.get("synopsis")))

                imdb_data['title'] = movie.get('title', 'N/A')

                plot = movie.get('plot', [])
                if isinstance(plot, list):
                    plot = plot[0] if plot else ""
                imdb_data['summary'] = plot
                imdb_data['synopsis'] = movie.get('synopsis', 'N/A')
        except Exception as e:
            print(f"[IMDb ERROR] title={title} error={e}")
            imdb_data['Error'] = f"IMDb error: {e}"

        # TMDB data
        search_url = f"{self.tmdb_v3_base_url}/search/movie"
        params = {"query": title, "include_adult": "false", "language": lang}
        res = requests.get(search_url, params=params, headers=self.tmdb_v4_headers).json()
        if not res.get("results"):
            imdb_data['TMDB'] = "Not found"
            return imdb_data

        movie_id = res["results"][0]["id"]

        details = requests.get(
            f"{self.tmdb_v3_base_url}/movie/{movie_id}",
            headers=self.tmdb_v4_headers
        ).json()
        credits = requests.get(
            f"{self.tmdb_v3_base_url}/movie/{movie_id}/credits",
            headers=self.tmdb_v4_headers
        ).json()
        releases = requests.get(
            f"{self.tmdb_v3_base_url}/movie/{movie_id}/release_dates",
            headers=self.tmdb_v4_headers
        ).json()
        alt_titles = requests.get(
            f"{self.tmdb_v3_base_url}/movie/{movie_id}/alternative_titles",
            headers=self.tmdb_v4_headers
        ).json()

        # fallback summary
        if not imdb_data.get("summary"):
            imdb_data["summary"] = details.get("overview", "")

        imdb_data['top5cast'] = [c["name"] for c in credits.get("cast", [])[:5]]

        release_info = {}
        for entry in releases.get("results", []):
            country = entry.get("iso_3166_1")
            for rd in entry.get("release_dates", []):
                if rd.get("type") == 3:
                    date = self._format_date(rd.get("release_date"))
                    release_info[country] = date

        imdb_data['release_dates'] = release_info
        imdb_data['aka'] = {
            aka.get("iso_3166_1", "--"): aka.get("title")
            for aka in alt_titles.get("titles", [])
            if aka.get("title")
        }

        return imdb_data


    def get_movies(self, time_range, ISOs, max_movies, lang="en-US", page_limit=100):
        start_str, end_str = time_range
        start = datetime.strptime(start_str, "%Y-%m-%d").date()
        end = datetime.strptime(end_str, "%Y-%m-%d").date()
        target_countries = set(ISOs)

        matching_titles = []
        url = f"{self.tmdb_v3_base_url}/discover/movie"
        with tqdm(total=max_movies, desc="Collecting movies", unit="movie") as pbar:
            for page in range(1, page_limit + 1):
                params = {
                    "api_key": self.tmdb_v3_key,
                    "language": lang,
                    "sort_by": "popularity.desc",
                    "include_adult": False,
                    "include_video": False,
                    "page": page,
                    "primary_release_date.gte": start_str,
                    "primary_release_date.lte": end_str,
                    "vote_count.gte": 10
                }
                response = requests.get(url, params=params).json()
                movies_page = response.get("results", [])

                for movie in movies_page:
                    if self._theatrical_release_in_all_countries(movie['id'], start, end, target_countries):
                        matching_titles.append(movie['title'])
                        pbar.update(1)
                        if len(matching_titles) >= max_movies:
                            return matching_titles  # ✅ Stop as soon as we reach limit

                if page >= response.get("total_pages", 1):
                    break

        return matching_titles


    def _theatrical_release_in_all_countries(self, movie_id, start_date, end_date, required_countries):
        url = f"{self.tmdb_v3_base_url}/movie/{movie_id}/release_dates"
        params = {"api_key": self.tmdb_v3_key}
        response = requests.get(url, params=params)
        data = response.json()

        matched_countries = set()

        for entry in data.get("results", []):
            country = entry.get("iso_3166_1")
            if country not in required_countries:
                continue

            for r in entry.get("release_dates", []):
                if r.get("type") != 3:
                    continue
                try:
                    date = datetime.fromisoformat(r["release_date"].replace("Z", "+00:00")).date()
                    if start_date <= date <= end_date:
                        matched_countries.add(country)
                        break
                except Exception:
                    continue

        return matched_countries == required_countries


    def _format_date(self, date_str):
        try:
            dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            return dt.strftime("%Y-%m-%d")
        except:
            return date_str
