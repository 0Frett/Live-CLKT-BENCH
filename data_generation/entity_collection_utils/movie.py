import json
import os
import re
import argparse
from tqdm import tqdm
from typing import List, Dict, Any
from movie_client import MovieClient
from constant import LANG_ISO_CODE 


def get_movie_entity(
    start_str: str, 
    end_str: str, 
    lang: str,
    output_dir: str,
    max_movies: int
) -> Dict[str, Any]:
    
    print(f"Initializing MovieClient...")
    client = MovieClient()

    print(f"\nFetching movies with theatrical (wide) release in: {LANG_ISO_CODE[lang]}")
    print(f"Time range: {start_str} to {end_str}\n")

    movies = client.get_movies(
        time_range=(start_str, end_str),
        ISOs=[LANG_ISO_CODE[lang]],
        max_movies=max_movies,
    )
    
    print(f"Found {len(movies)} valid movies.\n")

    movie_info = []
    cnt = 0
    for movie_title in tqdm(movies, desc="Downloading movie details"):
        try:
            info = client.get_movie_info(movie_title)
            # Required fields check
            if not info.get("title"):
                print(f"Skipping '{movie_title}': missing Title")
                continue
            if not info.get("summary"):
                print(f"Skipping '{movie_title}': missing Summary")
                continue
            if not info.get("top5cast"):
                print(f"Skipping '{movie_title}': missing Casts")
                continue
            if not info.get("release_dates"):
                print(f"Skipping '{movie_title}': missing Release_Dates")
                continue

            info["release_dates"] = info.get("release_dates").get(LANG_ISO_CODE[lang])
            info["aka"] = info.get("aka").get(LANG_ISO_CODE[lang])
            info['title'] = re.sub(r"[^\w\-_. ]", "_", info['title'])

            movie_info.append(info)
            cnt += 1

            if cnt >= max_movies:
                break
            
        except Exception as e:
            print(f"Error processing '{movie_title}': {e}")

    # save_dir = os.path.join(output_dir, f"{start_str}_{end_str}")
    os.makedirs(output_dir, exist_ok=True)

    save_path = os.path.join(output_dir, f"{start_str}_{end_str}.json")

    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(movie_info, f, indent=2, ensure_ascii=False)

    print(f"\nMovie Pool saved to: {save_path}")
    # return movie_info


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--start_str",
        type=str,
        default="2025-01-01",
        help="Start date for movie collection in YYYY-MM-DD format."
    )
    parser.add_argument(
        "--end_str",
        type=str,
        default="2025-07-31",
        help="End date for movie collection in YYYY-MM-DD format."
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="en",
        choices=["en", "zh", "ja", "fr", "es"],
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/movie/movie_pool",
        help="Directory to save the collected movie data."
    )
    parser.add_argument(
        "--max_movies",
        type=int,
        default=100,
        help="Maximum number of movies."
    )
    args = parser.parse_args()

    get_movie_entity(
        start_str=args.start_str, 
        end_str=args.end_str,
        lang=args.lang,
        output_dir=args.output_dir,
        max_movies=args.max_movies
    )
