import json
import os
import re
import argparse
from tqdm import tqdm
from entity_collection_utils.movie import get_movie_entity
from entity_collection_utils.sports import get_sport_entity
from entity_collection_utils.music import get_music_entity


def main(
    domain: str,
    lang: str, 
    start_str: str,
    end_str: str, 
    output_dir: str,
    max_entity: int
):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    print(f"Collecting {domain} entities for language: {lang}")
    
    if domain == "movie":
        get_movie_entity(
            start_str=start_str, 
            end_str=end_str,
            lang=lang,
            output_dir=os.path.join(output_dir, "movie"),
            max_movies=max_entity
        )
    elif domain == "sports":
        get_sport_entity(
            lang=lang, 
            start_str=start_str, 
            end_str=end_str, 
            output_dir=os.path.join(output_dir, "sports"),
            max_games=max_entity
        )
    elif domain == "music":
        get_music_entity(
            lang=lang, 
            start_str=start_str, 
            end_str=end_str, 
            output_dir=os.path.join(output_dir, "music"),
            max_music=max_entity    
        )
    print(f"Entity collection for {domain} completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--domain",
        type=str,
        # default="movie",
        choices=["movie", "sports", "music"],
        help="Domain for entity collection."
    )
    parser.add_argument(
        "--start_str",
        type=str,
        default="2025-01-01",
        help="Start date for entity collection in YYYY-MM-DD format."
    )
    parser.add_argument(
        "--end_str",
        type=str,
        default="2025-07-31",
        help="End date for entity collection in YYYY-MM-DD format."
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
        help="Directory to save the collected entity."
    )
    parser.add_argument(
        "--max_entity",
        type=int,
        default=100,
        help="Maximum number of entities."
    )
    args = parser.parse_args()

    main(
        args.domain,
        args.lang,
        args.start_str,
        args.end_str,
        args.output_dir,
        args.max_entity
    )