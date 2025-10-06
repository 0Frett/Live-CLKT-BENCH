import os
import json
import argparse
from pathlib import Path
import re
from tqdm import tqdm
from googleapiclient.errors import HttpError
from youtube_client import YouTubeClient


def main(lang: str, music_pool_dir: str, output_dir: str, min_comments: int, max_snippets: int):

    start_str, end_str = os.path.basename(music_pool_dir).split("_")
    data_file = os.path.join(music_pool_dir, f"{lang}.json")
    with open(data_file, "r", encoding="utf-8") as f:
        music_pool = json.load(f)

    yt = YouTubeClient()
    all_videos = []

    for snippet in tqdm(music_pool, desc="Fetching comments"):
        vid = snippet.get("vid")
        lang = snippet.get("language")

        try:
            video_data = yt.fetch_snippet_with_comments(
                vid, 
                max_page=50, 
                max_comments=min_comments,
                target_lang=lang, 
            )
            comment_count = len(video_data.get("comments", []))

            # Skip videos with too few comments
            if comment_count < min_comments:
                print(f"[Skip] {vid} only has {comment_count} comments.")
                continue

            video_data["lang"] = lang
            video_data['title'] = re.sub(r"[^\w\-_. ]", "_", video_data['title'])
            all_videos.append(video_data)

            if len(all_videos) >= max_snippets:
                print(f"[Early Stop] Already get {max_snippets} snippets.")
                break

        except HttpError as e:
            print(f"[Error] Failed to fetch {vid}: {e}")
            continue

    save_dir = os.path.join(output_dir, f"{start_str}_{end_str}")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{lang}.json")

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(all_videos, f, indent=2, ensure_ascii=False)

    print(f"[End] Saved {len(all_videos)} videos to {save_path}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Music Video comments")
    parser.add_argument(
        "--music_pool_dir",
        type=str,
        default="data/music/music_pool/2025-01-01_2025-07-31",
    )
    parser.add_argument(
        "--domain_unit_dir",
        type=str,
        default="data/music/domain_units",
    )
    parser.add_argument(
        "--min_comments",
        type=int,
        default=300,
        help="Minimum number of comments required to keep the video."
    )
    parser.add_argument(
        "--max_musics",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="en",
        help="Language code for movie trailers."
    )

    args = parser.parse_args()
    main(
        lang=args.lang,
        music_pool_dir=args.music_pool_dir, 
        output_dir=args.domain_unit_dir, 
        min_comments=args.min_comments, 
        max_snippets=args.max_musics
    )
