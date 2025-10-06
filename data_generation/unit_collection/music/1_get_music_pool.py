import os
import json
import argparse
from datetime import datetime
from tqdm import tqdm
from youtube_client import YouTubeClient


MUSIC_SEARCH_TERMS = {
    "en": "New English song {year}",
    "zh": "新中文歌曲 {year}",                      # 新中文歌曲 (New Mandarin song)
    "ja": "新しい日本の歌 {year}",                   # 新しい日本の歌 (New Japanese song)
    "fr": "Nouvelle chanson en français {year}",  # Nouvelle chanson française (New French song)
    "es": "Nueva canción en español {year}",      # Nueva canción en español (New Spanish song)
}

def single_channel_retrieval(
    yt: YouTubeClient,
    lang: str,
    start_str: str, 
    end_str: str,
    max_snippets: int
):
    all_videos = []
    seen_vids = set()  # Track added video IDs to remove duplicates

    playlist_ids = yt.search_playlists(
        keyword=MUSIC_SEARCH_TERMS[lang].format(year=start_str[:4]),
        max_results=10
    )
    published_after = datetime.strptime(start_str, '%Y-%m-%d')
    published_before = datetime.strptime(end_str, '%Y-%m-%d')
    
    for pid in playlist_ids:
        vids_in_playlist = yt.list_videos_in_playlist(pid, max_page=50)
        pl_info = yt.fetch_playlist_snippet(pid)
        print(pl_info)

        print(f"\nStart Fetching {pid} Playlist {len(vids_in_playlist)} videos ...")
        passvid = 0
        for vid in tqdm(vids_in_playlist, desc=f"Videos"):
            if vid in seen_vids:
                continue  # Skip duplicates
            
            try:
                snippet = yt.fetch_snippet(vid)
                snippet['language'] = lang
                date = datetime.strptime(snippet['published_time'][:10], '%Y-%m-%d')
                if published_after <= date <= published_before:
                    all_videos.append(snippet)
                    seen_vids.add(vid)
                    passvid += 1
                    if max_snippets and len(all_videos) >= max_snippets:
                        print(f"Reached max snippets limit: {max_snippets}")
                        return all_videos
            except Exception as e:
                print(f"Failed to fetch {vid}: {e}")
        print(f"Obtained {passvid} videos from {pid} Playlist...")
    return all_videos


def main(
    lang: str, 
    start_str: str,
    end_str: str, 
    output_dir: str,
    max_snippets: int
):

    yt = YouTubeClient()
    snippets = single_channel_retrieval(yt, lang, start_str, end_str, max_snippets)
    
    save_dir = os.path.join(output_dir, f"{start_str}_{end_str}")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{lang}.json")
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(snippets, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Music Snippets")
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
        default="2025-07-31",
        help="End date for video collection in YYYY-MM-DD format."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/musics/music_pool",
    )
    parser.add_argument(
        "--max_musics",
        type=int,
        default=100,
        help="Maximum number of musics."
    )

    args = parser.parse_args()

    main(
        args.lang,
        args.start_str,
        args.end_str,
        args.output_dir,
        args.max_musics
    )
