import os
import json
import argparse
from datetime import datetime, timedelta
from typing import List, Dict, Any
from tqdm import tqdm
from googleapiclient.errors import HttpError
from youtube_client import YouTubeClient
from openai_client import OpenAIModel_parallel
from prompts import movie_genQA_prompts as PROMPT_TEMPLATE


RELEASE_TERMS = {
    "en": "English movie trailer",
    "zh": "中文電影預告",
    "ja": "日本語映画予告",
    "fr": "bande-annonce en français",
    "es": "tráiler en español",
}

MOVIE_QUERY = {
    "en": "{title} English movie trailer",
    "zh": "{title} 中文電影預告",
    "ja": "{title} 日本語映画予告",
    "fr": "{title} bande-annonce en français",
    "es": "{title} tráiler en español",
}

def gen_query(movie_snippet, lang):
    title = movie_snippet.get("aka") or movie_snippet.get("title")
    return f"{title} {RELEASE_TERMS[lang]}"


def single_movie_retrieval(
    yt: YouTubeClient,
    llm: OpenAIModel_parallel,
    lang: str,
    start_str: str, 
    end_str: str,
    movie_snippet: Dict[str, Any],
    min_comments: int
):

    search_query = gen_query(movie_snippet, lang)
    movie_title = movie_snippet.get('title', '')
    print(f"\n=== Processing movie: {movie_title} ===")
    print(f"Search query: {search_query}")

    # Step 1: Search videos
    video_ids = yt.search_videos(
        keyword=search_query,
        published_after=datetime.strptime(start_str, '%Y-%m-%d').strftime('%Y-%m-%dT00:00:00Z'), 
        published_before=datetime.strptime(end_str, '%Y-%m-%d').strftime('%Y-%m-%dT00:00:00Z'),
        max_results=20,
    )
    print(f"Retrieved {len(video_ids)} videos from YouTube search.")

    # Step 2: Filter with LLM
    valid_vids = []
    print("Running LLM validation on retrieved videos...")
    for vid in tqdm(video_ids, desc=f"LLM Filtering"):
        snippet = yt.fetch_snippet(vid)
        response = llm.generate(
            prompt=PROMPT_TEMPLATE.VIDEO_VALIDATION_TEMPLATE.format(
                query=search_query,
                title=snippet['title'], 
                description=snippet['description']
            ),
            response_format={"type": "json_object"}
        )
        print('query:', search_query)
        print('searched:', snippet['title'])
        print(response.text[0])
        try:
            valid = json.loads(response.text[0])["valid"]
            print(type(valid), valid)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"[WARNING] Failed to parse LLM output for {vid}: {response.text}")
            continue
        if valid:
            valid_vids.append(vid)

    print(f"✅ {len(valid_vids)} videos passed LLM validation out of {len(video_ids)}.")

    # Step 3: Fetch comments from valid videos
    print(f"\nFetching comments from {len(valid_vids)} valid videos...")
    comment_num = 0
    comments, comments_source = [], []
    for vid in tqdm(valid_vids, desc=f"Fetching Comments"):
        try:
            snippet = yt.fetch_snippet_with_comments(
                vid, 
                max_page=50, 
                max_comments=500, 
                target_lang=lang
            )
            comments.extend(snippet["comments"])
            comments_source.append((vid, snippet["title"]))

            comment_num += len(snippet["comments"])
            print(f"Collected {len(snippet['comments'])} comments from {vid} (total so far: {comment_num}).")
            if comment_num >= min_comments:
                print("Reached minimum comment threshold.")
                break
        except HttpError as e:
            print(f"[ERROR] Failed to fetch {vid}: {e}")

    if comment_num < min_comments:
        print(f"❌ Not enough comments collected: {comment_num} / {min_comments}")
        return None
    
    print(f"✅ Finished collecting {comment_num} comments for movie: {movie_title}.")
    movie_snippet["comments_source"] = comments_source
    movie_snippet["comments"] = comments[:min_comments]

    return movie_snippet



def main(
    movie_pool_dir: str, 
    output_dir: str, 
    min_comments: int, 
    max_snippets: int,
    lang: str,
):
    
    start_str, end_str = os.path.basename(movie_pool_dir).split("_")
    data_file = os.path.join(movie_pool_dir, f"{lang}.json")
    with open(data_file, 'r', encoding='utf-8') as f:
        movie_pool = json.load(f)
    yt = YouTubeClient()
    llm = OpenAIModel_parallel('gpt-4o-mini', temperature=0.8, max_tokens=9999)

    selected_snippets = []
    for movie_snippet in tqdm(movie_pool):
        snippet = single_movie_retrieval(
            yt, llm, lang, 
            start_str, end_str, 
            movie_snippet, min_comments, 
        )
        if not snippet:
            continue
        selected_snippets.append(snippet)

        if len(selected_snippets) >= max_snippets:
            print("[END] Reached maximum snippet count.")
            break

    save_dir = os.path.join(output_dir, f"{start_str}_{end_str}")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{lang}.json")
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(selected_snippets, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Movie comments")
    parser.add_argument(
        "--movie_pool_dir",
        type=str,
        default="data/movie/movie_pool/2025-01-01_2025-07-31",
    )
    parser.add_argument(
        "--domain_unit_dir",
        type=str,
        default="data/movie/domain_units",
    )
    parser.add_argument(
        "--min_comments",
        type=int,
        default=300,
        help="Minimum number of comments required to keep the video."
    )
    parser.add_argument(
        "--max_movies",
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
        movie_pool_dir=args.movie_pool_dir,
        output_dir=args.domain_unit_dir,
        min_comments=args.min_comments,
        max_snippets=args.max_movies,
        lang=args.lang,
    )