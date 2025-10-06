import os
import json
import time
from tqdm import tqdm
from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException

load_dotenv()

class YouTubeClient:
    def __init__(self):
        key_str = os.getenv("YOUTUBE_API_KEYS", "")
        self.api_keys = [k.strip() for k in key_str.split(',') if k.strip()]
        # print(self.api_keys)
        if not self.api_keys:
            raise ValueError("No YouTube API keys found in YOUTUBE_API_KEYS environment variable.")
        self.key_idx = 0
        self.client = self._build_client(self.api_keys[self.key_idx])

    def _build_client(self, api_key):
        return build("youtube", "v3", developerKey=api_key)

    def _rotate_key(self):
        self.key_idx = (self.key_idx + 1) % len(self.api_keys)
        print(f"[YouTubeClient] Rotating to API key index {self.key_idx}")
        self.client = self._build_client(self.api_keys[self.key_idx])
    
    def _call_api(self, fn):
        while True:
            try:
                return fn()
            except HttpError as e:
                reason = None
                try:
                    err = json.loads(e.content.decode('utf-8'))
                    reason = err['error']['errors'][0].get('reason')
                    print(f"[YouTubeClient] API error reason: {reason}")
                except Exception:
                    pass

                # Rotate if quota-related or key is invalid
                if reason in (
                    'quotaExceeded', 'dailyLimitExceeded', 'userRateLimitExceeded',
                    'forbidden', 'keyInvalid', 'keyExpired', 'badRequest'
                ):
                    print("[YouTubeClient] API key issue (quota, invalid, or expired) — rotating key...")
                    self._rotate_key()
                    time.sleep(0.5)
                    continue

                # Special case: comments are disabled
                if reason == 'commentsDisabled':
                    return {'items': []}

                # Fallback: check content string for quota-related terms
                err_str = str(e).lower()
                if e.resp.status in (403, 429) and ('quota' in err_str or 'limit' in err_str or 'suspend' in err_str):
                    print("[YouTubeClient] Quota/suspension keyword in error string — rotating key...")
                    self._rotate_key()
                    time.sleep(0.5)
                    continue
                raise


    def search_videos(
        self, 
        keyword=None, 
        published_after=None, 
        published_before=None, 
        relevanceLanguage=None,
        channel_id=None, 
        max_results=50,
        order=None
    ):
        response = self._call_api(
            lambda: self.client.search().list(
                q=keyword,
                part="id",
                type="video",
                channelId=channel_id,
                maxResults=max_results,
                relevanceLanguage=relevanceLanguage,
                publishedAfter=published_after,
                publishedBefore=published_before,
                order=order
            ).execute()
        )
        
        return [item["id"]["videoId"] for item in response.get("items", [])]


    def fetch_snippet(self, video_id):
        video_resp = self._call_api(lambda: self.client.videos().list(
            part="snippet", id=video_id).execute())
        
        snippet = video_resp["items"][0]["snippet"]
        return {
            "vid": video_id,
            "title": snippet.get("title", ""),
            "published_time": snippet.get("publishedAt", ""),
            "description": snippet.get("description", ""),
        }


    def fetch_snippet_with_comments(self, video_id, max_page=20, max_comments=500, target_lang=None):
        data = self.fetch_snippet(video_id)
        data["comments"] = []
        next_page_token = None
        cbar = tqdm(desc=f"VID {video_id} comments", unit="page")
        at_page = 0

        while at_page < max_page and len(data["comments"]) < max_comments:
            comment_resp = self._call_api(
                lambda: self.client.commentThreads().list(
                    part="snippet",
                    videoId=video_id,
                    maxResults=50,
                    textFormat="plainText",
                    pageToken=next_page_token
                ).execute()
            )

            for item in comment_resp.get("items", []):
                cmt = item["snippet"]["topLevelComment"]["snippet"]
                cmt_text = cmt.get("textDisplay", "")

                # Language filter
                # print(target_lang)
  
                if target_lang:
                    try:
                        detected_lang = detect(cmt_text)
                        if target_lang == "zh":
                            allowed = {"zh-cn", "zh-tw", "zh", "ko"}  # strange langdetect bug ko
                            if detected_lang not in allowed:
                                # print(detected_lang, cmt_text)
                                continue
                        else:
                            if detected_lang != target_lang:
                                continue
                    except LangDetectException:
                        continue  # skip if detection fails

                data["comments"].append({
                    "text": cmt_text,
                    "published_time": cmt.get("publishedAt", "")
                })

                # Stop if reached max_comments
                if len(data["comments"]) >= max_comments:
                    break

            if len(data["comments"]) >= max_comments:
                break

            next_page_token = comment_resp.get("nextPageToken")
            cbar.update(1)
            if not next_page_token:
                break
            at_page += 1
            time.sleep(0.1)

        cbar.close()
        return data


    def search_playlists(
        self, 
        keyword=None, 
        published_after=None, 
        published_before=None, 
        relevanceLanguage=None,
        channel_id=None, 
        max_results=50,
        order=None
    ):
        response = self._call_api(
            lambda: self.client.search().list(
                q=keyword,
                part="id",
                type="playlist",
                channelId=channel_id,
                maxResults=max_results,
                relevanceLanguage=relevanceLanguage,
                publishedAfter=published_after,
                publishedBefore=published_before,
                order=order
            ).execute()
        )
        playlist_ids = []
        for item in response.get("items", []):
            if item.get("id", {}).get("kind") == "youtube#playlist":
                pid = item["id"].get("playlistId")
                if pid:
                    playlist_ids.append(pid)
        return playlist_ids


    def fetch_playlist_snippet(self, playlist_id):
        playlist_resp = self._call_api(lambda: self.client.playlists().list(
            part="snippet", id=playlist_id).execute())
        snippet = playlist_resp["items"][0]["snippet"]
        return {
            "pid": playlist_id,
            "title": snippet.get("title", ""),
            "published_time": snippet.get("publishedAt", ""),
            "description": snippet.get("description", ""),
        }


    def list_videos_in_playlist(self, playlist_id, max_page=20):
        """Return a list of video IDs in the playlist."""
        video_ids = []
        next_page_token = None
        at_page = 0
        pbar = tqdm(desc=f"Playlist {playlist_id} videos", unit="page")

        while at_page < max_page:
            pl_items_resp = self._call_api(
                lambda: self.client.playlistItems().list(
                    part="contentDetails",
                    playlistId=playlist_id,
                    maxResults=50,
                    pageToken=next_page_token
                ).execute()
            )

            for item in pl_items_resp.get("items", []):
                vid = item["contentDetails"]["videoId"]
                video_ids.append(vid)

            next_page_token = pl_items_resp.get("nextPageToken")
            pbar.update(1)
            if not next_page_token:
                break
            at_page += 1
            time.sleep(0.1)
        pbar.close()
        return video_ids


if __name__ == "__main__":
    youtube_client = YouTubeClient()
    
    # Search for playlists
    playlist_ids = youtube_client.search_playlists(
        keyword="Spanish Song 2025", 
        max_results=5
    )
    print("Playlists found:", playlist_ids)
    print("-----------")
        
    
    for pid in playlist_ids:
        # Fetch playlist info
        pl_info = youtube_client.fetch_playlist_snippet(pid)
        print(pl_info)
        print("-----------")
        
        # List videos in the playlist
        vids_in_playlist = youtube_client.list_videos_in_playlist(pid, max_page=1)
        print("Videos in playlist:", vids_in_playlist)

#     video_ids = youtube_client.search_videos(
#         keyword="election", 
#         channel_id="UCupvZG-5ko_eiXAupbDfxWw", 
#         published_after="2024-01-01T00:00:00Z", 
#         published_before="2024-12-31T00:00:00Z",
#         max_results=2
#     )
#     video_data = youtube_client.fetch_snippet(video_ids[0])
#     # print(video_data)







