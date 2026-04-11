"""
01_collect.py - Reddit data collection via Arctic Shift API
CS7NS6 / Text Analytics - Group H

Collects posts and comments via time-windowed requests to avoid API timeouts.
Medical phrase filtering is done locally after download.
Retries automatically on 422 errors.

Comment columns:
  - is_op:        True if the comment author is the original post author
  - is_top_level: True if direct reply to the post (parent_id starts with t3_)
  - comment_type: peer_advice | validation | general | op_reply

API docs: https://github.com/ArthurHeitmann/arctic_shift/blob/master/api/README.md
Dependencies: see requirements.txt
"""

import requests
import pandas as pd
import time
import logging
from datetime import datetime
from dateutil.relativedelta import relativedelta
from dataclasses import dataclass, field
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
log = logging.getLogger(__name__)


# ---
# CONFIGURATION
# ---
API_BASE   = "https://arctic-shift.photon-reddit.com"
OUTPUT_DIR = Path("output")

DATE_FROM      = "2019-01-01"
DATE_TO        = "2024-12-31"
WINDOW_MONTHS  = 2
MAX_PER_WINDOW = 1000

MIN_TEXT_LENGTH        = 50
MAX_COMMENTS_PER_POST  = 100
SLEEP_BETWEEN_REQUESTS = 1.5
RETRY_ON_TIMEOUT       = 3
RETRY_SLEEP            = 30

# ---
# MEDICAL POST KEY PHRASES
# ---
MEDICAL_PHRASES = [
    "doctor said",
    "doctor told me",
    "doctor told",
    "doctor recommended",
    "doctor suggested",
    "doctor advised",
    "doctor prescribed",
    "was told by my doctor",
    "was told by the doctor",
    "nurse said",
    "nurse told me",
    "therapist said",
    "therapist told me",
    "therapist suggested",
    "psychiatrist said",
    "psychiatrist told me",
    "psychologist said",
    "physician said",
    "physician told me",
    "specialist said",
    "specialist told me",
    "consultant said",
    "surgeon said",
    "neurologist said",
    "cardiologist said",
    "oncologist said",
    "was prescribed",
    "diagnosis was",
    "diagnosed me with",
    "referred me to",
    "my GP",
    "my consultant",
    "the specialist",
    "my conunsellor"
]

# ---
# COMMENT CLASSIFICATION PHRASES
# Applied only to non-OP comments.
# OP comments are handled separately as advice acceptance signals.
# ---
PEER_ADVICE_PHRASES = [
    "i went through the same",
    "same thing happened to me",
    "i had the same",
    "i also had",
    "i was also told",
    "i was also diagnosed",
    "i was also prescribed",
    "i experienced the same",
    "been there",
    "i know how you feel",
    "i felt the same",
    "in my experience",
    "what worked for me",
    "i found that",
    "personally i",
    "from my experience",
    "what helped me",
    "i also went through",
    "i dealt with the same",
    "i had a similar",
    "something similar happened to me",
    "i can relate",
    "same here",
    "me too",
    "i understand what you",
]

VALIDATION_PHRASES = [
    "my doctor said the same",
    "that's what my doctor said",
    "same diagnosis",
    "i was prescribed the same",
    "second opinion",
    "get a second opinion",
    "that sounds right",
    "that's correct",
    "that's normal",
    "this is common",
    "my doctor told me the same",
    "that's standard",
    "that's typical",
    "that's unusual",
    "that doesn't sound right",
    "you should get another opinion",
]

# ---
# TARGET SUBREDDITS
# ---
MEDICAL_SUBREDDITS = {
    "chronic_mental":   ["depression", "anxiety", "bipolar", "OCD", "mentalhealth", "depression_help", "bipolarreddit"],
    "chronic_physical": ["diabetes", "cancer", "ChronicPain", "MultipleSclerosis", "chronicillness", "lupus", "rheumatoid"],
    "acute_mental":     ["ptsd", "traumatoolbox"],
    "acute_physical":   ["medical", "AskDocs"],
}

NON_MEDICAL_SUBREDDITS = {
    "non_medical": ["personalfinance", "relationship_advice", "legaladvice"],
}

POST_FIELDS    = "id,author,subreddit,title,selftext,score,num_comments,created_utc,url"
COMMENT_FIELDS = "id,link_id,parent_id,author,body,score,created_utc"


# ---
# DATA MODEL
# ---
@dataclass
class RedditPost:
    post_id:      str
    author:       str
    subreddit:    str
    category:     str
    title:        str
    text:         str
    score:        int
    num_comments: int
    created_utc:  datetime
    url:          str = ""

    def to_dict(self) -> dict:
        return {
            "post_id":      self.post_id,
            "author":       self.author,
            "subreddit":    self.subreddit,
            "category":     self.category,
            "title":        self.title,
            "text":         self.text,
            "score":        self.score,
            "num_comments": self.num_comments,
            "created_utc":  self.created_utc.isoformat(),
            "url":          self.url,
        }


@dataclass
class RedditComment:
    comment_id:   str
    post_id:      str
    parent_id:    str        # t3_<post_id> = top-level, t1_<comment_id> = nested
    author:       str
    text:         str
    score:        int
    created_utc:  datetime
    is_op:        bool       # True if author == post author
    is_top_level: bool       # True if direct reply to post
    comment_type: str = "general"   # peer_advice | validation | general | op_reply

    def to_dict(self) -> dict:
        return {
            "comment_id":   self.comment_id,
            "post_id":      self.post_id,
            "parent_id":    self.parent_id,
            "author":       self.author,
            "text":         self.text,
            "score":        self.score,
            "created_utc":  self.created_utc.isoformat(),
            "is_op":        self.is_op,
            "is_top_level": self.is_top_level,
            "comment_type": self.comment_type,
        }


# ---
# API CLIENT with retry
# ---
def api_get(endpoint: str, params: dict) -> list[dict]:
    url = f"{API_BASE}{endpoint}"
    for attempt in range(1, RETRY_ON_TIMEOUT + 1):
        try:
            response = requests.get(url, params=params, timeout=60)
            if response.status_code == 200:
                return response.json().get("data", [])
            if response.status_code == 422:
                log.warning(
                    f"    [!] API timeout (attempt {attempt}/{RETRY_ON_TIMEOUT}), "
                    f"waiting {RETRY_SLEEP}s..."
                )
                time.sleep(RETRY_SLEEP)
                continue
            log.error(f"    [!] HTTP {response.status_code} [{endpoint}]: {response.text[:300]}")
            return []
        except requests.exceptions.RequestException as e:
            log.error(f"    [!] Network error [{endpoint}]: {e}")
            if attempt < RETRY_ON_TIMEOUT:
                time.sleep(RETRY_SLEEP)
    return []


# ---
# TIME WINDOWS
# ---
def time_windows(date_from: str, date_to: str, months: int):
    start = datetime.fromisoformat(date_from)
    end   = datetime.fromisoformat(date_to)
    while start < end:
        window_end = min(start + relativedelta(months=months), end)
        yield start.strftime("%Y-%m-%d"), window_end.strftime("%Y-%m-%d")
        start = window_end


# ---
# PAGINATION WITHIN A WINDOW
# ---
def paginate_window(endpoint: str, base_params: dict,
                    window_start: str, window_end: str) -> list[dict]:
    all_results  = []
    params       = {**base_params, "after": window_start, "before": window_end,
                    "limit": 100, "sort": "desc"}
    date_from_ts = datetime.fromisoformat(window_start).timestamp()

    while True:
        results = api_get(endpoint, params)
        if not results:
            break
        for item in results:
            if int(item.get("created_utc", 0)) < date_from_ts:
                return all_results
            all_results.append(item)
            if len(all_results) >= MAX_PER_WINDOW:
                log.info(f"        window limit reached ({MAX_PER_WINDOW}), moving to next")
                return all_results
        if len(results) < 100:
            break
        params["before"] = results[-1].get("created_utc")
        log.info(f"        page fetched: {len(all_results)} total results in window...")
        time.sleep(SLEEP_BETWEEN_REQUESTS)

    return all_results


# ---
# PHRASE FILTERS
# ---
def contains_medical_phrase(text: str) -> bool:
    t = text.lower()
    return any(p in t for p in MEDICAL_PHRASES)


def classify_comment(text: str, is_op: bool) -> str:
    """
    Classifies a comment into one of four types:
      - "op_reply"    : written by the original post author (advice acceptance context)
      - "peer_advice" : non-OP sharing personal experience
      - "validation"  : non-OP confirming or challenging medical advice
      - "general"     : everything else
    """
    if is_op:
        return "op_reply"
    t = text.lower()
    if any(p in t for p in PEER_ADVICE_PHRASES):
        return "peer_advice"
    if any(p in t for p in VALIDATION_PHRASES):
        return "validation"
    return "general"


# ---
# MEDICAL POST COLLECTION
# ---
def collect_medical_posts(subreddit_name: str, category: str) -> list[RedditPost]:
    windows   = list(time_windows(DATE_FROM, DATE_TO, WINDOW_MONTHS))
    n_windows = len(windows)
    log.info(f"  r/{subreddit_name} ({category}) - {n_windows} time windows")

    params = {
        "subreddit": subreddit_name,
        "fields":    POST_FIELDS,
    }

    posts     = []
    total_raw = 0

    for i, (w_start, w_end) in enumerate(windows, 1):
        raw        = paginate_window("/api/posts/search", params, w_start, w_end)
        total_raw += len(raw)
        before_filter = len(posts)

        for item in raw:
            text   = (item.get("selftext") or "").strip()
            title  = (item.get("title")    or "").strip()
            author = (item.get("author")   or "").strip()

            if not text or text in ("[deleted]", "[removed]"):
                continue
            if len(text) < MIN_TEXT_LENGTH:
                continue
            if author in ("[deleted]", "AutoModerator", ""):
                continue
            if not contains_medical_phrase(text):
                continue

            posts.append(RedditPost(
                post_id      = item.get("id", ""),
                author       = author,
                subreddit    = subreddit_name,
                category     = category,
                title        = title,
                text         = text,
                score        = item.get("score", 0),
                num_comments = item.get("num_comments", 0),
                created_utc  = datetime.utcfromtimestamp(int(item.get("created_utc", 0))),
                url          = item.get("url", ""),
            ))

        added = len(posts) - before_filter
        log.info(
            f"    window {i}/{n_windows} [{w_start} → {w_end}]: "
            f"{len(raw)} downloaded, {added} with phrases | total: {len(posts)}"
        )

    log.info(
        f"  r/{subreddit_name}: {len(posts)} filtered posts out of {total_raw} total "
        f"({len(posts)/max(total_raw,1)*100:.1f}%)"
    )
    return posts


# ---
# NON-MEDICAL POST COLLECTION
# ---
def collect_nonmedical_posts(subreddit_name: str, category: str,
                              medical_authors: set) -> list[RedditPost]:
    log.info(f"  r/{subreddit_name} - {len(medical_authors)} authors to search")

    posts       = []
    found_count = 0

    for i, author in enumerate(medical_authors, 1):
        params = {
            "author":    author,
            "subreddit": subreddit_name,
            "after":     DATE_FROM,
            "before":    DATE_TO,
            "fields":    POST_FIELDS,
        }

        raw          = api_get("/api/posts/search", params)
        author_posts = 0

        for item in raw:
            text  = (item.get("selftext") or "").strip()
            title = (item.get("title")    or "").strip()

            if not text or text in ("[deleted]", "[removed]"):
                continue
            if len(text) < MIN_TEXT_LENGTH:
                continue

            posts.append(RedditPost(
                post_id      = item.get("id", ""),
                author       = author,
                subreddit    = subreddit_name,
                category     = category,
                title        = title,
                text         = text,
                score        = item.get("score", 0),
                num_comments = item.get("num_comments", 0),
                created_utc  = datetime.utcfromtimestamp(int(item.get("created_utc", 0))),
                url          = item.get("url", ""),
            ))
            author_posts += 1

        if author_posts > 0:
            found_count += 1

        if i % 50 == 0:
            log.info(
                f"    author {i}/{len(medical_authors)} | "
                f"posts collected so far: {len(posts)} | "
                f"authors with posts: {found_count}"
            )

        time.sleep(SLEEP_BETWEEN_REQUESTS)

    log.info(
        f"  r/{subreddit_name}: {len(posts)} non-medical posts "
        f"from {found_count}/{len(medical_authors)} authors "
        f"({found_count/max(len(medical_authors),1)*100:.1f}% active here)"
    )
    return posts


# ---
# COMMENT COLLECTION - all comments, no phrase filter
# ---
def collect_comments(post: RedditPost,
                     limit: int = MAX_COMMENTS_PER_POST) -> list[RedditComment]:
    """Fetches all comments for a post, no phrase filter. Sets is_op, is_top_level, comment_type."""
    params = {
        "link_id": post.post_id,
        "limit":   min(limit, 100),
        "fields":  COMMENT_FIELDS,
    }

    raw      = api_get("/api/comments/search", params)
    comments = []

    for item in raw[:limit]:
        body      = (item.get("body")      or "").strip()
        author    = (item.get("author")    or "").strip()
        parent_id = (item.get("parent_id") or "").strip()

        if not body or body in ("[deleted]", "[removed]"):
            continue
        if author in ("[deleted]", "AutoModerator", ""):
            continue

        is_op        = (author == post.author)
        is_top_level = parent_id.startswith("t3_")
        comment_type = classify_comment(body, is_op)

        comments.append(RedditComment(
            comment_id   = item.get("id", ""),
            post_id      = post.post_id,
            parent_id    = parent_id,
            author       = author,
            text         = body,
            score        = item.get("score", 0),
            created_utc  = datetime.utcfromtimestamp(int(item.get("created_utc", 0))),
            is_op        = is_op,
            is_top_level = is_top_level,
            comment_type = comment_type,
        ))

    time.sleep(SLEEP_BETWEEN_REQUESTS)
    return comments


# ---
# MAIN PIPELINE
# ---
def run(collect_comments_flag: bool = True):
    OUTPUT_DIR.mkdir(exist_ok=True)

    total_subreddits = sum(len(v) for v in MEDICAL_SUBREDDITS.values())
    log.info(f"Medical phrases loaded:  {len(MEDICAL_PHRASES)}")
    log.info(f"Peer advice phrases:     {len(PEER_ADVICE_PHRASES)}")
    log.info(f"Validation phrases:      {len(VALIDATION_PHRASES)}")
    log.info(f"Medical subreddits:      {total_subreddits} | "
             f"Non-medical: {sum(len(v) for v in NON_MEDICAL_SUBREDDITS.values())}")
    log.info(f"Time range: {DATE_FROM} → {DATE_TO} ({WINDOW_MONTHS}-month windows)")
    log.info(f"Max comments per post:   {MAX_COMMENTS_PER_POST}")

    log.info("\nPhase 1: collecting medical posts")

    medical_posts = []
    sub_done      = 0

    for category, subreddit_list in MEDICAL_SUBREDDITS.items():
        log.info(f"\n[category: {category}]")
        for sub_name in subreddit_list:
            result = collect_medical_posts(sub_name, category)
            medical_posts.extend(result)
            sub_done += 1
            log.info(
                f"  PHASE 1 progress: {sub_done}/{total_subreddits} subreddits | "
                f"total medical posts: {len(medical_posts)}"
            )

    medical_authors = {p.author for p in medical_posts}
    log.info(f"\n{len(medical_authors)} unique authors in medical posts")

    total_nonmedical_subs = sum(len(v) for v in NON_MEDICAL_SUBREDDITS.values())
    log.info("\nPhase 2: collecting non-medical posts (same authors)")

    nonmedical_posts = []
    sub_done2        = 0

    for category, subreddit_list in NON_MEDICAL_SUBREDDITS.items():
        log.info(f"\n[category: {category}]")
        for sub_name in subreddit_list:
            result = collect_nonmedical_posts(sub_name, category, medical_authors)
            nonmedical_posts.extend(result)
            sub_done2 += 1
            log.info(
                f"  PHASE 2 progress: {sub_done2}/{total_nonmedical_subs} subreddits | "
                f"total non-medical posts: {len(nonmedical_posts)}"
            )

    all_posts = medical_posts + nonmedical_posts

    all_comments = []
    if collect_comments_flag:
        log.info(f"\nPhase 3: collecting comments ({len(all_posts)} posts)")

        for i, post in enumerate(all_posts, 1):
            comments = collect_comments(post)
            all_comments.extend(comments)
            if i % 100 == 0:
                # show comment type breakdown every 100 posts
                op_count  = sum(1 for c in all_comments if c.is_op)
                pa_count  = sum(1 for c in all_comments if c.comment_type == "peer_advice")
                val_count = sum(1 for c in all_comments if c.comment_type == "validation")
                log.info(
                    f"  {i}/{len(all_posts)} posts | "
                    f"comments: {len(all_comments)} total | "
                    f"op_reply: {op_count} | peer_advice: {pa_count} | validation: {val_count}"
                )

    # ---- SAVING -------------------------------------------------------------
    posts_df    = pd.DataFrame([p.to_dict() for p in all_posts])
    comments_df = pd.DataFrame([c.to_dict() for c in all_comments])

    timestamp     = datetime.now().strftime("%Y%m%d_%H%M%S")
    posts_path    = OUTPUT_DIR / f"reddit_posts.csv"
    comments_path = OUTPUT_DIR / f"reddit_comments.csv"

    posts_df.to_csv(posts_path,       index=False, encoding="utf-8")
    comments_df.to_csv(comments_path, index=False, encoding="utf-8")

    nonmedical_authors = {p.author for p in nonmedical_posts}
    crossover          = medical_authors & nonmedical_authors

    log.info("\nCollection complete:")
    log.info(f"Medical posts:                    {len(medical_posts)}")
    log.info(f"Non-medical posts:                {len(nonmedical_posts)}")
    log.info(f"Total posts:                      {len(posts_df)}")
    log.info(f"Total comments:                   {len(all_comments)}")
    log.info(f"Unique medical authors:           {len(medical_authors)}")
    log.info(f"Authors with posts in both:       {len(crossover)} "
             f"({len(crossover)/max(len(medical_authors),1)*100:.1f}%)")

    if not posts_df.empty:
        log.info(f"\nDistribution by category:")
        dist = posts_df.groupby("category")["post_id"].count()
        for cat, count in dist.items():
            log.info(f"  {cat:<25} {count}")

    if not comments_df.empty:
        log.info(f"\nComment type distribution:")
        cdist = comments_df["comment_type"].value_counts()
        for ctype, count in cdist.items():
            pct = count / len(comments_df) * 100
            log.info(f"  {ctype:<20} {count:>6}  ({pct:.1f}%)")

        op_comments = comments_df[comments_df["is_op"] == True]
        log.info(f"\nOP reply comments:              {len(op_comments)} "
                 f"({len(op_comments)/max(len(comments_df),1)*100:.1f}%)")
        top_level = comments_df[comments_df["is_top_level"] == True]
        log.info(f"Top-level comments:             {len(top_level)} "
                 f"({len(top_level)/max(len(comments_df),1)*100:.1f}%)")

    log.info(f"\nFiles saved:")
    log.info(f"  {posts_path}")
    log.info(f"  {comments_path}")

    return posts_df, comments_df


if __name__ == "__main__":
    run(collect_comments_flag=True)