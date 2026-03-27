"""
03_preprocess.py - Text Preprocessing for Reddit Medical Corpus
CS7NS6 / Text Analytics - Group H

What this script does, step by step:
  1. Loads the raw posts and comments CSVs
  2. Cleans the text (removes URLs, special chars, Reddit formatting)
  3. Lowercases everything
  4. Tokenizes into words
  5. Removes stopwords (standard English + custom Reddit noise words)
  6. Lemmatizes tokens using WordNetLemmatizer
  7. Saves two new CSVs with added columns:
       - text_clean     : cleaned raw text (no URLs, no markdown, lowercased)
       - text_processed : fully preprocessed text (joined tokens, ready for LDA/TF-IDF)
       - tokens         : space-separated lemmatized tokens

Why we keep both text_clean and text_processed:
  - text_clean is good for sentiment analysis (VADER works better on readable text)
  - text_processed is good for LDA and TF-IDF (bag-of-words style)

Dependencies:
  pip install pandas nltk tqdm
  python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('punkt_tab'); nltk.download('averaged_perceptron_tagger_eng')"
"""

import re
import logging
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from pathlib import Path
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
log = logging.getLogger(__name__)

# enable tqdm for pandas apply
tqdm.pandas()


# ---------------------------------------------------------------------------
# CONFIGURATION - change paths here if needed
# ---------------------------------------------------------------------------
POSTS_CSV    = "output/reddit_posts_20260324_080517.csv"
COMMENTS_CSV = "output/reddit_comments_20260324_080517.csv"
OUTPUT_DIR   = Path("output")

# ---------------------------------------------------------------------------
# CUSTOM STOPWORDS
# These are words that appear a lot on Reddit but carry no useful meaning
# for our analysis - things like "edit", "reddit", "post", etc.
# ---------------------------------------------------------------------------
REDDIT_NOISE_WORDS = {
    "edit", "update", "tldr", "tl", "dr", "reddit", "post", "comment",
    "thread", "subreddit", "upvote", "downvote", "crosspost", "repost",
    "op", "oc", "removed", "deleted", "mod", "moderator",
    "thanks", "thank", "please", "sorry", "hi", "hey", "hello",
    "yes", "no", "okay", "ok", "lol", "lmao", "omg", "tbh", "imo",
    "idk", "iirc", "afaik", "iiuc", "fwiw", "fyi", "btw",
    "also", "really", "actually", "basically", "literally",
    "going", "got", "get", "know", "think", "feel", "want",
    "would", "could", "should", "might", "may",
    "x200b", "x200", "amp", "nbsp",
}


# ---------------------------------------------------------------------------
# DOWNLOAD NLTK RESOURCES (only if not already downloaded)
# ---------------------------------------------------------------------------
def download_nltk_resources():
    resources = [
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords",    "stopwords"),
        ("corpora/wordnet",      "wordnet"),
        ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
    ]
    for path, name in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            log.info(f"Downloading NLTK resource: {name}")
            nltk.download(name, quiet=True)


# ---------------------------------------------------------------------------
# TEXT CLEANING
# ---------------------------------------------------------------------------
def clean_text(text: str) -> str:
    """
    Removes noise from raw Reddit text while keeping the actual words.
    We do this before tokenizing so the tokenizer sees clean input.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # remove URLs - they're not useful for NLP analysis
    text = re.sub(r"http\S+|www\.\S+", "", text)

    # remove Reddit markdown: **bold**, *italic*, ~~strikethrough~~, >quote
    text = re.sub(r"\*{1,2}(.*?)\*{1,2}", r"\1", text)
    text = re.sub(r"~~(.*?)~~", r"\1", text)
    text = re.sub(r"^>.*$", "", text, flags=re.MULTILINE)

    # remove subreddit and user mentions like r/depression or u/username
    text = re.sub(r"r/\w+|u/\w+", "", text)

    # remove special characters but keep apostrophes (they're part of words)
    text = re.sub(r"[^a-zA-Z0-9\s']", " ", text)

    # collapse multiple spaces into one
    text = re.sub(r"\s+", " ", text).strip()

    return text.lower()


# ---------------------------------------------------------------------------
# TOKENIZATION, STOPWORD REMOVAL, LEMMATIZATION
# ---------------------------------------------------------------------------
def preprocess_text(text: str,
                    stop_words: set,
                    lemmatizer: WordNetLemmatizer) -> tuple[str, str]:
    """
    Takes a cleaned text string and returns:
      - processed_text : space-joined lemmatized tokens (for LDA / TF-IDF)
      - tokens_str     : same thing (kept separate for clarity in the CSV)

    We only keep tokens that are:
      - alphabetic (no numbers, no punctuation leftovers)
      - longer than 2 characters (removes "it", "is", "a", etc.)
      - not in our combined stopword list
    """
    if not text:
        return "", ""

    tokens = word_tokenize(text)

    # filter and lemmatize
    filtered = []
    for token in tokens:
        if not token.isalpha():
            continue
        if len(token) <= 2:
            continue
        if token in stop_words:
            continue
        lemma = lemmatizer.lemmatize(token)
        filtered.append(lemma)

    joined = " ".join(filtered)
    return joined, joined


# ---------------------------------------------------------------------------
# MAIN PROCESSING FUNCTION
# ---------------------------------------------------------------------------
def process_dataframe(df: pd.DataFrame,
                      text_col: str,
                      stop_words: set,
                      lemmatizer: WordNetLemmatizer,
                      label: str) -> pd.DataFrame:
    """
    Applies the full preprocessing pipeline to a dataframe.
    Adds three new columns: text_clean, text_processed, tokens.
    """
    log.info(f"Cleaning {label} text...")
    df["text_clean"] = df[text_col].progress_apply(clean_text)

    log.info(f"Tokenizing, removing stopwords, lemmatizing {label}...")
    results = df["text_clean"].progress_apply(
        lambda t: preprocess_text(t, stop_words, lemmatizer)
    )

    df["text_processed"] = results.apply(lambda x: x[0])
    df["tokens"]         = results.apply(lambda x: x[1])

    # flag rows where preprocessing left us with almost nothing
    empty_after = (df["text_processed"].str.split().str.len() < 3).sum()
    if empty_after > 0:
        log.warning(
            f"  {empty_after} {label} have fewer than 3 tokens after preprocessing "
            f"- consider filtering them out before LDA/clustering."
        )

    return df


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    download_nltk_resources()

    # build combined stopword list
    english_stopwords = set(stopwords.words("english"))
    all_stopwords     = english_stopwords | REDDIT_NOISE_WORDS
    lemmatizer        = WordNetLemmatizer()

    log.info(f"Stopwords: {len(english_stopwords)} English + "
             f"{len(REDDIT_NOISE_WORDS)} Reddit-specific = {len(all_stopwords)} total")

    # --- POSTS ---
    log.info("\nLoading posts...")
    posts = pd.read_csv(POSTS_CSV)
    log.info(f"  {len(posts):,} posts loaded")

    posts = process_dataframe(posts, "text", all_stopwords, lemmatizer, "posts")

    posts_out = OUTPUT_DIR / "reddit_posts_preprocessed.csv"
    posts.to_csv(posts_out, index=False, encoding="utf-8")
    log.info(f"  Saved to: {posts_out}")

    # quick sanity check - show a before/after example
    sample = posts[posts["text_processed"].str.len() > 10].iloc[0]
    log.info(f"\n  Example post before: {sample['text'][:120]}...")
    log.info(f"  Example post after:  {sample['text_processed'][:120]}...")

    # --- COMMENTS ---
    log.info("\nLoading comments...")
    comments = pd.read_csv(COMMENTS_CSV)
    log.info(f"  {len(comments):,} comments loaded")

    comments = process_dataframe(comments, "text", all_stopwords, lemmatizer, "comments")

    comments_out = OUTPUT_DIR / "reddit_comments_preprocessed.csv"
    comments.to_csv(comments_out, index=False, encoding="utf-8")
    log.info(f"  Saved to: {comments_out}")

    # --- SUMMARY ---
    log.info("\n" + "=" * 55)
    log.info("PREPROCESSING COMPLETE")
    log.info("=" * 55)
    log.info(f"Posts processed:    {len(posts):,}")
    log.info(f"Comments processed: {len(comments):,}")
    log.info(f"\nNew columns added to both files:")
    log.info("  text_clean     - cleaned text, lowercase, no URLs/markdown")
    log.info("                   (use this for VADER sentiment analysis)")
    log.info("  text_processed - lemmatized tokens, stopwords removed")
    log.info("                   (use this for LDA and TF-IDF)")
    log.info("  tokens         - same as text_processed (explicit token column)")

    log.info(f"\nOutput files:")
    log.info(f"  {posts_out}")
    log.info(f"  {comments_out}")


if __name__ == "__main__":
    main()