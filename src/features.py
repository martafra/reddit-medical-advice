"""
Feature Engineering for Reddit Medical Corpus
CS7IS4 / Text Analytics - Group 10

Computes 6 features for each post and comment:
  1. Sentiment Score (VADER)        compound score, -1 to +1
  2. Shared Experience Flag (0/1)   does the author mention personal experience?
  3. Advice Acceptance Score        how much does the reader find the advice useful?
  4. Empathy Phrase Score           emotional support phrases per 100 words
  5. Lexical Density                proportion of content words (nouns, verbs, adj, adv)
  6. Uncertainty Vocabulary Density hedging words like "maybe", "I think", "might"

Input:  output/reddit_posts_preprocessed.csv
        output/reddit_comments_preprocessed.csv
Output: output/reddit_posts_features.csv
        output/reddit_comments_features.csv
"""

import re
import logging
import pandas as pd
import nltk
from nltk import pos_tag, word_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from tqdm import tqdm
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
log = logging.getLogger(__name__)
tqdm.pandas()


# ---
# CONFIGURATION
# ---
POSTS_CSV    = "output/reddit_posts_preprocessed.csv"
COMMENTS_CSV = "output/reddit_comments_preprocessed.csv"
OUTPUT_DIR   = Path("output")


# ---
# PHRASE LISTS
# ---

SHARED_EXPERIENCE_PHRASES = [
    "same thing happened to me",
    "i went through the same",
    "i had the same",
    "i also had",
    "i experienced the same",
    "been there",
    "i know how you feel",
    "i felt the same",
    "in my experience",
    "what worked for me",
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

ADVICE_ACCEPTANCE_PHRASES = [
    "that really helped",
    "this helped",
    "thank you so much",
    "i'll try that",
    "i will try",
    "going to try",
    "good point",
    "that makes sense",
    "you're right",
    "you are right",
    "i agree",
    "that's helpful",
    "this is helpful",
    "great advice",
    "really useful",
    "i appreciate",
    "that's a good idea",
    "i hadn't thought of that",
    "i never thought of",
    "i'll definitely",
    "following this advice",
    "took your advice",
]

EMPATHY_PHRASES = [
    "i'm so sorry",
    "i am so sorry",
    "that sounds really hard",
    "that must be so hard",
    "that must be difficult",
    "you're not alone",
    "you are not alone",
    "i hear you",
    "sending hugs",
    "sending love",
    "you're doing so well",
    "you're so strong",
    "proud of you",
    "it's okay to feel",
    "your feelings are valid",
    "you deserve",
    "you matter",
    "i care about you",
    "here for you",
    "rooting for you",
    "hang in there",
    "you've got this",
    "things will get better",
    "it gets better",
]

UNCERTAINTY_WORDS = [
    "maybe", "perhaps", "possibly", "probably", "might", "could",
    "uncertain", "unsure", "not sure", "i think", "i believe",
    "it seems", "it appears", "kind of", "sort of", "i guess",
    "i suppose", "apparently", "presumably", "theoretically",
    "potentially", "allegedly", "supposedly", "roughly", "approximately",
]

# POS tags that count as content words
CONTENT_POS_TAGS = {"NN", "NNS", "NNP", "NNPS",   # nouns
                    "VB", "VBD", "VBG", "VBN",      # verbs
                    "VBP", "VBZ",
                    "JJ", "JJR", "JJS",              # adjectives
                    "RB", "RBR", "RBS"}              # adverbs


# ---
# NLTK SETUP
# ---
def download_nltk_resources():
    for path, name in [
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
    ]:
        try:
            nltk.data.find(path)
        except LookupError:
            log.info(f"Downloading NLTK resource: {name}")
            nltk.download(name, quiet=True)


# ---
# FEATURE FUNCTIONS
# ---

vader = None  # set in main()

def sentiment_score(text: str) -> float:
    """VADER compound score (-1 to +1). Run on text_clean so punctuation/casing are preserved."""
    if not isinstance(text, str) or not text.strip():
        return 0.0
    return vader.polarity_scores(text)["compound"]


def shared_experience_flag(text: str) -> int:
    """1 if the text has at least one shared-experience phrase, else 0."""
    if not isinstance(text, str):
        return 0
    text_lower = text.lower()
    return int(any(phrase in text_lower for phrase in SHARED_EXPERIENCE_PHRASES))


def advice_acceptance_score(text: str) -> float:
    """Advice-acceptance phrase count per 100 words. Normalized so longer texts don't dominate."""
    if not isinstance(text, str) or not text.strip():
        return 0.0
    text_lower  = text.lower()
    word_count  = max(len(text.split()), 1)
    phrase_hits = sum(text_lower.count(phrase) for phrase in ADVICE_ACCEPTANCE_PHRASES)
    return round((phrase_hits / word_count) * 100, 4)


def empathy_score(text: str) -> float:
    """Empathy phrase count per 100 words. Same normalization as advice_acceptance_score."""
    if not isinstance(text, str) or not text.strip():
        return 0.0
    text_lower  = text.lower()
    word_count  = max(len(text.split()), 1)
    phrase_hits = sum(text_lower.count(phrase) for phrase in EMPATHY_PHRASES)
    return round((phrase_hits / word_count) * 100, 4)


def lexical_density(text: str) -> float:
    """
    Ratio of content words to total tokens (POS-tagged).
    High (~0.6+) = dense/technical, low (~0.3) = conversational.
    """
    if not isinstance(text, str) or not text.strip():
        return 0.0
    tokens = word_tokenize(text)
    if not tokens:
        return 0.0
    tagged        = pos_tag(tokens)
    content_count = sum(1 for _, tag in tagged if tag in CONTENT_POS_TAGS)
    return round(content_count / len(tokens), 4)


def uncertainty_density(text: str) -> float:
    """Hedging/uncertainty word count per 100 words."""
    if not isinstance(text, str) or not text.strip():
        return 0.0
    text_lower = text.lower()
    word_count = max(len(text.split()), 1)
    hits       = sum(text_lower.count(word) for word in UNCERTAINTY_WORDS)
    return round((hits / word_count) * 100, 4)


# ---
# APPLY ALL FEATURES TO A DATAFRAME
# ---
def compute_features(df: pd.DataFrame, text_col: str,
                     clean_col: str, label: str) -> pd.DataFrame:
    """
    Adds all six feature columns to df.
    text_col  = column used for phrase-based features
    clean_col = column used for VADER (needs punctuation/casing)
    """
    log.info(f"\nComputing features for {label} ({len(df):,} rows)...")

    log.info("  1/6 Sentiment score (VADER)...")
    df["sentiment_score"] = df[clean_col].progress_apply(sentiment_score)

    log.info("  2/6 Shared experience flag...")
    df["shared_experience"] = df[text_col].progress_apply(shared_experience_flag)

    log.info("  3/6 Advice acceptance score...")
    df["advice_acceptance"] = df[text_col].progress_apply(advice_acceptance_score)

    log.info("  4/6 Empathy phrase score...")
    df["empathy_score"] = df[text_col].progress_apply(empathy_score)

    log.info("  5/6 Lexical density...")
    df["lexical_density"] = df[clean_col].progress_apply(lexical_density)

    log.info("  6/6 Uncertainty density...")
    df["uncertainty_density"] = df[text_col].progress_apply(uncertainty_density)

    log.info(f"\n  Feature summary for {label}:")
    feature_cols = [
        "sentiment_score", "shared_experience", "advice_acceptance",
        "empathy_score", "lexical_density", "uncertainty_density",
    ]
    summary = df[feature_cols].describe().loc[["mean", "std", "min", "max"]]
    log.info(f"\n{summary.to_string()}")

    return df


# ---
# MAIN
# ---
def main():
    global vader
    download_nltk_resources()
    vader = SentimentIntensityAnalyzer()

    # --- POSTS ---
    log.info("Loading preprocessed posts...")
    posts = pd.read_csv(POSTS_CSV)
    log.info(f"  {len(posts):,} posts loaded")

    posts = compute_features(posts, text_col="text_clean",
                             clean_col="text_clean", label="posts")

    posts_out = OUTPUT_DIR / "reddit_posts_features.csv"
    posts.to_csv(posts_out, index=False, encoding="utf-8")
    log.info(f"\nPosts saved to: {posts_out}")

    # --- COMMENTS ---
    log.info("\nLoading preprocessed comments...")
    comments = pd.read_csv(COMMENTS_CSV)
    log.info(f"  {len(comments):,} comments loaded")

    comments = compute_features(comments, text_col="text_clean",
                                clean_col="text_clean", label="comments")

    comments_out = OUTPUT_DIR / "reddit_comments_features.csv"
    comments.to_csv(comments_out, index=False, encoding="utf-8")
    log.info(f"\nComments saved to: {comments_out}")

    log.info(f"\nDone. Posts: {len(posts):,} rows → {posts_out}")
    log.info(f"     Comments: {len(comments):,} rows → {comments_out}")


if __name__ == "__main__":
    main()
