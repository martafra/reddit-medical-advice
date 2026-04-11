"""
LDA Topic Modelling for Reddit Medical Corpus
CS7IS4 / Text Analytics - Group 10

Runs LDA on three separate corpora:
  1. Medical posts
  2. Comments (on medical posts)
  3. Non-medical posts

For each corpus:
  - Finds the best number of topics using coherence score (c_v)
  - Trains the final LDA model
  - Saves top words per topic to CSV
  - Assigns each document its dominant topic
  - Saves a coherence plot showing how the topic count was chosen

Input:  output/reddit_posts_features.csv
        output/reddit_comments_features.csv
Output: output/lda/  (one subfolder per corpus)
"""

import logging
import warnings
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

import gensim
import gensim.corpora as corpora
from gensim.models import LdaMulticore, CoherenceModel

warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
log = logging.getLogger(__name__)


# ---
# CONFIGURATION
# ---
POSTS_CSV    = "output/reddit_posts_features.csv"
COMMENTS_CSV = "output/reddit_comments_features.csv"
OUTPUT_DIR   = Path("output/lda")

TOPIC_RANGE  = range(3, 16)  # test 3 to 15 topics
PASSES       = 10            # more passes = slower but better results
RANDOM_STATE = 42
WORKERS      = 3

MIN_TOKENS = 5  # skip docs shorter than this after preprocessing


# ---
# HELPERS
# ---
def to_token_list(text: str) -> list[str]:
    """text_processed is space-separated - just split it back for gensim."""
    if not isinstance(text, str) or not text.strip():
        return []
    return text.strip().split()


def find_optimal_topics(corpus, dictionary, texts,
                        topic_range, passes, random_state, workers,
                        label: str) -> tuple[int, list[float]]:
    """
    Trains one model per topic count, picks the one with the best coherence score.
    Returns (best_n_topics, all_scores).
    """
    log.info(f"  Testing {len(list(topic_range))} topic counts for {label}...")
    scores = []

    for n_topics in tqdm(topic_range, desc=f"  Coherence search ({label})"):
        model = LdaMulticore(
            corpus=corpus,
            id2word=dictionary,
            num_topics=n_topics,
            passes=passes,
            random_state=random_state,
            workers=workers,
        )
        coherence = CoherenceModel(
            model=model,
            texts=texts,
            dictionary=dictionary,
            coherence="c_v",
        ).get_coherence()
        scores.append(coherence)

    best_idx    = scores.index(max(scores))
    best_topics = list(topic_range)[best_idx]
    log.info(
        f"  Best topic count for {label}: {best_topics} "
        f"(coherence: {max(scores):.4f})"
    )
    return best_topics, scores


def save_coherence_plot(topic_range, scores, label: str, output_dir: Path):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(list(topic_range), scores, marker="o", linewidth=1.5, color="#4A72B0")
    best_idx = scores.index(max(scores))
    ax.axvline(list(topic_range)[best_idx], color="red", linestyle="--",
               label=f"Best: {list(topic_range)[best_idx]} topics")
    ax.set_title(f"Coherence score by number of topics - {label}")
    ax.set_xlabel("Number of topics")
    ax.set_ylabel("Coherence score (c_v)")
    ax.legend()
    plt.tight_layout()
    path = output_dir / "coherence_plot.png"
    plt.savefig(path, dpi=150)
    plt.close()
    log.info(f"  Coherence plot saved: {path}")


def save_topic_words(model, n_topics: int, output_dir: Path, n_words: int = 15):
    rows = []
    for topic_id in range(n_topics):
        words = model.show_topic(topic_id, topn=n_words)
        for rank, (word, prob) in enumerate(words, 1):
            rows.append({
                "topic_id":  topic_id,
                "rank":      rank,
                "word":      word,
                "probability": round(prob, 5),
            })
    df = pd.DataFrame(rows)
    path = output_dir / "topic_words.csv"
    df.to_csv(path, index=False)
    log.info(f"  Topic words saved: {path}")

    log.info(f"\n  Top words per topic:")
    for topic_id in range(n_topics):
        top = df[df["topic_id"] == topic_id]["word"].tolist()[:10]
        log.info(f"    Topic {topic_id:2d}: {', '.join(top)}")


def assign_topics(df: pd.DataFrame, model, corpus,
                  output_dir: Path, id_col: str) -> pd.DataFrame:
    """Adds dominant_topic and topic_probability columns, saves to CSV."""
    dominant_topics = []
    topic_probs     = []

    for doc_bow in corpus:
        topic_dist = model.get_document_topics(doc_bow, minimum_probability=0)
        best       = max(topic_dist, key=lambda x: x[1])
        dominant_topics.append(best[0])
        topic_probs.append(round(best[1], 4))

    df = df.copy()
    df["dominant_topic"]    = dominant_topics
    df["topic_probability"] = topic_probs

    path = output_dir / "documents_with_topics.csv"
    df.to_csv(path, index=False)
    log.info(f"  Documents with topics saved: {path}")

    dist = df["dominant_topic"].value_counts().sort_index()
    log.info(f"\n  Document distribution across topics:")
    for topic_id, count in dist.items():
        log.info(f"    Topic {topic_id:2d}: {count:>5} documents ({count/len(df)*100:.1f}%)")

    return df


# ---
# MAIN PIPELINE FOR ONE CORPUS
# ---
def run_lda(df: pd.DataFrame, label: str, id_col: str, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"\nLDA: {label} ({len(df):,} documents)")

    texts = df["text_processed"].apply(to_token_list).tolist()

    # drop very short docs
    before = len(texts)
    texts  = [t for t in texts if len(t) >= MIN_TOKENS]
    df     = df[df["text_processed"].apply(
        lambda x: len(to_token_list(x)) >= MIN_TOKENS
    )].reset_index(drop=True)

    removed = before - len(texts)
    if removed > 0:
        log.info(f"  Removed {removed} documents with fewer than {MIN_TOKENS} tokens")
    log.info(f"  Documents for LDA: {len(texts):,}")

    dictionary = corpora.Dictionary(texts)

    # drop words in <5 docs or in >50% of docs
    dictionary.filter_extremes(no_below=5, no_above=0.5)
    log.info(f"  Dictionary size after filtering: {len(dictionary):,} tokens")

    bow_corpus = [dictionary.doc2bow(text) for text in texts]

    best_n, scores = find_optimal_topics(
        bow_corpus, dictionary, texts,
        TOPIC_RANGE, PASSES, RANDOM_STATE, WORKERS, label
    )

    save_coherence_plot(TOPIC_RANGE, scores, label, output_dir)

    log.info(f"\n  Training final LDA model with {best_n} topics...")
    final_model = LdaMulticore(
        corpus=bow_corpus,
        id2word=dictionary,
        num_topics=best_n,
        passes=PASSES,
        random_state=RANDOM_STATE,
        workers=WORKERS,
    )

    save_topic_words(final_model, best_n, output_dir)

    df = assign_topics(df, final_model, bow_corpus, output_dir, id_col)

    # save model files (needed for pyLDAvis or inference on new data)
    model_path = output_dir / "lda_model"
    final_model.save(str(model_path))
    dictionary.save(str(output_dir / "dictionary.gensim"))
    log.info(f"  Model saved: {model_path}")

    return df, final_model, dictionary


# ---
# MAIN
# ---
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Loading posts...")
    posts = pd.read_csv(POSTS_CSV)
    log.info(f"  {len(posts):,} posts loaded")

    log.info("Loading comments...")
    comments = pd.read_csv(COMMENTS_CSV)
    log.info(f"  {len(comments):,} comments loaded")

    # split posts by category
    medical_cats    = ["chronic_mental", "chronic_physical", "acute_mental", "acute_physical"]
    nonmedical_cats = ["non_medical"]

    medical_posts    = posts[posts["category"].isin(medical_cats)].copy()
    nonmedical_posts = posts[posts["category"].isin(nonmedical_cats)].copy()

    log.info(f"\nMedical posts:     {len(medical_posts):,}")
    log.info(f"Non-medical posts: {len(nonmedical_posts):,}")
    log.info(f"Comments:          {len(comments):,}")

    run_lda(medical_posts,    "medical_posts",    "post_id",    OUTPUT_DIR / "medical_posts")
    run_lda(comments,         "comments",         "comment_id", OUTPUT_DIR / "comments")
    run_lda(nonmedical_posts, "nonmedical_posts", "post_id",    OUTPUT_DIR / "nonmedical_posts")

    log.info("\nDone. Output in output/lda/{medical_posts,comments,nonmedical_posts}/")


if __name__ == "__main__":
    main()
