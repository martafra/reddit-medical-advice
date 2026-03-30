"""
K-Means Clustering on TF-IDF for Reddit Medical Corpus
CS7IS4 / Text Analytics - Group 10

Groups posts by semantic similarity using TF-IDF + K-Means.

For each corpus the script:
  1. Builds a TF-IDF matrix from text_processed
  2. Finds the optimal number of clusters using the elbow method (inertia)
     and silhouette score
  3. Trains the final K-Means model
  4. Assigns each document to a cluster
  5. Inspects each cluster: most frequent words + dominant LDA topic (if available)
  6. Saves results and plots

Input:  output/reddit_posts_features.csv
        output/reddit_comments_features.csv
        output/lda/medical_posts/documents_with_topics.csv    (optional)
        output/lda/comments/documents_with_topics.csv         (optional)
        output/lda/nonmedical_posts/documents_with_topics.csv (optional)

Output: output/clustering/  (one subfolder per corpus)
"""

import logging
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
POSTS_CSV    = "output/reddit_posts_features.csv"
COMMENTS_CSV = "output/reddit_comments_features.csv"
OUTPUT_DIR   = Path("output/clustering")

# LDA topic assignment files - used to enrich cluster descriptions
LDA_POSTS_CSV       = "output/lda/medical_posts/documents_with_topics.csv"
LDA_COMMENTS_CSV    = "output/lda/comments/documents_with_topics.csv"
LDA_NONMEDICAL_CSV  = "output/lda/nonmedical_posts/documents_with_topics.csv"

# Range of cluster counts to evaluate
CLUSTER_RANGE = range(3, 12)   # test 3 to 11 clusters

# TF-IDF parameters
MAX_FEATURES  = 10000   # vocabulary size cap
MIN_DF        = 5       # ignore words appearing in fewer than 5 documents
MAX_DF        = 0.5     # ignore words appearing in more than 50% of documents
NGRAM_RANGE   = (1, 2)  # unigrams and bigrams

# SVD dimensionality reduction before K-Means (speeds up clustering)
# Using LSA (Latent Semantic Analysis) = TF-IDF + SVD
N_COMPONENTS  = 100     # reduce to 100 dimensions

RANDOM_STATE  = 42
N_WORDS_PER_CLUSTER = 15   # top words to show per cluster


# ---------------------------------------------------------------------------
# HELPER: build TF-IDF + LSA pipeline
# ---------------------------------------------------------------------------
def build_tfidf_matrix(texts: list[str]) -> tuple:
    """TF-IDF + SVD (LSA). Returns the reduced matrix, vectorizer, svd model, and raw tfidf matrix."""
    log.info("  Building TF-IDF matrix...")
    vectorizer = TfidfVectorizer(
        max_features=MAX_FEATURES,
        min_df=MIN_DF,
        max_df=MAX_DF,
        ngram_range=NGRAM_RANGE,
        sublinear_tf=True,   # log-scale term frequencies
    )
    tfidf_matrix = vectorizer.fit_transform(texts)
    log.info(f"  TF-IDF matrix: {tfidf_matrix.shape[0]} docs × {tfidf_matrix.shape[1]} terms")

    # reduce dimensions with SVD then normalise (LSA)
    log.info(f"  Reducing to {N_COMPONENTS} dimensions with SVD...")
    svd  = TruncatedSVD(n_components=N_COMPONENTS, random_state=RANDOM_STATE)
    lsa  = make_pipeline(svd, Normalizer(copy=False))
    X    = lsa.fit_transform(tfidf_matrix)
    log.info(f"  LSA matrix: {X.shape}")

    return X, vectorizer, svd, tfidf_matrix


# ---------------------------------------------------------------------------
# HELPER: elbow + silhouette search
# ---------------------------------------------------------------------------
def find_optimal_clusters(X: np.ndarray, cluster_range,
                          label: str, output_dir: Path) -> int:
    """Tries each cluster count, picks the K with the highest silhouette score."""
    log.info(f"  Searching optimal cluster count for {label}...")
    inertias    = []
    silhouettes = []

    for k in tqdm(cluster_range, desc=f"  Cluster search ({label})"):
        km = MiniBatchKMeans(n_clusters=k, random_state=RANDOM_STATE,
                             n_init=10, batch_size=1024)
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)
        sil = silhouette_score(X, labels, sample_size=min(5000, len(labels)))
        silhouettes.append(sil)

    best_idx = silhouettes.index(max(silhouettes))
    best_k   = list(cluster_range)[best_idx]

    log.info(f"  Best cluster count for {label}: {best_k} "
             f"(silhouette: {max(silhouettes):.4f})")

    # plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(list(cluster_range), inertias, marker="o", linewidth=1.5, color="#4A72B0")
    axes[0].set_title(f"Elbow method - {label}")
    axes[0].set_xlabel("Number of clusters")
    axes[0].set_ylabel("Inertia")
    axes[0].axvline(best_k, color="red", linestyle="--", label=f"Best: {best_k}")
    axes[0].legend()

    axes[1].plot(list(cluster_range), silhouettes, marker="o", linewidth=1.5, color="#E06C4A")
    axes[1].set_title(f"Silhouette score - {label}")
    axes[1].set_xlabel("Number of clusters")
    axes[1].set_ylabel("Silhouette score")
    axes[1].axvline(best_k, color="red", linestyle="--", label=f"Best: {best_k}")
    axes[1].legend()

    plt.tight_layout()
    path = output_dir / "cluster_search_plot.png"
    plt.savefig(path, dpi=150)
    plt.close()
    log.info(f"  Cluster search plot saved: {path}")

    return best_k


# ---------------------------------------------------------------------------
# HELPER: get top TF-IDF words for each cluster
# ---------------------------------------------------------------------------
def get_cluster_top_words(km_model, vectorizer, n_words: int) -> dict[int, list[str]]:
    """Top n_words per cluster based on centroid weights in TF-IDF space."""
    feature_names = vectorizer.get_feature_names_out()
    top_words     = {}

    for cluster_id, centroid in enumerate(km_model.cluster_centers_):
        top_indices = centroid.argsort()[::-1][:n_words]
        top_words[cluster_id] = [feature_names[i] for i in top_indices]

    return top_words


# ---------------------------------------------------------------------------
# MAIN PIPELINE FOR ONE CORPUS
# ---------------------------------------------------------------------------
def run_clustering(df: pd.DataFrame, label: str, id_col: str,
                   output_dir: Path, lda_csv: str = None):
    output_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"\n{'='*55}")
    log.info(f"Clustering: {label} ({len(df):,} documents)")
    log.info(f"{'='*55}")

    # filter empty text
    df = df[df["text_processed"].notna() & (df["text_processed"].str.strip() != "")].copy()
    df = df.reset_index(drop=True)
    log.info(f"  Documents after filtering empty text: {len(df):,}")

    texts = df["text_processed"].tolist()

    # build TF-IDF + LSA matrix
    X, vectorizer, svd, tfidf_matrix = build_tfidf_matrix(texts)

    # find optimal cluster count
    best_k = find_optimal_clusters(X, CLUSTER_RANGE, label, output_dir)

    # train final K-Means model
    log.info(f"\n  Training final K-Means with {best_k} clusters...")
    km = KMeans(n_clusters=best_k, random_state=RANDOM_STATE,
                n_init=20, max_iter=300)
    df["cluster"] = km.fit_predict(X)

    # get top words per cluster from original TF-IDF space
    # re-fit vectorizer on the same texts to get centroid words
    log.info("  Extracting top words per cluster...")
    tfidf_full = TfidfVectorizer(
        max_features=MAX_FEATURES, min_df=MIN_DF, max_df=MAX_DF,
        ngram_range=NGRAM_RANGE, sublinear_tf=True,
    )
    tfidf_full.fit(texts)

    # compute per-cluster centroids in TF-IDF space
    cluster_centroids = np.zeros((best_k, len(tfidf_full.get_feature_names_out())))
    for cluster_id in range(best_k):
        cluster_texts = [texts[i] for i, c in enumerate(df["cluster"]) if c == cluster_id]
        if cluster_texts:
            cluster_matrix = tfidf_full.transform(cluster_texts)
            cluster_centroids[cluster_id] = cluster_matrix.mean(axis=0)

    feature_names = tfidf_full.get_feature_names_out()
    top_words_rows = []
    log.info(f"\n  Top words per cluster:")
    for cluster_id in range(best_k):
        top_idx   = cluster_centroids[cluster_id].argsort()[::-1][:N_WORDS_PER_CLUSTER]
        top_words = [feature_names[i] for i in top_idx]
        log.info(f"    Cluster {cluster_id:2d} ({(df['cluster']==cluster_id).sum()} docs): "
                 f"{', '.join(top_words[:8])}")
        for rank, word in enumerate(top_words, 1):
            top_words_rows.append({
                "cluster_id": cluster_id,
                "rank":       rank,
                "word":       word,
                "tfidf_weight": round(cluster_centroids[cluster_id][
                    list(feature_names).index(word)
                ], 5),
            })

    cluster_words_df = pd.DataFrame(top_words_rows)
    words_path = output_dir / "cluster_words.csv"
    cluster_words_df.to_csv(words_path, index=False)
    log.info(f"\n  Cluster words saved: {words_path}")

    # merge with LDA topic assignments if available
    if lda_csv and Path(lda_csv).exists():
        log.info("  Merging with LDA topic assignments...")
        lda_df = pd.read_csv(lda_csv)[[id_col, "dominant_topic"]].rename(
            columns={"dominant_topic": "lda_topic"}
        )
        df = df.merge(lda_df, on=id_col, how="left")

        # show cluster–topic overlap
        if "lda_topic" in df.columns:
            log.info("\n  Cluster vs LDA topic overlap:")
            overlap = df.groupby(["cluster", "lda_topic"]).size().unstack(fill_value=0)
            log.info(f"\n{overlap.to_string()}")

    # cluster size distribution
    dist = df["cluster"].value_counts().sort_index()
    log.info(f"\n  Cluster size distribution:")
    for cluster_id, count in dist.items():
        log.info(f"    Cluster {cluster_id:2d}: {count:>5} docs ({count/len(df)*100:.1f}%)")

    # save documents with cluster assignments
    docs_path = output_dir / "documents_with_clusters.csv"
    df.to_csv(docs_path, index=False)
    log.info(f"\n  Documents with clusters saved: {docs_path}")

    # cluster size bar chart
    fig, ax = plt.subplots(figsize=(8, 4))
    dist.plot(kind="bar", ax=ax, color=sns.color_palette("muted", len(dist)))
    ax.set_title(f"Cluster size distribution - {label}")
    ax.set_xlabel("Cluster ID")
    ax.set_ylabel("Number of documents")
    ax.tick_params(axis="x", rotation=0)
    plt.tight_layout()
    plot_path = output_dir / "cluster_distribution.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    log.info(f"  Distribution plot saved: {plot_path}")

    return df


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Loading posts...")
    posts = pd.read_csv(POSTS_CSV)
    log.info(f"  {len(posts):,} posts loaded")

    log.info("Loading comments...")
    comments = pd.read_csv(COMMENTS_CSV)
    log.info(f"  {len(comments):,} comments loaded")

    medical_cats = ["chronic_mental", "chronic_physical", "acute_mental", "acute_physical"]
    medical_posts    = posts[posts["category"].isin(medical_cats)].copy()
    nonmedical_posts = posts[posts["category"] == "non_medical"].copy()

    log.info(f"\nMedical posts:     {len(medical_posts):,}")
    log.info(f"Non-medical posts: {len(nonmedical_posts):,}")
    log.info(f"Comments:          {len(comments):,}")

    run_clustering(
        medical_posts, "medical_posts", "post_id",
        OUTPUT_DIR / "medical_posts", LDA_POSTS_CSV
    )
    run_clustering(
        comments, "comments", "comment_id",
        OUTPUT_DIR / "comments", LDA_COMMENTS_CSV
    )
    run_clustering(
        nonmedical_posts, "nonmedical_posts", "post_id",
        OUTPUT_DIR / "nonmedical_posts", LDA_NONMEDICAL_CSV
    )

    log.info("\n" + "=" * 55)
    log.info("CLUSTERING COMPLETE")
    log.info("=" * 55)
    log.info("Output folders:")
    log.info("  output/clustering/medical_posts/")
    log.info("  output/clustering/comments/")
    log.info("  output/clustering/nonmedical_posts/")
    log.info("\nEach folder contains:")
    log.info("  cluster_search_plot.png      - elbow + silhouette curves")
    log.info("  cluster_words.csv            - top TF-IDF words per cluster")
    log.info("  cluster_distribution.png     - cluster size bar chart")
    log.info("  documents_with_clusters.csv  - each doc assigned to a cluster")


if __name__ == "__main__":
    main()