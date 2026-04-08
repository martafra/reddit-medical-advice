"""
analysis.py  –  CS7IS4 / Text Analytics – Group 10

Analyses LDA topic-modelling and K-Means clustering outputs to answer
three Research Questions and test the associated hypotheses.

RQ1: How does the nature and severity of medical conditions influence
     how the patient reacts to the received advice?
RQ2: To what extent does the relatability effect of "shared experience"
     influence the acceptance of the advice?
RQ3: Do the linguistic patterns found in reactions to important medical
     advice also generalise to other areas and scenarios (e.g. financial)?

Run:
    python analysis.py

Fill in the PLACEHOLDER paths in SECTION 0 before running.
Topic/cluster naming CSVs are generated on first run; edit the
'manual_name' column in output/analysis/naming/ then re-run to apply
your custom names throughout the analysis.
"""

import warnings
import logging
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
log = logging.getLogger(__name__)

# =============================================================================
# SECTION 0 – CONFIGURATION  (fill in paths before running)
# =============================================================================

# --- LDA outputs (one set per corpus) ----------------------------------------
LDA_MEDICAL_DOCS         = "../Output - Reddit/lda/medical_posts/documents_with_topics.csv"
LDA_MEDICAL_WORDS        = "../Output - Reddit/lda/medical_posts/topic_words.csv"
LDA_NONMEDICAL_DOCS      = "../Output - Reddit/lda/nonmedical_posts/documents_with_topics.csv"
LDA_NONMEDICAL_WORDS     = "../Output - Reddit/lda/nonmedical_posts/topic_words.csv"
LDA_COMMENTS_DOCS        = "../Output - Reddit/lda/comments/documents_with_topics.csv"
LDA_COMMENTS_WORDS       = "../Output - Reddit/lda/comments/topic_words.csv"

# --- Clustering outputs (one set per corpus) ---------------------------------
CLUSTER_MEDICAL_DOCS     = "../Output - Reddit/clustering/medical_posts/documents_with_clusters.csv"
CLUSTER_MEDICAL_WORDS    = "../Output - Reddit/clustering/medical_posts/cluster_words.csv"
CLUSTER_NONMEDICAL_DOCS  = "../Output - Reddit/clustering/nonmedical_posts/documents_with_clusters.csv"
CLUSTER_NONMEDICAL_WORDS = "../Output - Reddit/clustering/nonmedical_posts/cluster_words.csv"
CLUSTER_COMMENTS_DOCS    = "../Output - Reddit/clustering/comments/documents_with_clusters.csv"
CLUSTER_COMMENTS_WORDS   = "../Output - Reddit/clustering/comments/cluster_words.csv"

# --- Output directory --------------------------------------------------------
OUTPUT_DIR = Path("output/analysis")

# --- Severity mapping (chronic = more severe than acute) ---------------------
SEVERITY_MAP = {
    "chronic_mental":   "chronic",
    "chronic_physical": "chronic",
    "acute_mental":     "acute",
    "acute_physical":   "acute",
}

NATURE_MAP = {
    "chronic_mental":   "mental",
    "chronic_physical": "physical",
    "acute_mental":     "mental",
    "acute_physical":   "physical",
}

# --- Additional phrase lists (extends features.py; used in RQ1 H2) ----------
# Clarity phrases – language indicating the advice was well communicated
CLARITY_PHRASES = [
    "clearly explained", "easy to understand", "makes sense", "well explained",
    "straightforward", "easy to follow", "very clear", "so clear",
    "clearly stated", "concise", "understandable",
]

# Accuracy phrases – language indicating the advice was factually sound
ACCURACY_PHRASES = [
    "accurate", "correct", "factually", "evidence based", "evidence-based",
    "proven", "verified", "credible", "reliable source", "backed by",
    "well researched", "well-researched", "scientifically",
]

# Number of top words to compare per topic for RQ3 Jaccard similarity
TOP_WORDS_FOR_SIMILARITY = 20

# Significance threshold
ALPHA = 0.05

# =============================================================================
# SECTION 1 – UTILITIES
# =============================================================================

def _is_placeholder(path: str) -> bool:
    return path == "PLACEHOLDER" or not Path(path).exists()


def get_text_column(df: pd.DataFrame) -> str:
    """Return the best available text column (prefers original/clean text over processed)."""
    for col in ("text_clean", "selftext", "body", "text", "text_processed"):
        if col in df.columns:
            return col
    raise ValueError(f"No text column found. Available columns: {list(df.columns)}")


def detect_id_col(df: pd.DataFrame) -> str:
    for col in ("post_id", "comment_id", "id"):
        if col in df.columns:
            return col
    return df.columns[0]


def _phrase_density(text: str, phrases: list) -> float:
    """Total phrase occurrences per 100 words. Returns 0.0 for empty text."""
    if not isinstance(text, str) or not text.strip():
        return 0.0
    text_lower = text.lower()
    word_count = max(len(text.split()), 1)
    hits = sum(text_lower.count(p) for p in phrases)
    return round((hits / word_count) * 100, 4)


def add_extra_features(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    """
    Add features not already in features.py:
      - clarity_density, accuracy_density  (RQ1 H2)
      - word_count                          (RQ3 H2)

    Columns already computed by features.py and present in documents_with_topics.csv:
      sentiment_score, empathy_score, shared_experience,
      advice_acceptance, uncertainty_density
    are used as-is and are NOT recomputed here.
    """
    t = df[text_col].fillna("")
    log.info("  Adding clarity_density and accuracy_density...")
    df["clarity_density"]  = t.apply(lambda x: _phrase_density(x, CLARITY_PHRASES))
    df["accuracy_density"] = t.apply(lambda x: _phrase_density(x, ACCURACY_PHRASES))
    df["word_count"]       = t.apply(lambda x: len(str(x).split()))
    return df


def load_and_prepare(docs_csv: str, label: str, cluster_docs_csv: str = None) -> pd.DataFrame:
    """
    Load a documents CSV (LDA or clustering output), compute fresh VADER sentiment,
    add extra phrase features, and optionally merge cluster assignments.
    """
    if _is_placeholder(docs_csv):
        log.warning(f"  Skipping '{label}': path not set or file not found ({docs_csv})")
        return pd.DataFrame()

    log.info(f"\nLoading {label}  ({docs_csv})...")
    df = pd.read_csv(docs_csv)
    log.info(f"  {len(df):,} documents")

    text_col = get_text_column(df)
    log.info(f"  Text column: '{text_col}'")

    df = add_extra_features(df, text_col)

    # Merge cluster assignments if provided
    if cluster_docs_csv and not _is_placeholder(cluster_docs_csv):
        id_col = detect_id_col(df)
        try:
            cl = pd.read_csv(cluster_docs_csv)[[id_col, "cluster"]]
            df = df.merge(cl, on=id_col, how="left")
            log.info(f"  Merged cluster assignments ({df['cluster'].notna().sum():,} matches)")
        except Exception as e:
            log.warning(f"  Could not merge clusters: {e}")

    return df


def fisher_z_test(r1: float, n1: int, r2: float, n2: int) -> tuple:
    """
    Fisher's Z-test: tests whether two independent correlations differ.
    Returns (z_statistic, p_value).
    """
    z1 = np.arctanh(np.clip(r1, -0.9999, 0.9999))
    z2 = np.arctanh(np.clip(r2, -0.9999, 0.9999))
    se = np.sqrt(1 / (n1 - 3) + 1 / (n2 - 3))
    z  = (z1 - z2) / se
    p  = 2 * (1 - stats.norm.cdf(abs(z)))
    return round(z, 4), round(p, 4)


def safe_spearman(x: pd.Series, y: pd.Series) -> tuple:
    """Spearman r with fallback for constant series."""
    mask = x.notna() & y.notna()
    x, y = x[mask], y[mask]
    if len(x) < 5 or x.std() == 0 or y.std() == 0:
        return float("nan"), float("nan")
    r, p = stats.spearmanr(x, y)
    return round(float(r), 4), round(float(p), 4)


# =============================================================================
# SECTION 2 – TOPIC AND CLUSTER NAMING
# =============================================================================

def _auto_name(top_words: list) -> str:
    """Join the first 3 words with underscores as an auto-generated topic label."""
    return "_".join(top_words[:3])


def generate_naming_file(words_csv: str, id_col: str, out_path: Path):
    """
    Read a topic_words.csv or cluster_words.csv, auto-generate names,
    and save a CSV that the user can edit to add manual names.
    """
    if _is_placeholder(words_csv):
        log.warning(f"  Skipping naming file for {out_path.name}: path not set")
        return

    df = pd.read_csv(words_csv)
    rows = []
    for tid, grp in df.sort_values([id_col, "rank"]).groupby(id_col):
        top_words = grp["word"].tolist()[:TOP_WORDS_FOR_SIMILARITY]
        rows.append({
            id_col:          tid,
            "auto_name":     _auto_name(top_words),
            "top_words":     ", ".join(top_words),
            "manual_name":   "",   # <-- edit this column to override auto_name
        })
    naming_df = pd.DataFrame(rows)
    naming_df.to_csv(out_path, index=False)
    log.info(f"  Saved: {out_path}")


def load_topic_names(naming_csv: str, id_col: str = "topic_id") -> dict:
    """
    Load a naming CSV. Uses manual_name if filled, otherwise auto_name.
    Returns {id: name_string}.
    """
    if not Path(naming_csv).exists():
        return {}
    df = pd.read_csv(naming_csv)
    out = {}
    for _, row in df.iterrows():
        manual = row.get("manual_name", "")
        name   = str(manual).strip() if pd.notna(manual) and str(manual).strip() else row["auto_name"]
        out[row[id_col]] = name
    return out


def save_all_naming_files(output_dir: Path):
    """Generate topic/cluster naming CSVs for all six word files."""
    naming_dir = output_dir / "naming"
    naming_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"\n{'='*55}")
    log.info("STEP 1: Generating topic / cluster naming files")
    log.info(f"{'='*55}")

    configs = [
        (LDA_MEDICAL_WORDS,        "topic_id",   "lda_medical_topics.csv"),
        (LDA_NONMEDICAL_WORDS,     "topic_id",   "lda_nonmedical_topics.csv"),
        (LDA_COMMENTS_WORDS,       "topic_id",   "lda_comments_topics.csv"),
        (CLUSTER_MEDICAL_WORDS,    "cluster_id", "cluster_medical.csv"),
        (CLUSTER_NONMEDICAL_WORDS, "cluster_id", "cluster_nonmedical.csv"),
        (CLUSTER_COMMENTS_WORDS,   "cluster_id", "cluster_comments.csv"),
    ]
    for words_csv, id_col, fname in configs:
        generate_naming_file(words_csv, id_col, naming_dir / fname)

    log.info("\n  → Edit the 'manual_name' column in output/analysis/naming/*.csv")
    log.info("    then re-run to apply your names throughout the analysis.\n")


# =============================================================================
# SECTION 3 – RQ1: SEVERITY AND NATURE ANALYSIS
# =============================================================================

def rq1_analysis(medical_df: pd.DataFrame, output_dir: Path):
    """
    RQ1: How does the nature and severity of medical conditions influence
         how the patient reacts to the received advice?

    H1: Posts about more severe conditions (chronic) will show a higher
        prevalence of empathy and emotional support phrases than posts
        about less severe conditions (acute).
        Test: Mann-Whitney U on empathy_score (chronic vs acute).

    H2: In severe (chronic) conditions, the empathy of the response will
        be more strongly correlated with positive sentiment than clarity
        or accuracy language.
        Test: Spearman correlations within chronic posts, compared across
        the three metrics.

    Input columns used (from features.py + extra):
      category, empathy_score, sentiment_score (or vader_compound),
      clarity_density, accuracy_density
    """
    if medical_df.empty:
        log.warning("RQ1: medical_df is empty – skipping")
        return

    rq1_dir = output_dir / "rq1"
    rq1_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"\n{'='*55}")
    log.info("STEP 3: RQ1 – Severity and Nature Analysis")
    log.info(f"{'='*55}")

    # ── Prepare: filter to medical categories and add severity/nature labels ──
    if "category" not in medical_df.columns:
        log.warning("  RQ1: 'category' column not found – cannot label severity. Skipping.")
        return

    df = medical_df[medical_df["category"].isin(SEVERITY_MAP)].copy()
    df["severity"] = df["category"].map(SEVERITY_MAP)
    df["nature"]   = df["category"].map(NATURE_MAP)

    n_chronic = (df["severity"] == "chronic").sum()
    n_acute   = (df["severity"] == "acute").sum()
    log.info(f"  Medical posts: {len(df):,}  (chronic={n_chronic}, acute={n_acute})")

    # Use pre-computed empathy_score when available; fall back to vader_compound
    sentiment_col = "sentiment_score" if "sentiment_score" in df.columns else "vader_compound"
    empathy_col   = "empathy_score"   if "empathy_score"   in df.columns else None

    if empathy_col is None:
        log.warning("  empathy_score column not found – RQ1 H1 cannot be tested")
        return

    # ─────────────────────────────────────────────────────────────────────────
    # H1: Empathy density  –  chronic vs acute
    # ─────────────────────────────────────────────────────────────────────────
    log.info("\n  -- H1: Empathy density in chronic vs acute posts --")

    chronic_emp = df.loc[df["severity"] == "chronic", empathy_col].dropna()
    acute_emp   = df.loc[df["severity"] == "acute",   empathy_col].dropna()

    u_stat, p_mw = stats.mannwhitneyu(chronic_emp, acute_emp, alternative="greater")
    # Rank-biserial correlation as effect size
    r_rb = 1 - (2 * u_stat) / (len(chronic_emp) * len(acute_emp))

    log.info(f"  Chronic  –  mean={chronic_emp.mean():.4f}, median={chronic_emp.median():.4f}, n={len(chronic_emp)}")
    log.info(f"  Acute    –  mean={acute_emp.mean():.4f}, median={acute_emp.median():.4f}, n={len(acute_emp)}")
    log.info(f"  Mann-Whitney U={u_stat:.1f}, p={p_mw:.4f}, effect size (r_rb)={r_rb:.4f}")
    log.info(f"  H1 → {'SUPPORTED' if p_mw < ALPHA else 'NOT SUPPORTED'} (α={ALPHA})")

    # Group stats table (severity × nature)
    group_stats = (
        df.groupby(["severity", "nature"])
        .agg(
            n=(empathy_col, "count"),
            mean_empathy=(empathy_col, "mean"),
            median_empathy=(empathy_col, "median"),
            mean_sentiment=(sentiment_col, "mean"),
        )
        .round(4)
        .reset_index()
    )
    group_stats.to_csv(rq1_dir / "h1_group_stats.csv", index=False)
    log.info(f"\n  Group stats:\n{group_stats.to_string(index=False)}")

    # H1 results text file
    with open(rq1_dir / "h1_results.txt", "w", encoding="utf-8") as f:
        f.write("H1: Empathy phrase density higher in chronic (more severe) posts than acute posts\n\n")
        f.write(f"Chronic: mean={chronic_emp.mean():.4f}, median={chronic_emp.median():.4f}, n={len(chronic_emp)}\n")
        f.write(f"Acute:   mean={acute_emp.mean():.4f}, median={acute_emp.median():.4f}, n={len(acute_emp)}\n")
        f.write(f"Mann-Whitney U = {u_stat:.1f}\n")
        f.write(f"p-value        = {p_mw:.4f}  (one-tailed, chronic > acute)\n")
        f.write(f"Effect size    = {r_rb:.4f}  (rank-biserial r; |r| > 0.1 small, > 0.3 medium, > 0.5 large)\n")
        f.write(f"Result         = {'SUPPORTED' if p_mw < ALPHA else 'NOT SUPPORTED'} at α={ALPHA}\n")

    # Boxplot: empathy and sentiment by severity × nature
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, col, title in zip(
        axes,
        [empathy_col, sentiment_col],
        ["Empathy Phrase Density (per 100 words)", "Sentiment Score"],
    ):
        sns.boxplot(
            data=df, x="severity", y=col, hue="nature",
            palette="muted", ax=ax, order=["chronic", "acute"],
        )
        ax.set_title(f"RQ1 H1 – {title}")
        ax.set_xlabel("Severity")
        ax.set_ylabel(col)
    plt.tight_layout()
    plt.savefig(rq1_dir / "h1_boxplots.png", dpi=150)
    plt.close()

    # ─────────────────────────────────────────────────────────────────────────
    # H2: Correlation comparison in chronic posts
    # ─────────────────────────────────────────────────────────────────────────
    log.info("\n  -- H2: Correlation with sentiment in chronic posts --")

    chronic_df = df[df["severity"] == "chronic"].copy()

    metrics = {
        "empathy_score":    empathy_col,
        "clarity_density":  "clarity_density",
        "accuracy_density": "accuracy_density",
    }
    corr_rows = []
    for label, col in metrics.items():
        if col not in chronic_df.columns:
            log.warning(f"  Column '{col}' not found – skipping")
            continue
        r, p = safe_spearman(chronic_df[col], chronic_df[sentiment_col])
        corr_rows.append({"metric": label, "column": col,
                          "spearman_r": r, "p_value": p,
                          "n": len(chronic_df), "significant": p < ALPHA if not np.isnan(p) else False})
        log.info(f"  {label}: r={r}, p={p}")

    corr_df = pd.DataFrame(corr_rows)
    corr_df.to_csv(rq1_dir / "h2_correlations.csv", index=False)

    # H2 verdict: is empathy r > both clarity r and accuracy r?
    if len(corr_df) == 3:
        emp_r  = corr_df.loc[corr_df["metric"] == "empathy_score",    "spearman_r"].values[0]
        cla_r  = corr_df.loc[corr_df["metric"] == "clarity_density",  "spearman_r"].values[0]
        acc_r  = corr_df.loc[corr_df["metric"] == "accuracy_density", "spearman_r"].values[0]
        h2_ok  = (not np.isnan(emp_r)) and emp_r > cla_r and emp_r > acc_r
        log.info(f"  H2 → {'SUPPORTED' if h2_ok else 'NOT SUPPORTED'}"
                 f" (empathy r={emp_r:.4f} vs clarity r={cla_r:.4f}, accuracy r={acc_r:.4f})")
        with open(rq1_dir / "h2_results.txt", "w", encoding="utf-8") as f:
            f.write("H2: Empathy more strongly correlated with positive sentiment than clarity/accuracy (chronic posts)\n\n")
            for _, row in corr_df.iterrows():
                sig = "*" if row["significant"] else "n.s."
                f.write(f"  {row['metric']:25s}: r={row['spearman_r']:.4f}, p={row['p_value']:.4f} {sig}\n")
            f.write(f"\nResult = {'SUPPORTED' if h2_ok else 'NOT SUPPORTED'} at α={ALPHA}\n")

    # Correlation bar chart
    if not corr_df.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ["#4A72B0" if m == "empathy_score" else "#AABDD4" for m in corr_df["metric"]]
        bars   = ax.bar(corr_df["metric"], corr_df["spearman_r"], color=colors)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title("RQ1 H2 – Correlation with Positive Sentiment (Chronic Posts Only)")
        ax.set_ylabel("Spearman r")
        labels = ["Empathy +\nSupport", "Clarity", "Accuracy"]
        ax.set_xticks(range(len(corr_df)))
        ax.set_xticklabels(labels[:len(corr_df)])
        for bar, row in zip(bars, corr_df.itertuples()):
            sig_str = "*" if row.significant else "n.s."
            offset  = 0.005 if row.spearman_r >= 0 else -0.015
            ax.text(bar.get_x() + bar.get_width() / 2,
                    row.spearman_r + offset,
                    f"r={row.spearman_r:.3f}\n{sig_str}", ha="center", va="bottom", fontsize=9)
        plt.tight_layout()
        plt.savefig(rq1_dir / "h2_correlation_bars.png", dpi=150)
        plt.close()

    # Scatter: empathy vs sentiment, chronic vs acute side by side
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, sev, color in zip(axes, ["chronic", "acute"], ["#4A72B0", "#E06C4A"]):
        grp = df[df["severity"] == sev]
        x, y = grp[empathy_col].fillna(0), grp[sentiment_col].fillna(0)
        ax.scatter(x, y, alpha=0.25, s=12, color=color)
        if x.std() > 0:
            m, b, r_val, p_val, _ = stats.linregress(x, y)
            xr = np.linspace(x.min(), x.max(), 100)
            ax.plot(xr, m * xr + b, "r--", linewidth=1.5)
            ax.set_title(f"Empathy vs Sentiment ({sev}) r={r_val:.3f} p={p_val:.3f}")
        else:
            ax.set_title(f"Empathy vs Sentiment ({sev})")
        ax.set_xlabel("Empathy Phrase Density")
        ax.set_ylabel("Sentiment Score")
    plt.tight_layout()
    plt.savefig(rq1_dir / "h2_scatter_plots.png", dpi=150)
    plt.close()

    # Per-topic empathy summary (if LDA topics available)
    if "dominant_topic" in df.columns:
        topic_agg = (
            df.groupby(["dominant_topic", "severity"])
            .agg(n=(empathy_col, "count"),
                 mean_empathy=(empathy_col, "mean"),
                 mean_sentiment=(sentiment_col, "mean"))
            .round(4)
            .reset_index()
        )
        topic_agg.to_csv(rq1_dir / "empathy_by_topic_severity.csv", index=False)

    # Per-cluster empathy summary (if clusters available)
    if "cluster" in df.columns:
        cluster_agg = (
            df.groupby(["cluster", "severity"])
            .agg(n=(empathy_col, "count"),
                 mean_empathy=(empathy_col, "mean"),
                 mean_sentiment=(sentiment_col, "mean"))
            .round(4)
            .reset_index()
        )
        cluster_agg.to_csv(rq1_dir / "empathy_by_cluster_severity.csv", index=False)

    log.info(f"\n  RQ1 outputs → {rq1_dir}")


# =============================================================================
# SECTION 4 – RQ2: SHARED EXPERIENCE AND SENTIMENT
# =============================================================================

def rq2_analysis(df: pd.DataFrame, output_dir: Path, label: str):
    """
    RQ2: Do comments containing 'shared experience' phrases show higher positive
         sentiment scores than comments without such language?

    H: Comments containing shared-experience phrases ('in my case', 'same thing
       happened to me', etc.) will have significantly higher positive sentiment
       scores than those without, indicating that relatable advice is framed more
       positively.

    The groups are highly imbalanced (WITH << WITHOUT).  At such scales, p-values
    from full-data tests collapse to ~0 even for trivial effects.  We therefore
    run N_BOOTSTRAP size-matched iterations (random subsamples of WITHOUT equal in
    size to WITH) and base the primary verdict on median effect size (r_rb, Cohen's d)
    and the fraction of bootstrap trials that are individually significant.

    Primary Test – Bootstrapped Mann-Whitney U (one-tailed, size-matched):
        N_BOOTSTRAP random samples of size n_with drawn from WITHOUT group.
        Verdict: median r_rb > EFFECT_THRESHOLD AND correct direction.

    Secondary – Full-data descriptives + Welch t-test (reported, p treated as
        informational only because n is very large).

    Effect sizes: rank-biserial r (Mann-Whitney) and Cohen's d (t-test).

    Input columns used (from features.py):
        shared_experience (binary 0/1), sentiment_score / vader_compound
    """
    if df.empty:
        log.warning(f"RQ2 ({label}): dataframe is empty – skipping")
        return

    rq2_dir = output_dir / "rq2"
    rq2_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"\n  -- RQ2 ({label}): Shared Experience vs Sentiment Score --")

    # Column aliases
    se_col   = "shared_experience" if "shared_experience" in df.columns else None
    sent_col = "sentiment_score"   if "sentiment_score"   in df.columns else "vader_compound"

    if se_col is None:
        log.warning(f"  RQ2 ({label}): 'shared_experience' column missing – skipping")
        return
    if sent_col not in df.columns:
        log.warning(f"  RQ2 ({label}): sentiment column ('{sent_col}') missing – skipping")
        return

    # Split into two groups
    with_sent    = df.loc[df[se_col] == 1, sent_col].dropna()
    without_sent = df.loc[df[se_col] == 0, sent_col].dropna()

    n_with    = len(with_sent)
    n_without = len(without_sent)
    imbalance_ratio = n_without / max(n_with, 1)
    log.info(f"  Comments WITH shared experience: {n_with:,}  |  WITHOUT: {n_without:,}  "
             f"(imbalance ratio {imbalance_ratio:.0f}:1)")

    if n_with < 2 or n_without < 2:
        log.warning(f"  RQ2 ({label}): insufficient data for group comparison – skipping")
        return

    # ── Full-data descriptive statistics ──────────────────────────────────────
    log.info(f"  WITH    – mean={with_sent.mean():.4f}, median={with_sent.median():.4f}, "
             f"std={with_sent.std():.4f}, n={n_with}")
    log.info(f"  WITHOUT – mean={without_sent.mean():.4f}, median={without_sent.median():.4f}, "
             f"std={without_sent.std():.4f}, n={n_without}")

    # ── Full-data Mann-Whitney U (informational only — p is inflated by large n) ─
    u_stat_full, p_mw_full = stats.mannwhitneyu(with_sent, without_sent, alternative="greater")
    r_rb_full = 1 - (2 * u_stat_full) / (n_with * n_without)
    log.info(f"  [Full data] Mann-Whitney U={u_stat_full:.1f}, p={p_mw_full:.4e}, "
             f"r_rb={r_rb_full:.4f}  <- p inflated by large n, treat as descriptive only")

    # ── Bootstrapped downsampling of the majority (WITHOUT) group ─────────────
    #
    #   The WITHOUT group is potentially 50x larger than WITH.  At this scale
    #   every test produces p ≈ 0 even for trivial effect sizes.  We instead
    #   draw N_BOOTSTRAP random samples of size n_with from WITHOUT, run the
    #   same tests on each balanced pair, and report medians.  This gives
    #   reliable, size-matched effect-size estimates.
    #
    N_BOOTSTRAP = 500
    rng = np.random.default_rng(seed=42)

    u_vals, r_rb_vals, t_vals, d_vals, p_mw_vals, p_t_vals = [], [], [], [], [], []

    log.info(f"  Running {N_BOOTSTRAP} bootstrapped size-matched comparisons "
             f"(n={n_with:,} per group) ...")

    without_arr = without_sent.values
    with_arr    = with_sent.values

    for _ in range(N_BOOTSTRAP):
        sample = rng.choice(without_arr, size=n_with, replace=False)

        u, p_u  = stats.mannwhitneyu(with_arr, sample, alternative="greater")
        r_rb_b  = 1 - (2 * u) / (n_with * n_with)

        t, p_t  = stats.ttest_ind(with_arr, sample, equal_var=False)

        s_with   = with_arr.std()
        s_sample = sample.std()
        pooled   = np.sqrt(((n_with - 1) * s_with**2 + (n_with - 1) * s_sample**2)
                           / (2 * n_with - 2))
        d = (with_arr.mean() - sample.mean()) / pooled if pooled > 0 else float("nan")

        u_vals.append(u);      r_rb_vals.append(r_rb_b)
        t_vals.append(t);      d_vals.append(d)
        p_mw_vals.append(p_u); p_t_vals.append(p_t)

    med_r_rb = float(np.nanmedian(r_rb_vals))
    med_d    = float(np.nanmedian(d_vals))
    med_p_mw = float(np.nanmedian(p_mw_vals))
    med_p_t  = float(np.nanmedian(p_t_vals))
    pct_sig  = float(np.mean(np.array(p_mw_vals) < ALPHA)) * 100

    log.info(f"  [Bootstrap median] r_rb={med_r_rb:.4f}, Cohen's d={med_d:.4f}, "
             f"MW p={med_p_mw:.4f}, t-test p={med_p_t:.4f}")
    log.info(f"  % bootstrap trials significant (MW p < {ALPHA}): {pct_sig:.1f}%")

    # ── Full-data Cohen's d (for reference; pooled SD biased by large n) ──────
    pooled_std_full = np.sqrt(
        ((n_with - 1) * with_sent.std() ** 2 + (n_without - 1) * without_sent.std() ** 2)
        / (n_with + n_without - 2)
    )
    cohens_d_full = ((with_sent.mean() - without_sent.mean()) / pooled_std_full
                     if pooled_std_full > 0 else float("nan"))

    # ── Hypothesis verdict: effect size + consistent direction ────────────────
    #   With n up to 3.6 M, p-values are meaningless as a sole criterion.
    #   Verdict is based on:
    #     (1) bootstrapped median r_rb > 0.05  (small but non-trivial effect)
    #     (2) correct direction  (WITH mean > WITHOUT mean)
    #     (3) >= 70% of bootstrap trials are individually significant
    EFFECT_THRESHOLD  = 0.05
    correct_direction = float(with_sent.mean()) > float(without_sent.mean())
    supported = (med_r_rb > EFFECT_THRESHOLD) and correct_direction and (pct_sig >= 70.0)
    log.info(f"  H -> {'SUPPORTED' if supported else 'NOT SUPPORTED'}  "
             f"(r_rb>{EFFECT_THRESHOLD}, correct direction, >=70% bootstrap trials sig)")

    # ── Save numerical results ────────────────────────────────────────────────
    results = pd.DataFrame([{
        "corpus":                       label,
        "n_with_shared_exp":            int(n_with),
        "n_without_shared_exp":         int(n_without),
        "imbalance_ratio":              round(imbalance_ratio, 1),
        "mean_sentiment_with":          round(float(with_sent.mean()), 4),
        "mean_sentiment_without":       round(float(without_sent.mean()), 4),
        "median_sentiment_with":        round(float(with_sent.median()), 4),
        "median_sentiment_without":     round(float(without_sent.median()), 4),
        # Full-data tests (descriptive only — p inflated by large n)
        "full_mann_whitney_u":          round(float(u_stat_full), 1),
        "full_mann_whitney_p":          round(float(p_mw_full), 6),
        "full_r_rb":                    round(float(r_rb_full), 4),
        "full_cohens_d":                round(float(cohens_d_full), 4) if not np.isnan(cohens_d_full) else float("nan"),
        # Bootstrapped size-matched tests (primary)
        "bootstrap_n":                  N_BOOTSTRAP,
        "bootstrap_sample_size":        int(n_with),
        "bootstrap_median_r_rb":        round(med_r_rb, 4),
        "bootstrap_median_cohens_d":    round(med_d, 4),
        "bootstrap_median_mw_p":        round(med_p_mw, 4),
        "bootstrap_median_t_p":         round(med_p_t, 4),
        "bootstrap_pct_sig_trials":     round(pct_sig, 1),
        "hypothesis_supported":         supported,
    }])
    results.to_csv(rq2_dir / f"results_{label}.csv", index=False)
    log.info(f"  Results saved -> {rq2_dir / f'results_{label}.csv'}")

    # ── Write human-readable result file ─────────────────────────────────────
    with open(rq2_dir / f"h_results_{label}.txt", "w", encoding="utf-8") as f:
        f.write("H: Comments with shared-experience phrases have higher positive sentiment\n\n")
        f.write(f"Corpus: {label}\n")
        f.write(f"n WITH shared experience:    {n_with:,}\n")
        f.write(f"n WITHOUT shared experience: {n_without:,}  "
                f"(imbalance ratio {imbalance_ratio:.0f}:1)\n\n")
        f.write("NOTE ON IMBALANCE\n")
        f.write("-----------------\n")
        f.write(f"The WITHOUT group is {imbalance_ratio:.0f}x larger than WITH.\n")
        f.write("At this scale, p-values from full-data tests approach 0 even for\n")
        f.write("negligible effect sizes. Full-data tests are reported for completeness\n")
        f.write(f"but the primary verdict uses {N_BOOTSTRAP} bootstrapped size-matched\n")
        f.write(f"comparisons (n={n_with:,} per group) and effect size thresholds.\n\n")
        f.write("Sentiment Score Summary:\n")
        f.write(f"  WITH    - mean={with_sent.mean():.4f}, median={with_sent.median():.4f}, "
                f"std={with_sent.std():.4f}\n")
        f.write(f"  WITHOUT - mean={without_sent.mean():.4f}, median={without_sent.median():.4f}, "
                f"std={without_sent.std():.4f}\n\n")
        f.write("Full-Data Tests (DESCRIPTIVE ONLY -- p inflated by large n):\n")
        f.write(f"  Mann-Whitney U = {u_stat_full:.1f},  p = {p_mw_full:.4e}\n")
        f.write(f"  Rank-biserial r = {r_rb_full:.4f}\n")
        f.write(f"  Cohen's d (full) = {cohens_d_full:.4f}\n\n")
        f.write(f"Bootstrapped Size-Matched Tests (PRIMARY, {N_BOOTSTRAP} iterations, "
                f"n={n_with:,} per group):\n")
        f.write(f"  Median rank-biserial r  = {med_r_rb:.4f}  "
                "(|r| > 0.1 small, > 0.3 medium, > 0.5 large)\n")
        f.write(f"  Median Cohen's d        = {med_d:.4f}  "
                "(|d| > 0.2 small, > 0.5 medium, > 0.8 large)\n")
        f.write(f"  Median Mann-Whitney p   = {med_p_mw:.4f}\n")
        f.write(f"  % trials significant    = {pct_sig:.1f}%\n\n")
        f.write("Verdict criteria:\n")
        f.write(f"  (1) bootstrap median r_rb > {EFFECT_THRESHOLD}  -> "
                f"{'PASS' if med_r_rb > EFFECT_THRESHOLD else 'FAIL'}\n")
        f.write(f"  (2) WITH mean > WITHOUT mean               -> "
                f"{'PASS' if correct_direction else 'FAIL'}\n")
        f.write(f"  (3) >= 70% bootstrap trials significant    -> "
                f"{'PASS' if pct_sig >= 70.0 else 'FAIL'} ({pct_sig:.1f}%)\n\n")
        f.write(f"Result: {'SUPPORTED' if supported else 'NOT SUPPORTED'} at alpha={ALPHA}\n")

    # ── Violin / box plot: sentiment split by shared_experience ───────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    plot_df = df[[se_col, sent_col]].dropna().copy()
    plot_df["Shared Experience"] = plot_df[se_col].map({1: "Yes", 0: "No"})

    # Cap plot size so violin kernel estimation stays fast
    MAX_PLOT = 50_000
    plot_sample = plot_df.groupby("Shared Experience", group_keys=False).apply(
        lambda g: g.sample(min(len(g), MAX_PLOT), random_state=42)
    )

    # Violin
    sns.violinplot(
        data=plot_sample, x="Shared Experience", y=sent_col,
        palette={"Yes": "#4A72B0", "No": "#AABDD4"},
        order=["Yes", "No"], ax=axes[0], inner="box",
    )
    axes[0].set_title(
        f"RQ2 ({label}): Sentiment by Shared Experience\n"
        f"bootstrap r_rb={med_r_rb:.3f}, Cohen's d={med_d:.3f}  "
        f"(n<={MAX_PLOT:,} per group shown)"
    )
    axes[0].set_ylabel("Sentiment Score")

    # Mean-bar with 95% CI (computed on FULL data)
    grp_stats = plot_df.groupby("Shared Experience")[sent_col].agg(["mean", "sem"]).reindex(["Yes", "No"])
    ci95 = grp_stats["sem"] * 1.96
    axes[1].bar(
        grp_stats.index, grp_stats["mean"],
        yerr=ci95, capsize=5,
        color=["#4A72B0", "#AABDD4"], edgecolor="black", linewidth=0.7,
    )
    axes[1].set_title(
        f"RQ2 ({label}): Mean Sentiment +/- 95% CI (full data)\n"
        f"bootstrap Cohen's d={med_d:.3f}  r_rb={med_r_rb:.3f}"
    )
    axes[1].set_ylabel("Mean Sentiment Score")
    axes[1].set_xlabel("Shared Experience")

    plt.tight_layout()
    plt.savefig(rq2_dir / f"violin_{label}.png", dpi=150)
    plt.close()

    # ── Bootstrap distribution plot ───────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(r_rb_vals, bins=40, color="#4A72B0", edgecolor="white", linewidth=0.4)
    axes[0].axvline(med_r_rb, color="red", linestyle="--", label=f"median={med_r_rb:.4f}")
    axes[0].axvline(EFFECT_THRESHOLD, color="orange", linestyle=":", label=f"threshold={EFFECT_THRESHOLD}")
    axes[0].set_title(f"RQ2 ({label}): Bootstrap rank-biserial r\n({N_BOOTSTRAP} size-matched trials)")
    axes[0].set_xlabel("Rank-biserial r")
    axes[0].legend()

    axes[1].hist(d_vals, bins=40, color="#E06C4A", edgecolor="white", linewidth=0.4)
    axes[1].axvline(med_d, color="red", linestyle="--", label=f"median={med_d:.4f}")
    axes[1].axvline(0.2, color="orange", linestyle=":", label="small effect (0.2)")
    axes[1].set_title(f"RQ2 ({label}): Bootstrap Cohen's d\n({N_BOOTSTRAP} size-matched trials)")
    axes[1].set_xlabel("Cohen's d")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(rq2_dir / f"bootstrap_{label}.png", dpi=150)
    plt.close()

    # ── Per-topic breakdown (if LDA topics available) ─────────────────────────
    if "dominant_topic" in df.columns:
        topic_agg = (
            df.groupby("dominant_topic")
            .agg(
                n=(se_col, "count"),
                pct_shared_exp=(se_col, "mean"),
                mean_sentiment_with=(
                    sent_col,
                    lambda s: s[df.loc[s.index, se_col] == 1].mean(),
                ),
                mean_sentiment_all=(sent_col, "mean"),
            )
            .round(4)
            .reset_index()
        )
        topic_agg.to_csv(rq2_dir / f"by_topic_{label}.csv", index=False)

    # ── Per-cluster breakdown ─────────────────────────────────────────────────
    if "cluster" in df.columns:
        cluster_agg = (
            df.groupby("cluster")
            .agg(
                n=(se_col, "count"),
                pct_shared_exp=(se_col, "mean"),
                mean_sentiment_all=(sent_col, "mean"),
            )
            .round(4)
            .reset_index()
        )
        cluster_agg.to_csv(rq2_dir / f"by_cluster_{label}.csv", index=False)

    log.info(f"  RQ2 ({label}) outputs -> {rq2_dir}")


def rq2_h2_analysis(df: pd.DataFrame, output_dir: Path, label: str):
    """
    RQ2 H2: Is uncertainty_density negatively correlated with lexical_density
            in advice comments?

    Rationale: advisors who hedge with uncertain language ('perhaps', 'might',
    'I'm not sure') may deliberately use simpler, less jargon-heavy vocabulary
    to come across as more approachable and less authoritative.  A negative
    Spearman correlation would support this.

    Test: Spearman r between uncertainty_density and lexical_density.
          One-tailed directional test: H1: r < 0 (negative relationship).

    No class-imbalance issue: both variables are continuous density scores.

    Input columns used (from features.py):
        uncertainty_density, lexical_density
    """
    if df.empty:
        log.warning(f"RQ2-H2 ({label}): dataframe is empty - skipping")
        return

    rq2_dir = output_dir / "rq2"
    rq2_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"\n  -- RQ2 H2 ({label}): Uncertainty Density vs Lexical Density --")

    ud_col = "uncertainty_density"
    ld_col = "lexical_density"

    for col in (ud_col, ld_col):
        if col not in df.columns:
            log.warning(f"  RQ2-H2 ({label}): column '{col}' missing - skipping")
            return

    # Drop rows where either column is NaN
    pair = df[[ud_col, ld_col]].dropna()
    n    = len(pair)
    log.info(f"  Valid pairs for correlation: {n:,}")

    if n < 10:
        log.warning(f"  RQ2-H2 ({label}): too few observations ({n}) - skipping")
        return

    # Spearman correlation (non-parametric; no normality assumption)
    r_sp, p_two = stats.spearmanr(pair[ud_col], pair[ld_col])
    # Convert two-tailed p to one-tailed (for negative direction)
    p_one = p_two / 2 if r_sp < 0 else 1 - p_two / 2

    log.info(f"  Spearman r = {r_sp:.4f},  p (two-tailed) = {p_two:.4f},  "
             f"p (one-tailed, r<0) = {p_one:.4f}")
    log.info(f"  uncertainty_density - mean={pair[ud_col].mean():.4f}, "
             f"std={pair[ud_col].std():.4f}")
    log.info(f"  lexical_density     - mean={pair[ld_col].mean():.4f}, "
             f"std={pair[ld_col].std():.4f}")

    # Verdict: r negative AND one-tailed p < alpha
    supported = (r_sp < 0) and (p_one < ALPHA)
    log.info(f"  H2 -> {'SUPPORTED' if supported else 'NOT SUPPORTED'} "
             f"(r<0 and one-tailed p<{ALPHA})")

    # Save numerical results
    results = pd.DataFrame([{
        "corpus":               label,
        "n":                    int(n),
        "spearman_r":           round(float(r_sp), 4),
        "p_two_tailed":         round(float(p_two), 4),
        "p_one_tailed_neg":     round(float(p_one), 4),
        "mean_uncertainty":     round(float(pair[ud_col].mean()), 4),
        "mean_lexical":         round(float(pair[ld_col].mean()), 4),
        "hypothesis_supported": supported,
    }])
    results.to_csv(rq2_dir / f"h2_results_{label}.csv", index=False)
    log.info(f"  Results saved -> {rq2_dir / f'h2_results_{label}.csv'}")

    # Human-readable result file
    with open(rq2_dir / f"h2_results_{label}.txt", "w", encoding="utf-8") as f:
        f.write("H2: uncertainty_density is negatively correlated with lexical_density "
                "in advice comments\n\n")
        f.write(f"Corpus: {label}  |  n = {n:,}\n\n")
        f.write(f"  uncertainty_density - mean={pair[ud_col].mean():.4f}, "
                f"std={pair[ud_col].std():.4f}\n")
        f.write(f"  lexical_density     - mean={pair[ld_col].mean():.4f}, "
                f"std={pair[ld_col].std():.4f}\n\n")
        f.write("Spearman Correlation:\n")
        f.write(f"  r                       = {r_sp:.4f}\n")
        f.write(f"  p (two-tailed)          = {p_two:.4f}\n")
        f.write(f"  p (one-tailed, r < 0)   = {p_one:.4f}\n")
        f.write(f"  alpha                   = {ALPHA}\n\n")
        f.write("  |r| interpretation: > 0.1 small, > 0.3 medium, > 0.5 large\n\n")
        f.write(f"Result: {'SUPPORTED' if supported else 'NOT SUPPORTED'} ")
        f.write(f"(r={'negative' if r_sp < 0 else 'positive'}, "
                f"one-tailed p={'<' if p_one < ALPHA else '>='} alpha)\n")

    # Scatter plot with regression line
    MAX_PLOT = 30_000
    plot_df  = pair.sample(min(n, MAX_PLOT), random_state=42)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(
        plot_df[ud_col], plot_df[ld_col],
        alpha=0.15, s=8, color="#4A72B0", rasterized=True,
    )
    if plot_df[ud_col].std() > 0:
        m, b, *_ = stats.linregress(plot_df[ud_col], plot_df[ld_col])
        x_range  = np.linspace(plot_df[ud_col].min(), plot_df[ud_col].max(), 200)
        ax.plot(x_range, m * x_range + b, color="red", linewidth=1.5,
                label=f"OLS slope={m:.4f}")

    ax.set_xlabel("Uncertainty Density (phrases per 100 words)")
    ax.set_ylabel("Lexical Density (content-word proportion)")
    ax.set_title(
        f"RQ2 H2 ({label}): Uncertainty vs Lexical Density\n"
        f"Spearman r={r_sp:.4f},  one-tailed p={p_one:.4f}  "
        f"({'SUPPORTED' if supported else 'NOT SUPPORTED'})  "
        f"n<={MAX_PLOT:,} shown"
    )
    ax.legend()
    plt.tight_layout()
    plt.savefig(rq2_dir / f"h2_scatter_{label}.png", dpi=150)
    plt.close()

    log.info(f"  RQ2-H2 ({label}) outputs -> {rq2_dir}")


# =============================================================================
# SECTION 5 – RQ3: GENERALISATION TO NON-MEDICAL
# =============================================================================

def jaccard(set1: set, set2: set) -> float:
    if not set1 or not set2:
        return 0.0
    return len(set1 & set2) / len(set1 | set2)


def rq3_topic_overlap(medical_words_csv: str, nonmedical_words_csv: str,
                      output_dir: Path):
    """
    RQ3 H1: The top-ranked LDA topics from medical posts will also emerge
            in non-medical posts.

    Method: Jaccard similarity on the top-N words of each topic pair.
    A mean best-match similarity above the threshold suggests generalisation.
    """
    rq3_dir = output_dir / "rq3"
    rq3_dir.mkdir(parents=True, exist_ok=True)

    log.info("\n  -- RQ3 H1: LDA Topic Overlap (Medical vs Non-Medical) --")

    if _is_placeholder(medical_words_csv) or _is_placeholder(nonmedical_words_csv):
        log.warning("  RQ3 H1: Placeholder paths – skipping topic overlap analysis")
        return

    med_df    = pd.read_csv(medical_words_csv)
    nonmed_df = pd.read_csv(nonmedical_words_csv)

    def top_word_sets(df, id_col="topic_id"):
        return {
            tid: set(grp.sort_values("rank")["word"].tolist()[:TOP_WORDS_FOR_SIMILARITY])
            for tid, grp in df.groupby(id_col)
        }

    med_sets    = top_word_sets(med_df)
    nonmed_sets = top_word_sets(nonmed_df)
    med_ids     = sorted(med_sets)
    nonmed_ids  = sorted(nonmed_sets)

    # Jaccard similarity matrix
    sim_matrix = np.array([
        [jaccard(med_sets[m], nonmed_sets[n]) for n in nonmed_ids]
        for m in med_ids
    ])
    sim_df = pd.DataFrame(sim_matrix, index=med_ids, columns=nonmed_ids)
    sim_df.index.name   = "medical_topic"
    sim_df.columns.name = "nonmedical_topic"
    sim_df.to_csv(rq3_dir / "topic_similarity_matrix.csv")

    # Best match per medical topic
    best_matches = []
    for i, mid in enumerate(med_ids):
        j_best    = int(sim_matrix[i].argmax())
        best_sim  = float(sim_matrix[i, j_best])
        best_matches.append({
            "medical_topic":        mid,
            "best_nonmedical_match": nonmed_ids[j_best],
            "jaccard_similarity":   round(best_sim, 4),
            "medical_top_words":    ", ".join(sorted(med_sets[mid])[:5]),
            "nonmedical_top_words": ", ".join(sorted(nonmed_sets[nonmed_ids[j_best]])[:5]),
        })
    best_df = pd.DataFrame(best_matches)
    best_df.to_csv(rq3_dir / "topic_best_matches.csv", index=False)
    log.info(f"\n{best_df.to_string(index=False)}")

    mean_sim = float(sim_matrix.max(axis=1).mean())
    OVERLAP_THRESHOLD = 0.10   # Jaccard > 0.1 suggests meaningful word overlap
    supported = mean_sim > OVERLAP_THRESHOLD
    log.info(f"\n  Mean best-match Jaccard similarity: {mean_sim:.4f}")
    log.info(f"  H1 → {'SUPPORTED' if supported else 'NOT SUPPORTED'} "
             f"(threshold={OVERLAP_THRESHOLD})")

    with open(rq3_dir / "h1_results.txt", "w", encoding="utf-8") as f:
        f.write("H1: Medical LDA topics re-emerge in non-medical posts\n\n")
        f.write(f"Method: Jaccard similarity on top-{TOP_WORDS_FOR_SIMILARITY} words per topic\n")
        f.write(f"Mean best-match similarity: {mean_sim:.4f}\n")
        f.write(f"Threshold for 'generalisation': {OVERLAP_THRESHOLD}\n")
        f.write(f"Result: {'SUPPORTED' if supported else 'NOT SUPPORTED'}\n")

    # Heatmap
    fig_w = max(8, len(nonmed_ids) * 1.1)
    fig_h = max(6, len(med_ids) * 0.8)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    sns.heatmap(sim_df, annot=True, fmt=".2f", cmap="Blues", ax=ax,
                linewidths=0.5, cbar_kws={"label": "Jaccard Similarity"})
    ax.set_title(f"RQ3 H1 – Topic Similarity: Medical vs Non-Medical\n"
                 f"(top-{TOP_WORDS_FOR_SIMILARITY} words, mean best-match={mean_sim:.3f})")
    ax.set_xlabel("Non-Medical Topic")
    ax.set_ylabel("Medical Topic")
    plt.tight_layout()
    plt.savefig(rq3_dir / "h1_topic_similarity_heatmap.png", dpi=150)
    plt.close()
    log.info(f"  Heatmap saved → {rq3_dir / 'h1_topic_similarity_heatmap.png'}")


def rq3_correlation_generalisation(medical_df: pd.DataFrame,
                                   nonmedical_df: pd.DataFrame,
                                   output_dir: Path):
    """
    RQ3 H2: Correlations between positive sentiment and (a) text length
            and (b) uncertainty vocabulary density found in medical posts
            also appear in non-medical posts.

    Test: Spearman correlations computed independently for each corpus.
    Generalisation check: same direction + both statistically significant.
    Fisher's Z-test: formally compares the two correlation coefficients.

    Input columns used (from features.py + extra):
        sentiment_score / vader_compound, uncertainty_density, word_count
    """
    if medical_df.empty or nonmedical_df.empty:
        log.warning("RQ3 H2: one or both dataframes empty – skipping")
        return

    rq3_dir = output_dir / "rq3"
    rq3_dir.mkdir(parents=True, exist_ok=True)

    log.info("\n  -- RQ3 H2: Correlation Generalisation (Medical vs Non-Medical) --")

    sent_col = "sentiment_score" if "sentiment_score" in medical_df.columns else "vader_compound"
    features = ["word_count", "uncertainty_density"]

    rows = []
    for corpus_label, df in [("medical", medical_df), ("non_medical", nonmedical_df)]:
        if sent_col not in df.columns:
            log.warning(f"  {corpus_label}: '{sent_col}' not found – skipping")
            continue
        for feat in features:
            if feat not in df.columns:
                log.warning(f"  {corpus_label}: '{feat}' not found – skipping")
                continue
            r, p = safe_spearman(df[feat], df[sent_col])
            rows.append({
                "corpus":      corpus_label,
                "feature":     feat,
                "spearman_r":  r,
                "p_value":     p,
                "n":           len(df),
                "significant": bool(not np.isnan(p) and p < ALPHA),
            })
            log.info(f"  [{corpus_label}] {feat}: r={r}, p={p}")

    results_df = pd.DataFrame(rows)
    results_df.to_csv(rq3_dir / "h2_correlations.csv", index=False)

    # Fisher Z-test between medical and non-medical for each feature
    fisher_rows = []
    for feat in features:
        med = results_df[(results_df["corpus"] == "medical")    & (results_df["feature"] == feat)]
        non = results_df[(results_df["corpus"] == "non_medical") & (results_df["feature"] == feat)]
        if med.empty or non.empty:
            continue
        r1, n1 = med.iloc[0]["spearman_r"], int(med.iloc[0]["n"])
        r2, n2 = non.iloc[0]["spearman_r"], int(non.iloc[0]["n"])
        if np.isnan(r1) or np.isnan(r2):
            continue
        z_stat, p_fisher = fisher_z_test(r1, n1, r2, n2)
        same_dir  = (r1 > 0) == (r2 > 0)
        both_sig  = bool(med.iloc[0]["significant"] and non.iloc[0]["significant"])
        generalises = same_dir and both_sig
        fisher_rows.append({
            "feature":        feat,
            "medical_r":      r1,
            "nonmedical_r":   r2,
            "fisher_z":       z_stat,
            "fisher_p":       p_fisher,
            "same_direction": same_dir,
            "both_significant": both_sig,
            "generalises":    generalises,
        })
        log.info(f"  {feat}: Fisher Z={z_stat}, p={p_fisher} | generalises={generalises}")

    if fisher_rows:
        fisher_df = pd.DataFrame(fisher_rows)
        fisher_df.to_csv(rq3_dir / "h2_fisher_tests.csv", index=False)

    with open(rq3_dir / "h2_results.txt", "w", encoding="utf-8") as f:
        f.write("H2: Correlations between sentiment and (text length, uncertainty) generalise to non-medical posts\n\n")
        for feat in features:
            med = results_df[(results_df["corpus"] == "medical")    & (results_df["feature"] == feat)]
            non = results_df[(results_df["corpus"] == "non_medical") & (results_df["feature"] == feat)]
            if med.empty or non.empty:
                continue
            f.write(f"Feature: {feat}\n")
            f.write(f"  Medical:     r={med.iloc[0]['spearman_r']:.4f}, p={med.iloc[0]['p_value']:.4f}\n")
            f.write(f"  Non-medical: r={non.iloc[0]['spearman_r']:.4f}, p={non.iloc[0]['p_value']:.4f}\n")
            fr = [r for r in fisher_rows if r["feature"] == feat]
            if fr:
                f.write(f"  Fisher Z={fr[0]['fisher_z']}, p={fr[0]['fisher_p']}\n")
                f.write(f"  Generalises: {fr[0]['generalises']}\n\n")

    # Grouped bar chart
    if not results_df.empty:
        fig, axes = plt.subplots(1, len(features), figsize=(6 * len(features), 5))
        if len(features) == 1:
            axes = [axes]
        for ax, feat in zip(axes, features):
            sub = results_df[results_df["feature"] == feat]
            if sub.empty:
                continue
            colors = ["#4A72B0", "#E06C4A"]
            bars   = ax.bar(sub["corpus"], sub["spearman_r"], color=colors[:len(sub)])
            ax.axhline(0, color="black", linewidth=0.8)
            ax.set_title(feat.replace("_", " ").title() + " vs Sentiment")
            ax.set_ylabel("Spearman r")
            for bar, row in zip(bars, sub.itertuples()):
                sig_str = "*" if row.significant else "n.s."
                offset  = 0.005 if row.spearman_r >= 0 else -0.02
                ax.text(bar.get_x() + bar.get_width() / 2,
                        row.spearman_r + offset,
                        f"r={row.spearman_r:.3f}\n{sig_str}",
                        ha="center", va="bottom", fontsize=9)
        plt.suptitle("RQ3 H2 – Do Correlations Generalise to Non-Medical Posts?")
        plt.tight_layout()
        plt.savefig(rq3_dir / "h2_correlation_bars.png", dpi=150)
        plt.close()

    log.info(f"\n  RQ3 outputs → {rq3_dir}")


# =============================================================================
# SECTION 6 – MAIN
# =============================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Generate naming files ────────────────────────────────────────
    save_all_naming_files(OUTPUT_DIR)

    # ── Step 2: Load and prepare data ────────────────────────────────────────
    log.info(f"\n{'='*55}")
    log.info("STEP 2: Loading and preparing data")
    log.info(f"{'='*55}")

    medical_df    = load_and_prepare(LDA_MEDICAL_DOCS,    "medical posts (LDA)",
                                     CLUSTER_MEDICAL_DOCS)
    nonmedical_df = load_and_prepare(LDA_NONMEDICAL_DOCS, "non-medical posts (LDA)",
                                     CLUSTER_NONMEDICAL_DOCS)
    comments_df   = load_and_prepare(LDA_COMMENTS_DOCS,   "comments (LDA)",
                                     CLUSTER_COMMENTS_DOCS)

    # ── Step 3: RQ1 ──────────────────────────────────────────────────────────
    log.info(f"\n{'='*55}")
    log.info("STEP 3: RQ1 – Severity and Nature of Medical Conditions")
    log.info(f"{'='*55}")
    rq1_analysis(medical_df, OUTPUT_DIR)

    # ── Step 4: RQ2 (run on all three corpora) ───────────────────────────────
    log.info(f"\n{'='*55}")
    log.info("STEP 4: RQ2 – Shared Experience and Positive Sentiment")
    log.info(f"{'='*55}")
    rq2_analysis(medical_df,    OUTPUT_DIR, label="medical")
    rq2_analysis(nonmedical_df, OUTPUT_DIR, label="non_medical")
    rq2_analysis(comments_df,   OUTPUT_DIR, label="comments")

    log.info(f"\n{'='*55}")
    log.info("STEP 4b: RQ2 H2 – Uncertainty vs Lexical Density")
    log.info(f"{'='*55}")
    rq2_h2_analysis(medical_df,    OUTPUT_DIR, label="medical")
    rq2_h2_analysis(nonmedical_df, OUTPUT_DIR, label="non_medical")
    rq2_h2_analysis(comments_df,   OUTPUT_DIR, label="comments")

    # ── Step 5: RQ3 ──────────────────────────────────────────────────────────
    log.info(f"\n{'='*55}")
    log.info("STEP 5: RQ3 – Generalisation to Non-Medical Posts")
    log.info(f"{'='*55}")
    rq3_topic_overlap(LDA_MEDICAL_WORDS, LDA_NONMEDICAL_WORDS, OUTPUT_DIR)
    rq3_correlation_generalisation(medical_df, nonmedical_df, OUTPUT_DIR)

    # ── Summary ───────────────────────────────────────────────────────────────
    log.info(f"\n{'='*55}")
    log.info("ANALYSIS COMPLETE")
    log.info(f"{'='*55}")
    log.info("Output structure:")
    log.info("  output/analysis/naming/                      topic and cluster naming CSVs (edit manual_name)")
    log.info("  output/analysis/rq1/h1_group_stats.csv       empathy by severity x nature")
    log.info("  output/analysis/rq1/h1_results.txt           H1 test result")
    log.info("  output/analysis/rq1/h1_boxplots.png          empathy and sentiment boxplots")
    log.info("  output/analysis/rq1/h2_correlations.csv      correlation table (chronic posts)")
    log.info("  output/analysis/rq1/h2_results.txt           H2 test result")
    log.info("  output/analysis/rq1/h2_correlation_bars.png")
    log.info("  output/analysis/rq1/h2_scatter_plots.png")
    log.info("  -- RQ2 H1: Shared experience -> higher positive sentiment --")
    log.info("  output/analysis/rq2/results_<corpus>.csv     H1 test result (bootstrap MW + t-test)")
    log.info("  output/analysis/rq2/h_results_<corpus>.txt   H1 human-readable result")
    log.info("  output/analysis/rq2/violin_<corpus>.png      H1 violin + mean-bar plots")
    log.info("  output/analysis/rq2/bootstrap_<corpus>.png   H1 bootstrap effect-size distributions")
    log.info("  output/analysis/rq2/by_topic_<corpus>.csv")
    log.info("  output/analysis/rq2/by_cluster_<corpus>.csv")
    log.info("  -- RQ2 H2: uncertainty_density negatively correlated with lexical_density --")
    log.info("  output/analysis/rq2/h2_results_<corpus>.csv  H2 Spearman result")
    log.info("  output/analysis/rq2/h2_results_<corpus>.txt  H2 human-readable result")
    log.info("  output/analysis/rq2/h2_scatter_<corpus>.png  H2 scatter plot")
    log.info("  output/analysis/rq3/topic_similarity_matrix.csv")
    log.info("  output/analysis/rq3/topic_best_matches.csv")
    log.info("  output/analysis/rq3/h1_results.txt")
    log.info("  output/analysis/rq3/h1_topic_similarity_heatmap.png")
    log.info("  output/analysis/rq3/h2_correlations.csv")
    log.info("  output/analysis/rq3/h2_fisher_tests.csv")
    log.info("  output/analysis/rq3/h2_results.txt")
    log.info("  output/analysis/rq3/h2_correlation_bars.png")


if __name__ == "__main__":
    main()
