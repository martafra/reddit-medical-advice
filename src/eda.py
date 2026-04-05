"""
Exploratory Data Analysis for Reddit Medical Advice Corpus
CS7IS4 / Text Analytics - Group 10

Produces:
  - Console summary statistics
  - output/eda_plots/ - PNG charts
  - output/eda_summary.txt - full text report

"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import nltk
import re
import logging
from pathlib import Path
from collections import Counter

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
POSTS_CSV    = "output/reddit_posts.csv"
COMMENTS_CSV = "output/reddit_comments.csv"
PLOTS_DIR    = Path("output/eda_plots")
REPORT_PATH  = Path("output/eda_summary.txt")

# conditions to track in the text
CONDITION_KEYWORDS = {
    "depression":        ["depression", "depressed", "depressive"],
    "anxiety":           ["anxiety", "anxious", "panic attack"],
    "bipolar":           ["bipolar", "manic", "mania"],
    "OCD":               ["ocd", "obsessive", "compulsive"],
    "PTSD":              ["ptsd", "trauma", "traumatic"],
    "diabetes":          ["diabetes", "diabetic", "insulin", "glucose"],
    "cancer":            ["cancer", "tumor", "chemotherapy", "oncology"],
    "chronic pain":      ["chronic pain", "fibromyalgia", "pain management"],
    "multiple sclerosis":["multiple sclerosis", "ms diagnosis", "ms flare"],
}

sns.set_theme(style="whitegrid", palette="muted")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

report_lines = []

def section(title: str):
    line = f"\n{'='*55}\n{title}\n{'='*55}"
    log.info(line)
    report_lines.append(line)

def info(msg: str):
    log.info(msg)
    report_lines.append(msg)

def save_fig(name: str):
    path = PLOTS_DIR / f"{name}.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    log.info(f"  Plot saved: {path}")


# ---------------------------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------------------------
section("LOADING DATA")
log.info("Loading posts...")
posts = pd.read_csv(POSTS_CSV, parse_dates=["created_utc"])
log.info("Loading comments...")
comments = pd.read_csv(COMMENTS_CSV)
comments["score"] = pd.to_numeric(comments["score"], errors="coerce").fillna(0).astype(int)

info(f"Total posts:    {len(posts):,}")
info(f"Total comments: {len(comments):,}")
info(f"Post columns:    {list(posts.columns)}")
info(f"Comment columns: {list(comments.columns)}")


# ---------------------------------------------------------------------------
# 1. BASIC STATISTICS
# ---------------------------------------------------------------------------
section("1. BASIC STATISTICS")

posts["text_len"]   = posts["text"].str.len()
posts["word_count"] = posts["text"].str.split().str.len()

info(f"\nPost text length (characters):")
info(f"  mean:   {posts['text_len'].mean():.0f}")
info(f"  median: {posts['text_len'].median():.0f}")
info(f"  min:    {posts['text_len'].min()}")
info(f"  max:    {posts['text_len'].max()}")

info(f"\nPost word count:")
info(f"  mean:   {posts['word_count'].mean():.0f}")
info(f"  median: {posts['word_count'].median():.0f}")

comments["text_len"]   = comments["text"].str.len()
comments["word_count"] = comments["text"].str.split().str.len()

info(f"\nComment text length (characters):")
info(f"  mean:   {comments['text_len'].mean():.0f}")
info(f"  median: {comments['text_len'].median():.0f}")

info(f"\nComment word count:")
info(f"  mean:   {comments['word_count'].mean():.0f}")
info(f"  median: {comments['word_count'].median():.0f}")

info(f"\nPost score - mean: {posts['score'].mean():.1f}, median: {posts['score'].median():.0f}")
info(f"Comment score - mean: {comments['score'].mean():.1f}, median: {comments['score'].median():.0f}")


# ---------------------------------------------------------------------------
# 2. DISTRIBUTION BY CATEGORY AND SUBREDDIT
# ---------------------------------------------------------------------------
section("2. DISTRIBUTION BY CATEGORY AND SUBREDDIT")

cat_dist = posts["category"].value_counts()
info(f"\nPosts per category:")
for cat, n in cat_dist.items():
    info(f"  {cat:<25} {n:>6,}  ({n/len(posts)*100:.1f}%)")

sub_dist = posts["subreddit"].value_counts()
info(f"\nPosts per subreddit:")
for sub, n in sub_dist.items():
    info(f"  {sub:<25} {n:>6,}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

cat_dist.plot(kind="bar", ax=axes[0], color=sns.color_palette("muted", len(cat_dist)))
axes[0].set_title("Posts per category")
axes[0].set_xlabel("")
axes[0].set_ylabel("Count")
axes[0].tick_params(axis="x", rotation=30)
for bar in axes[0].patches:
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                 f"{int(bar.get_height()):,}", ha="center", va="bottom", fontsize=9)

sub_dist.plot(kind="barh", ax=axes[1], color=sns.color_palette("muted", len(sub_dist)))
axes[1].set_title("Posts per subreddit")
axes[1].set_xlabel("Count")
axes[1].invert_yaxis()
save_fig("01_distribution_category_subreddit")


# ---------------------------------------------------------------------------
# 3. TEXT LENGTH DISTRIBUTION
# ---------------------------------------------------------------------------
section("3. TEXT LENGTH DISTRIBUTION")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(posts["word_count"].clip(upper=1000), bins=50,
             color=sns.color_palette("muted")[0], edgecolor="white")
axes[0].set_title("Post word count distribution")
axes[0].set_xlabel("Word count (capped at 1000)")
axes[0].set_ylabel("Frequency")
axes[0].axvline(posts["word_count"].median(), color="red", linestyle="--",
                label=f"Median: {posts['word_count'].median():.0f}")
axes[0].legend()

axes[1].hist(comments["word_count"].clip(upper=500), bins=50,
             color=sns.color_palette("muted")[1], edgecolor="white")
axes[1].set_title("Comment word count distribution")
axes[1].set_xlabel("Word count (capped at 500)")
axes[1].set_ylabel("Frequency")
axes[1].axvline(comments["word_count"].median(), color="red", linestyle="--",
                label=f"Median: {comments['word_count'].median():.0f}")
axes[1].legend()

save_fig("02_text_length_distribution")

fig, ax = plt.subplots(figsize=(10, 5))
posts.boxplot(column="word_count", by="category", ax=ax,
              showfliers=False, patch_artist=True)
ax.set_title("Word count by category")
ax.set_xlabel("Category")
ax.set_ylabel("Word count")
plt.suptitle("")
ax.tick_params(axis="x", rotation=20)
save_fig("03_wordcount_by_category")


# ---------------------------------------------------------------------------
# 4. TEMPORAL DISTRIBUTION
# ---------------------------------------------------------------------------
section("4. TEMPORAL DISTRIBUTION")

posts["year_month"] = posts["created_utc"].dt.to_period("M")
temporal = posts.groupby(["year_month", "category"]).size().unstack(fill_value=0)

info(f"\nTime range: {posts['created_utc'].min()} → {posts['created_utc'].max()}")
info(f"Months covered: {posts['year_month'].nunique()}")

fig, ax = plt.subplots(figsize=(14, 5))
temporal.plot(ax=ax, linewidth=1.5)
ax.set_title("Posts over time by category")
ax.set_xlabel("Month")
ax.set_ylabel("Post count")
ax.legend(title="Category", bbox_to_anchor=(1.01, 1), loc="upper left")
ax.xaxis.set_major_locator(mticker.MaxNLocator(12))
plt.xticks(rotation=45)
save_fig("04_temporal_distribution")


# ---------------------------------------------------------------------------
# 5. CONDITION FREQUENCY
# ---------------------------------------------------------------------------
section("5. MEDICAL CONDITION FREQUENCY")

posts["text_lower"] = posts["text"].str.lower().fillna("")

condition_counts = {}
for condition, keywords in CONDITION_KEYWORDS.items():
    pattern = "|".join(re.escape(k) for k in keywords)
    count = posts["text_lower"].str.contains(pattern, regex=True).sum()
    condition_counts[condition] = count
    info(f"  {condition:<25} {count:>6,}  ({count/len(posts)*100:.1f}%)")

# flag conditions that barely appear
threshold = len(posts) * 0.01
low_freq = [c for c, n in condition_counts.items() if n < threshold]
if low_freq:
    info(f"\n[!] Conditions with frequency < 1% (consider removing): {low_freq}")

fig, ax = plt.subplots(figsize=(10, 5))
cond_series = pd.Series(condition_counts).sort_values(ascending=True)
cond_series.plot(kind="barh", ax=ax, color=sns.color_palette("muted", len(cond_series)))
ax.set_title("Medical condition frequency in post text")
ax.set_xlabel("Number of posts mentioning condition")
for bar in ax.patches:
    ax.text(bar.get_width() + 10, bar.get_y() + bar.get_height()/2,
            f"{int(bar.get_width()):,}", va="center", fontsize=9)
save_fig("05_condition_frequency")


# ---------------------------------------------------------------------------
# 6. COMMENT TYPE DISTRIBUTION
# ---------------------------------------------------------------------------
section("6. COMMENT TYPE DISTRIBUTION (RQ2)")

ct_dist = comments["comment_type"].value_counts()
info(f"\nComment type distribution:")
for ct, n in ct_dist.items():
    info(f"  {ct:<20} {n:>6,}  ({n/len(comments)*100:.1f}%)")

# join posts to get category per comment
post_cats = posts[["post_id", "category"]].drop_duplicates()
comments_with_cat = comments.merge(post_cats, on="post_id", how="left")
ct_by_cat = comments_with_cat.groupby(["category", "comment_type"]).size().unstack(fill_value=0)
ct_by_cat_pct = ct_by_cat.div(ct_by_cat.sum(axis=1), axis=0) * 100

info(f"\nComment type % by category:")
info(ct_by_cat_pct.to_string())

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ct_dist.plot(kind="bar", ax=axes[0], color=sns.color_palette("muted", 3))
axes[0].set_title("Comment type distribution (overall)")
axes[0].set_xlabel("")
axes[0].set_ylabel("Count")
axes[0].tick_params(axis="x", rotation=0)
for bar in axes[0].patches:
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                 f"{int(bar.get_height()):,}", ha="center", va="bottom", fontsize=9)

ct_by_cat_pct.plot(kind="bar", ax=axes[1], colormap="tab10")
axes[1].set_title("Comment type % by category")
axes[1].set_xlabel("")
axes[1].set_ylabel("Percentage (%)")
axes[1].tick_params(axis="x", rotation=30)
axes[1].legend(title="Type", bbox_to_anchor=(1.01, 1), loc="upper left")
save_fig("06_comment_type_distribution")


# ---------------------------------------------------------------------------
# 7. INCONSISTENCY FLAGS
# ---------------------------------------------------------------------------
section("7. INCONSISTENCY FLAGS")

short_posts = posts[posts["word_count"] < 20]
info(f"\nPosts with fewer than 20 words: {len(short_posts)} ({len(short_posts)/len(posts)*100:.1f}%)")

# authors with suspiciously many posts
author_counts = posts["author"].value_counts()
prolific = author_counts[author_counts > 50]
info(f"Authors with more than 50 posts: {len(prolific)}")
if len(prolific) > 0:
    info(f"  Top 5: {prolific.head().to_dict()}")

dupes = posts["post_id"].duplicated().sum()
info(f"Duplicate post_ids: {dupes}")

dupes_c = comments["comment_id"].duplicated().sum()
info(f"Duplicate comment_ids: {dupes_c}")

posts_with_comments = comments["post_id"].nunique()
info(f"\nPosts with at least 1 comment in the dataset: {posts_with_comments:,} "
     f"({posts_with_comments/len(posts)*100:.1f}%)")

comments_per_post = comments.groupby("post_id").size()
info(f"Comments per post - mean: {comments_per_post.mean():.1f}, "
     f"median: {comments_per_post.median():.0f}, "
     f"max: {comments_per_post.max()}")


# ---------------------------------------------------------------------------
# 8. CROSS-USER ANALYSIS (RQ2 setup)
# ---------------------------------------------------------------------------
section("8. CROSS-USER ANALYSIS")

medical_cats    = ["chronic_mental", "chronic_physical", "acute_mental", "acute_physical"]
nonmedical_cats = ["non_medical"]

medical_authors    = set(posts[posts["category"].isin(medical_cats)]["author"])
nonmedical_authors = set(posts[posts["category"].isin(nonmedical_cats)]["author"])
crossover          = medical_authors & nonmedical_authors

info(f"\nAuthors with medical posts:      {len(medical_authors):,}")
info(f"Authors with non-medical posts:  {len(nonmedical_authors):,}")
info(f"Authors in both contexts: {len(crossover):,} "
     f"({len(crossover)/max(len(medical_authors),1)*100:.1f}% of medical authors)")

crossover_posts = posts[posts["author"].isin(crossover)]
info(f"Posts from authors in both contexts: {len(crossover_posts):,}")


# ---------------------------------------------------------------------------
# SAVE REPORT
# ---------------------------------------------------------------------------
REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")
log.info(f"\nReport saved to: {REPORT_PATH}")
log.info(f"Plots saved to:  {PLOTS_DIR}/")
log.info("\nEDA complete.")
