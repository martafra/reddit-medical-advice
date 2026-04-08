# Analysis Methodology
CS7IS4 / Text Analytics – Group 10

This document explains how `analysis.py` uses the LDA and clustering outputs
to answer the three Research Questions and test each hypothesis.

---

## Data Sources

| File | Source | Used for |
|---|---|---|
| `documents_with_topics.csv` | LDA (per corpus) | Main analysis dataframe — contains all original feature columns plus `dominant_topic` and `topic_probability` |
| `topic_words.csv` | LDA (per corpus) | Topic naming and RQ3 topic overlap |
| `documents_with_clusters.csv` | Clustering (per corpus) | Merged in to add `cluster` column for per-cluster breakdowns |
| `cluster_words.csv` | Clustering (per corpus) | Cluster naming |

The three corpora are **medical posts**, **non-medical posts**, and **comments**.
Medical posts are further split by category: `chronic_mental`, `chronic_physical`,
`acute_mental`, `acute_physical`.

---

## Pre-computed Feature Columns (from `features.py`)

These columns are already present in `documents_with_topics.csv` and are used
directly by the analysis script:

| Column | Description |
|---|---|
| `sentiment_score` | VADER compound score (−1 to +1), computed on `text_clean` |
| `empathy_score` | Empathy phrase count per 100 words |
| `shared_experience` | Binary flag: 1 if post contains at least one shared-experience phrase, else 0 |
| `advice_acceptance` | Advice-acceptance phrase count per 100 words |
| `uncertainty_density` | Hedging/uncertainty word count per 100 words |

The analysis script adds three more:

| Column | Description |
|---|---|
| `vader_compound` | Fresh VADER score computed in analysis (on the same text column) |
| `clarity_density` | Clarity phrase count per 100 words (e.g. "easy to understand", "makes sense") |
| `accuracy_density` | Accuracy phrase count per 100 words (e.g. "accurate", "evidence based") |
| `word_count` | Total word count of the text |

---

## Topic and Cluster Naming

### How auto-naming works
For each topic/cluster, the script takes the top 10 words from
`topic_words.csv` / `cluster_words.csv` (ranked by probability / TF-IDF weight)
and joins the first three with underscores as an auto-generated label
(e.g. `pain_chronic_treatment`).

### How to apply manual names
On first run, naming CSVs are saved to `output/analysis/naming/`:

```
lda_medical_topics.csv
lda_nonmedical_topics.csv
lda_comments_topics.csv
cluster_medical.csv
cluster_nonmedical.csv
cluster_comments.csv
```

Each file has four columns: `topic_id` (or `cluster_id`), `auto_name`,
`top_words`, `manual_name`. Fill in `manual_name` for any topic you want to
rename, then re-run `analysis.py`. The script uses `manual_name` when it is
filled and falls back to `auto_name` otherwise.

---

## RQ1 – Severity and Nature of Medical Conditions

**Research Question:** How does the nature and severity of medical conditions
influence how the patient reacts to the received advice?

**Severity mapping used:**
- `chronic_mental`, `chronic_physical` → **chronic** (more severe)
- `acute_mental`, `acute_physical` → **acute** (less severe)

**Nature mapping used:**
- `*_mental` → **mental**
- `*_physical` → **physical**

### Hypothesis H1
> Advice given in cases involving more severe conditions will show a higher
> prevalence of phrases related to empathy and emotional support.

**What is measured:** `empathy_score` (empathy phrase density per 100 words,
pre-computed by `features.py` using 23 curated phrases such as
"you're not alone", "I hear you", "sending hugs").

**Statistical test:** Mann-Whitney U test (one-tailed, chronic > acute).
This non-parametric test is appropriate because phrase density distributions
are typically right-skewed and non-normal.

**Effect size:** Rank-biserial correlation `r_rb = 1 − 2U / (n₁·n₂)`.
Interpretation: |r| > 0.1 small, > 0.3 medium, > 0.5 large.

**Outputs:**
- `rq1/h1_group_stats.csv` — mean/median empathy and sentiment by severity × nature
- `rq1/h1_results.txt` — U statistic, p-value, effect size, verdict
- `rq1/h1_boxplots.png` — empathy and sentiment distributions by severity and nature
- `rq1/empathy_by_topic_severity.csv` — per-LDA-topic breakdown
- `rq1/empathy_by_cluster_severity.csv` — per-cluster breakdown

**How to interpret:**
- p < 0.05 and chronic mean > acute mean → H1 **supported**
- Check the effect size: a statistically significant but tiny effect may have
  limited practical relevance

### Hypothesis H2
> In cases involving more severe conditions, the empathy of the response will
> be more strongly correlated with positive sentiment than other aspects such
> as clarity and accuracy.

**What is measured:** Within **chronic posts only**, compute Spearman
correlations of `sentiment_score` with each of:
1. `empathy_score` (empathy + emotional support)
2. `clarity_density` (language signalling clear, helpful advice)
3. `accuracy_density` (language signalling factually correct advice)

**Statistical test:** Spearman correlation (non-parametric; robust to
non-normal sentiment and phrase-density distributions).

**Verdict logic:** H2 is supported if `r(empathy, sentiment)` is greater than
both `r(clarity, sentiment)` and `r(accuracy, sentiment)` in the chronic group.

**Outputs:**
- `rq1/h2_correlations.csv` — Spearman r and p-value for each metric
- `rq1/h2_results.txt` — comparison table and verdict
- `rq1/h2_correlation_bars.png` — bar chart of the three correlations
- `rq1/h2_scatter_plots.png` — scatter plots (empathy vs sentiment) for chronic and acute

**How to interpret:**
- If empathy has the highest r AND is significant (p < 0.05) → H2 **supported**
- A non-significant empathy correlation with a higher r than the others gives
  weak support — worth discussing
- Compare chronic vs acute scatter plots: if the regression slope is steeper
  in chronic, that further supports H2

---

## RQ2 – Shared Experience and Advice Acceptance

**Research Question:** To what extent does the relatability effect of
"shared experience" influence the acceptance of the advice?

**Hypothesis:**
> Advice-acceptance language ("thanks", "makes sense") in relevant Reddit
> threads will be correlated with the responses containing "shared experience"
> phrases ("in my case", etc.).

**What is measured:**
- `shared_experience` (binary flag, 0/1) — pre-computed by `features.py`
  using 20 curated phrases (e.g. "same thing happened to me", "i can relate")
- `advice_acceptance` (density per 100 words) — pre-computed by `features.py`
  using 21 curated phrases (e.g. "i'll try that", "that makes sense")

**This analysis is run on all three corpora** (medical posts, non-medical
posts, comments) so you can compare the effect across contexts.

**Statistical tests:**

1. **Spearman correlation** (point-biserial, since `shared_experience` is binary):
   Tests whether posts with shared-experience language also have higher
   advice-acceptance scores.

2. **Mann-Whitney U test** (one-tailed, with-SE > without-SE):
   Compares `advice_acceptance` density in posts WITH vs WITHOUT shared-experience
   phrases. Provides a complementary group-comparison perspective.

**Effect size:** Rank-biserial correlation from Mann-Whitney U.

**Outputs (per corpus):**
- `rq2/results_<corpus>.csv` — correlation, U statistic, effect size, verdict
- `rq2/violin_<corpus>.png` — advice-acceptance distribution split by shared-experience flag
- `rq2/by_topic_<corpus>.csv` — % shared-experience and mean acceptance per LDA topic
- `rq2/by_cluster_<corpus>.csv` — same aggregation per cluster

**How to interpret:**
- Positive r with p < 0.05 → hypothesis **supported**
- The `by_topic` and `by_cluster` tables reveal **which topics** show the
  strongest link between shared experience and acceptance — useful for
  qualitative interpretation
- Comparing medical vs non-medical results for RQ2 also feeds into RQ3

---

## RQ3 – Generalisation to Non-Medical Posts

**Research Question:** Do the linguistic patterns found in reactions to
important medical advice also generalise to other areas and scenarios
(e.g. financial advice)?

### Hypothesis H1
> The top-ranked topics identified by LDA in medical advice reactions will
> also emerge in the case of non-medical advice.

**Method — Jaccard Similarity on Top Words:**

For each LDA topic, the top-10 words (by probability from `topic_words.csv`)
form a word set. For every pair of (medical topic, non-medical topic), the
script computes:

```
Jaccard(A, B) = |A ∩ B| / |A ∪ B|
```

This measures how much the defining vocabulary of a medical topic overlaps with
that of a non-medical topic. A score of 0 means no shared words; 1 means
identical word sets.

**Threshold:** A mean best-match Jaccard similarity > 0.10 is used as the
criterion for "generalisation". This threshold is conservative — even a small
lexical overlap (1–2 shared words out of 10) is meaningful if the words are
domain-specific.

**Outputs:**
- `rq3/topic_similarity_matrix.csv` — full N_med × N_nonmed Jaccard matrix
- `rq3/topic_best_matches.csv` — for each medical topic, its closest non-medical match and their top words
- `rq3/h1_results.txt` — mean similarity and verdict
- `rq3/h1_topic_similarity_heatmap.png` — colour-coded similarity matrix

**How to interpret:**
- High-similarity pairs (Jaccard > 0.2): the same thematic vocabulary appears
  in both corpora — strong evidence of generalisation
- Zero-similarity topics: these are domain-specific to medical discourse
- The `topic_best_matches.csv` lets you manually inspect whether the shared
  words are substantively meaningful or coincidental function words

### Hypothesis H2
> Correlations between positive sentiment towards advice and (a) advice
> length and (b) density of uncertainty vocabulary will generalise to
> non-medical scenarios.

**What is measured:**
- `word_count` — proxy for advice length (longer posts may signal more
  detailed, effort-ful sharing)
- `uncertainty_density` — density of hedging language (e.g. "maybe", "I think",
  "possibly"), pre-computed by `features.py`
- `sentiment_score` — VADER compound score

Both correlations are computed separately for medical posts and non-medical
posts using Spearman's ρ.

**Generalisation criterion:**
1. Same direction (both positive or both negative) in both corpora
2. Both statistically significant (p < 0.05)

**Formal comparison — Fisher's Z-test:**
Tests whether the correlation coefficients in the two corpora are
statistically different from each other. A non-significant Fisher Z (p > 0.05)
means the correlations do not differ significantly — i.e., they generalise.

```
z_Fisher = (arctanh(r₁) − arctanh(r₂)) / sqrt(1/(n₁−3) + 1/(n₂−3))
```

**Outputs:**
- `rq3/h2_correlations.csv` — Spearman r and p-value for each feature × corpus
- `rq3/h2_fisher_tests.csv` — Fisher Z-test results for each feature
- `rq3/h2_results.txt` — per-feature verdict
- `rq3/h2_correlation_bars.png` — side-by-side bar chart (medical vs non-medical)

**How to interpret:**
- Same direction + both significant + non-significant Fisher Z → correlation
  **generalises** (the two corpora show the same relationship at similar strength)
- Same direction + both significant + significant Fisher Z → the relationship
  exists in both, but at different magnitudes (partial generalisation)
- Opposite directions or one non-significant → does **not** generalise

---

## Statistical Tests Summary

| RQ | Hypothesis | Test | Why |
|---|---|---|---|
| RQ1 | H1 | Mann-Whitney U (one-tailed) | Two independent groups, non-normal phrase densities |
| RQ1 | H2 | Spearman correlation | Non-parametric; robust to skewed distributions |
| RQ2 | H | Spearman + Mann-Whitney U | Binary predictor × continuous outcome |
| RQ3 | H1 | Jaccard similarity | Set-overlap measure for word-level topic comparison |
| RQ3 | H2 | Spearman + Fisher's Z | Independent correlation comparison across two corpora |

Significance level: **α = 0.05** throughout.

---

## Limitations

1. **Phrase lists** — empathy, acceptance, and shared-experience phrases are
   manually curated. They may miss paraphrases or colloquial expressions common
   on Reddit. A learned classifier (e.g. fine-tuned BERT) would improve recall.

2. **Binary shared_experience flag** — using presence/absence loses information
   about how extensively shared experience is discussed. The correlation tests
   are consequently conservative.

3. **Jaccard similarity for topic overlap** — sensitive to the number of top
   words chosen (`TOP_WORDS_FOR_SIMILARITY`). Larger sets dilute rare
   domain-specific words; smaller sets are noisier. Consider checking
   sensitivity by varying this parameter (e.g. 5, 10, 15).

4. **Cross-corpus comparisons** — medical and non-medical posts may differ in
   length, community norms, and writing style, which could independently
   explain differences in phrase densities rather than condition severity alone.

5. **No causal claims** — all tests are correlational. A correlation between
   shared experience and acceptance language does not establish that shared
   experience *causes* acceptance.
