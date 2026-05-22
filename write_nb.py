import json

# This script generates review_analysis.ipynb with 3 sections:
#   Section 3 - Sentiment categorisation (histogram, bar, heatmap)
#   Section 4 - Inconsistent ratings    (histogram, bar, heatmap)
#   Section 5 - Recommended vs not      (histogram, bar, heatmap)

def md(cid, text):
    return {"cell_type": "markdown", "id": cid, "metadata": {}, "source": [text]}

def code(cid, src):
    return {"cell_type": "code", "id": cid, "metadata": {},
            "execution_count": None, "outputs": [], "source": [src]}

cells = [
    md("md_title", "# Amazon Reviews — Sentiment, Inconsistency & Recommendation Analysis\n\nLoads from the pre-cleaned CSV so no re-cleaning needed."),
    code("c_imports", """\
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid')
plt.rcParams.update({'figure.dpi': 120, 'font.family': 'sans-serif'})
"""),
    code("c_load", """\
df = pd.read_csv('../Datasets/Amazon_Product_Review_Cleaned.csv')
df['reviews.numHelpful'] = df['reviews.numHelpful'].fillna(0)
print(f'Shape: {df.shape}')
df[['name', 'device_type', 'reviews.rating', 'sentiment', 'price_usd']].head()
"""),

    # ── Section 3 ───────────────────────────────────────────────────────────
    md("md_s3", "---\n## Section 3 — Sentiment Categorisation\n\nLabels: **Positive** (rating ≥ 4), **Neutral** (= 3), **Negative** (≤ 2)"),
    code("c_s3_counts", """\
print(df['sentiment'].value_counts().to_string())
"""),
    md("md_s3a", "### 3a — Rating Distribution (Histogram)"),
    code("c_s3_hist", """\
rating_counts = df['reviews.rating'].value_counts().sort_index()
colors = ['#d32f2f', '#f57c00', '#fbc02d', '#388e3c', '#1565c0']

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(rating_counts.index.astype(int), rating_counts.values,
              color=colors, width=0.6, edgecolor='white')
for bar in bars:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
            str(int(bar.get_height())), ha='center', fontsize=10, fontweight='bold')

ax.set_title('Rating Distribution', fontsize=14, fontweight='bold')
ax.set_xlabel('Star Rating')
ax.set_ylabel('Number of Reviews')
ax.set_xticks([1, 2, 3, 4, 5])
ax.set_xticklabels(['1 star', '2 star', '3 star', '4 star', '5 star'])
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.savefig('../outputs/charts/s3_hist_ratings.png', dpi=150)
plt.show()
"""),
    md("md_s3b", "### 3b — Top Reviewed Products (Bar Plot)"),
    code("c_s3_bar", """\
top10 = (
    df.groupby('name')['reviews.rating']
      .agg(['count', 'mean'])
      .rename(columns={'count': 'reviews', 'mean': 'avg_rating'})
      .sort_values('reviews', ascending=False)
      .head(10)
)

fig, ax = plt.subplots(figsize=(11, 6))
bars = ax.barh(top10.index[::-1], top10['reviews'][::-1],
               color='#1565c0', height=0.6, edgecolor='white')
for bar, (_, row) in zip(bars, top10[::-1].iterrows()):
    ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
            f"{int(bar.get_width())} reviews  |  avg {row['avg_rating']:.1f} stars",
            va='center', fontsize=9)

ax.set_title('Top 10 Most Reviewed Products', fontsize=14, fontweight='bold')
ax.set_xlabel('Number of Reviews')
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.savefig('../outputs/charts/s3_bar_top_products.png', dpi=150)
plt.show()
"""),
    md("md_s3c", "### 3c — Review Correlations (Heatmap)"),
    code("c_s3_heatmap", """\
df['sentiment_score'] = df['sentiment'].map({'Positive': 1, 'Neutral': 0, 'Negative': -1})

corr = df[['reviews.rating', 'reviews.numHelpful', 'sentiment_score', 'price_usd']].rename(columns={
    'reviews.rating': 'Rating',
    'reviews.numHelpful': 'Helpful Votes',
    'sentiment_score': 'Sentiment',
    'price_usd': 'Price USD'
}).corr()

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdYlGn', vmin=-1, vmax=1,
            linewidths=1.2, linecolor='white', square=True,
            annot_kws={'size': 12, 'weight': 'bold'}, ax=ax)
ax.set_title('Review Correlation Heatmap', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('../outputs/charts/s3_heatmap.png', dpi=150)
plt.show()
"""),

    # ── Section 4 ───────────────────────────────────────────────────────────
    md("md_s4", "---\n## Section 4 — Products with Inconsistent Ratings\n\nHigh standard deviation in ratings means reviewers strongly disagree about the product."),
    code("c_s4_find", """\
rating_stats = (
    df.groupby('name')['reviews.rating']
      .agg(['mean', 'std', 'count'])
      .rename(columns={'mean': 'avg', 'std': 'std_dev', 'count': 'reviews'})
      .dropna(subset=['std_dev'])
      .query('reviews >= 10')
      .sort_values('std_dev', ascending=False)
)
print('Top inconsistent products:')
print(rating_stats.head(10).round(2).to_string())
"""),
    md("md_s4a", "### 4a — Std Dev Distribution (Histogram)"),
    code("c_s4_hist", """\
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(rating_stats['std_dev'], bins=20, color='#5c6bc0', edgecolor='white')
ax.axvline(rating_stats['std_dev'].mean(), color='#e53935', linewidth=2,
           linestyle='--', label=f"Mean = {rating_stats['std_dev'].mean():.2f}")
ax.set_title('Distribution of Rating Std Dev per Product', fontsize=13, fontweight='bold')
ax.set_xlabel('Std Deviation of Ratings')
ax.set_ylabel('Number of Products')
ax.legend(fontsize=10)
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.savefig('../outputs/charts/s4_hist_inconsistency.png', dpi=150)
plt.show()
"""),
    md("md_s4b", "### 4b — Most Inconsistent Products (Bar Plot)"),
    code("c_s4_bar", """\
top_inc = rating_stats.head(10).reset_index()

fig, ax = plt.subplots(figsize=(11, 6))
bars = ax.barh(top_inc['name'][::-1], top_inc['std_dev'][::-1],
               color='#ef5350', height=0.6, edgecolor='white')
for bar, row in zip(bars, top_inc[::-1].itertuples()):
    ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
            f"std={row.std_dev:.2f}  avg={row.avg:.1f}  ({int(row.reviews)} reviews)",
            va='center', fontsize=8.5)

ax.set_title('Top 10 Most Inconsistently Rated Products', fontsize=13, fontweight='bold')
ax.set_xlabel('Rating Std Deviation')
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.savefig('../outputs/charts/s4_bar_inconsistent.png', dpi=150)
plt.show()
"""),
    md("md_s4c", "### 4c — Correlations for Inconsistent Products (Heatmap)"),
    code("c_s4_heatmap", """\
inconsistent_names = rating_stats.head(10).index.tolist()
df_inc = df[df['name'].isin(inconsistent_names)].copy()
df_inc['sentiment_score'] = df_inc['sentiment'].map({'Positive': 1, 'Neutral': 0, 'Negative': -1})

corr_inc = df_inc[['reviews.rating', 'reviews.numHelpful', 'sentiment_score', 'price_usd']].rename(columns={
    'reviews.rating': 'Rating',
    'reviews.numHelpful': 'Helpful Votes',
    'sentiment_score': 'Sentiment',
    'price_usd': 'Price USD'
}).corr()

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(corr_inc, annot=True, fmt='.2f', cmap='RdYlGn', vmin=-1, vmax=1,
            linewidths=1.2, linecolor='white', square=True,
            annot_kws={'size': 12, 'weight': 'bold'}, ax=ax)
ax.set_title('Correlations — Inconsistent Products Only', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('../outputs/charts/s4_heatmap_inconsistent.png', dpi=150)
plt.show()
"""),

    # ── Section 5 ───────────────────────────────────────────────────────────
    md("md_s5", "---\n## Section 5 — Recommended vs Not Recommended\n\nThe dataset has no verified/unverified column. `reviews.doRecommend` (True/False) is the closest equivalent."),
    code("c_s5_split", """\
df_rec = df[df['reviews.doRecommend'] == True]
df_not = df[df['reviews.doRecommend'] == False]

print(f"Recommended:     {len(df_rec)} reviews  |  avg rating: {df_rec['reviews.rating'].mean():.2f}")
print(f"Not recommended: {len(df_not)} reviews  |  avg rating: {df_not['reviews.rating'].mean():.2f}")
"""),
    md("md_s5a", "### 5a — Rating Distribution Comparison (Histogram)"),
    code("c_s5_hist", """\
fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

for ax, subset, label, color in zip(
    axes,
    [df_rec, df_not],
    ['Recommended', 'Not Recommended'],
    ['#388e3c', '#e53935']
):
    counts = subset['reviews.rating'].value_counts().sort_index()
    ax.bar(counts.index.astype(int), counts.values, color=color, edgecolor='white', width=0.6)
    for x, y in zip(counts.index.astype(int), counts.values):
        ax.text(x, y + 1, str(y), ha='center', fontsize=9, fontweight='bold')
    ax.set_title(label, fontsize=13, fontweight='bold')
    ax.set_xlabel('Star Rating')
    ax.set_ylabel('Reviews')
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.spines[['top', 'right']].set_visible(False)

plt.suptitle('Rating Distribution: Recommended vs Not Recommended', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('../outputs/charts/s5_hist_rec_vs_not.png', dpi=150)
plt.show()
"""),
    md("md_s5b", "### 5b — Avg Rating per Product (Bar Plot)"),
    code("c_s5_bar", """\
top_names = df['name'].value_counts().head(10).index.tolist()

rec_avg = df_rec[df_rec['name'].isin(top_names)].groupby('name')['reviews.rating'].mean().rename('Recommended')
not_avg = df_not[df_not['name'].isin(top_names)].groupby('name')['reviews.rating'].mean().rename('Not Recommended')

compare = pd.concat([rec_avg, not_avg], axis=1).reindex(top_names).dropna()
short_names = [n[:28] + '...' if len(n) > 28 else n for n in compare.index]

x = np.arange(len(compare))
w = 0.38

fig, ax = plt.subplots(figsize=(13, 6))
ax.bar(x - w/2, compare['Recommended'],     width=w, color='#388e3c', label='Recommended',     edgecolor='white')
ax.bar(x + w/2, compare['Not Recommended'], width=w, color='#e53935', label='Not Recommended', edgecolor='white')

ax.set_xticks(x)
ax.set_xticklabels(short_names, rotation=30, ha='right', fontsize=9)
ax.set_title('Avg Rating per Product — Recommended vs Not Recommended', fontsize=13, fontweight='bold')
ax.set_ylabel('Average Rating')
ax.set_ylim(0, 6)
ax.legend(fontsize=10)
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.savefig('../outputs/charts/s5_bar_rec_vs_not.png', dpi=150)
plt.show()
"""),
    md("md_s5c", "### 5c — Correlation Heatmaps Side by Side"),
    code("c_s5_heatmap", """\
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for ax, subset, label in zip(axes, [df_rec, df_not], ['Recommended', 'Not Recommended']):
    sub = subset.copy()
    sub['sentiment_score'] = sub['sentiment'].map({'Positive': 1, 'Neutral': 0, 'Negative': -1})
    corr = sub[['reviews.rating', 'reviews.numHelpful', 'sentiment_score', 'price_usd']].rename(columns={
        'reviews.rating': 'Rating',
        'reviews.numHelpful': 'Helpful Votes',
        'sentiment_score': 'Sentiment',
        'price_usd': 'Price USD'
    }).corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdYlGn', vmin=-1, vmax=1,
                linewidths=1.2, linecolor='white', square=True,
                annot_kws={'size': 11, 'weight': 'bold'}, ax=ax)
    ax.set_title(label, fontsize=12, fontweight='bold')
    ax.tick_params(axis='x', rotation=20, labelsize=9)
    ax.tick_params(axis='y', rotation=0,  labelsize=9)

plt.suptitle('Correlation Heatmaps: Recommended vs Not Recommended', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('../outputs/charts/s5_heatmap_rec_vs_not.png', dpi=150)
plt.show()
"""),
]

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.9.0"}
    },
    "cells": cells
}

out = r'c:\Users\Mukund\Amzon_Product_review_analysis_problem_4\notebooks\review_analysis.ipynb'
with open(out, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f'Written {len(cells)} cells to review_analysis.ipynb')
for c in cells:
    preview = ''.join(c['source'])[:55].replace('\n', ' ')
    print(f'  [{c["cell_type"]:8}] {c["id"]:18} | {preview}')

