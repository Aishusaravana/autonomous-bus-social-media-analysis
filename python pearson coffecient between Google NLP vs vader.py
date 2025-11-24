import pandas as pd
from scipy.stats import pearsonr
import seaborn as sns
import matplotlib.pyplot as plt

# --- Load your merged file ---
df = pd.read_csv("lda_doi_common_with_vader.csv")

# --- Drop missing or invalid values (just in case) ---
df = df.dropna(subset=['vader_score', 'sentiment_score_lda'])

# --- Calculate Pearson correlation ---
r, p = pearsonr(df['vader_score'], df['sentiment_score_lda'])

print(f"✅ Pearson’s r (VADER vs LDA): {r:.4f}")
print(f"   p-value: {p:.4e}")

# --- Scatterplot with regression line ---
plt.figure(figsize=(8,6))
sns.regplot(
    x='vader_score', 
    y='sentiment_score_lda', 
    data=df, 
    scatter_kws={"alpha": 0.5}, 
    line_kws={"color": "red"}
)
plt.title(f"VADER vs LDA Sentiment Scores\nPearson’s r = {r:.2f}")
plt.xlabel("VADER Sentiment Score (Compound)")
plt.ylabel("LDA Sentiment Score (Google NLP)")
plt.grid(True)
plt.tight_layout()
plt.savefig("vader_vs_lda_sentiment_correlation.png", dpi=300)
plt.show()
