import pandas as pd
from scipy.stats import pearsonr

# Load the CSV you downloaded earlier (with 2819 matched comments)
df = pd.read_csv("lda_doi_common_comments.csv")

# Check columns (optional)
print(df.columns)

# Calculate Pearson correlation between LDA and DOI sentiment scores
r_value, p_value = pearsonr(df['sentiment_score_lda'], df['sentiment_score_doi'])

# Display result
print(f"Pearson’s r: {r_value:.4f}")
print(f"P-value: {p_value:.4e}")

# Optional: Show scatterplot for visual confirmation
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(6, 6))
sns.scatterplot(x='sentiment_score_lda', y='sentiment_score_doi', data=df, alpha=0.5)
plt.axline((0, 0), slope=1, linestyle='--', color='gray')
plt.title("LDA vs DOI Sentiment Scores")
plt.xlabel("LDA Sentiment")
plt.ylabel("DOI Sentiment")
plt.tight_layout()
plt.show()
