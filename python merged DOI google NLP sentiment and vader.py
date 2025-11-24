import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# --- Load your file ---
df = pd.read_csv("lda_doi_common_comments_deduplicated.csv")

# --- Run VADER sentiment analysis ---
analyzer = SentimentIntensityAnalyzer()
df['vader_score'] = df['reconstructed_text'].apply(lambda x: analyzer.polarity_scores(str(x))['compound'])

# --- Save updated file with VADER score ---
df.to_csv("lda_doi_common_with_vader.csv", index=False)

print("✅ File saved with VADER sentiment as 'lda_doi_common_with_vader.csv'")
