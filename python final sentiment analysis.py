import pandas as pd
import os
from google.cloud import language_v1

print("🚀 Sentiment analysis on reconstructed comments is running...")

# ✅ Step 1: Set up credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"C:\Users\aishw\OneDrive\Desktop\PBL Autonomous buses\sentiment-analysis-453802-aa0fef4c1b62.json"

# ✅ Step 2: Load the reconstructed comments file
try:
    file_path = r"C:\Users\aishw\OneDrive\Desktop\PBL Autonomous buses\reconstructed_for_sentiment.csv"
    df = pd.read_csv(file_path)
    print(f"✅ File loaded successfully: {file_path}")
except Exception as e:
    print(f"❌ Failed to load CSV file: {e}")
    exit()

# ✅ Step 3: Check column and drop blanks
if 'reconstructed_text' not in df.columns:
    print("❌ 'reconstructed_text' column not found in the file.")
    print("🧠 Available columns are:", df.columns.tolist())
    exit()

df = df.dropna(subset=['reconstructed_text'])

# ✅ Step 4: Initialize Google Cloud NLP client
try:
    client = language_v1.LanguageServiceClient()
except Exception as e:
    print(f"❌ Failed to initialize Google NLP client: {e}")
    exit()

# ✅ Step 5: Define sentiment function
def analyze_sentiment(text):
    if not text or not str(text).strip():
        return 0.0, 0.0
    try:
        document = language_v1.Document(content=text, type_=language_v1.Document.Type.PLAIN_TEXT)
        sentiment = client.analyze_sentiment(request={'document': document}).document_sentiment
        return sentiment.score, sentiment.magnitude
    except Exception as e:
        print(f"⚠️ Error analyzing comment: '{text[:50]}...' → {e}")
        return 0.0, 0.0

# ✅ Step 6: Apply sentiment analysis
try:
    print("⏳ Analyzing sentiment for each reconstructed comment...")
    df[['sentiment_score', 'sentiment_magnitude']] = df['reconstructed_text'].apply(
        lambda x: pd.Series(analyze_sentiment(str(x)))
    )
    print("✅ Sentiment analysis complete!")
except Exception as e:
    print(f"❌ Error during sentiment analysis: {e}")
    exit()

# ✅ Step 7: Save results to Excel
try:
    output_path = r"C:\Users\aishw\OneDrive\Desktop\PBL Autonomous buses\reconstructed_comments_with_sentiment.xlsx"
    df.to_excel(output_path, index=False)
    print(f"📁 File saved with sentiment results:\n{output_path}")
except Exception as e:
    print(f"❌ Failed to save Excel file: {e}")
