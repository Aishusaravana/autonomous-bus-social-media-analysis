import pickle

# Load your tokenized texts
with open(r"C:\Users\aishw\OneDrive\Desktop\PBL Autonomous buses\texts_filtered.pkl", "rb") as f:
    texts = pickle.load(f)

# Reconstruct into full sentences
reconstructed_comments = [" ".join(tokens) for tokens in texts]

# Save to CSV to upload to Google NLP
import pandas as pd
df = pd.DataFrame({"reconstructed_text": reconstructed_comments})
df.to_csv(r"C:\Users\aishw\OneDrive\Desktop\PBL Autonomous buses\reconstructed_for_sentiment.csv", index=False)

print("✅ Reconstructed comments saved. Ready for sentiment analysis.")
