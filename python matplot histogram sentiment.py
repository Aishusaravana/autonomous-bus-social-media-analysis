import pandas as pd
import matplotlib.pyplot as plt

# ✅ Load your final sentiment result file
file_path = r"C:\Users\aishw\OneDrive\Desktop\PBL Autonomous buses\sentiment_only_output.xlsx"
df = pd.read_excel(file_path)

# ✅ Plot the histogram of sentiment scores
plt.figure(figsize=(10, 5))
plt.hist(df['sentiment_score'], bins=30, edgecolor='black')
plt.title("Sentiment Score Distribution")
plt.xlabel("Sentiment Score (-1.0 = Negative, 1.0 = Positive)")
plt.ylabel("Number of Comments")
plt.grid(axis='y')
plt.tight_layout()
plt.show()
