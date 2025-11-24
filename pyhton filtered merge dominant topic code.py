import pandas as pd

# === Load the merged file ===
file_path = r"C:\Users\aishw\OneDrive\Desktop\PBL Autonomous buses\final_topic_sentiment_data_fixed.xlsx"
df = pd.read_excel(file_path)

# === Filter only topics 0 to 3 ===
df = df[df['dominant_topic'].isin([0, 1, 2, 3])]

# === Assign correct topic labels ===
topic_labels = {
    0: "Technological Superiority",
    1: "Infrastructure & Transit Compatibility",
    2: "Human Driving vs Automation Concerns",
    3: "Initial Experiences & Public Reaction"
}
df['topic_label'] = df['dominant_topic'].map(topic_labels)

# === Save cleaned version ===
output_path = r"C:\Users\aishw\OneDrive\Desktop\PBL Autonomous buses\final_lda_sentiment_data_cleaned.xlsx"
df.to_excel(output_path, index=False)

print("✅ Cleaned file saved at:")
print(output_path)
print("Final row count:", len(df))
print("Topics:", df['dominant_topic'].unique())
