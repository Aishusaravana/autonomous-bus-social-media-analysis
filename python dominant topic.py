import pandas as pd
import pickle
from gensim import corpora, models

# === Paths ===
sentiment_path = r"C:\Users\aishw\OneDrive\Desktop\PBL Autonomous buses\reconstructed_comments_with_sentiment.xlsx"
texts_path = r"C:\Users\aishw\OneDrive\Desktop\PBL Autonomous buses\texts_filtered.pkl"
dict_path = r"C:\Users\aishw\OneDrive\Desktop\PBL Autonomous buses\lda_dictionary_4topics.gensim"
lda_model_path = r"C:\Users\aishw\OneDrive\Desktop\PBL Autonomous buses\lda_model_4topics.gensim"
output_path = r"C:\Users\aishw\OneDrive\Desktop\PBL Autonomous buses\final_lda_sentiment_data.xlsx"

# === Load Data ===
df_sentiment = pd.read_excel(sentiment_path)
with open(texts_path, "rb") as f:
    texts = pickle.load(f)

dictionary = corpora.Dictionary.load(dict_path)
lda_model = models.LdaModel.load(lda_model_path)

print("🧠 Topics in model:", lda_model.num_topics)

# === Rebuild Corpus from texts ===
corpus = [dictionary.doc2bow(text) for text in texts]

# === Get Dominant Topic for each doc ===
document_topic = []
for doc in corpus:
    topic_probs = lda_model.get_document_topics(doc)
    if topic_probs:
        dominant_topic = max(topic_probs, key=lambda x: x[1])[0]
        document_topic.append(dominant_topic)
    else:
        document_topic.append(None)

# === Create processed_text (for merging) ===
topic_df = pd.DataFrame({
    'processed_text': [' '.join(text) for text in texts],
    'dominant_topic': document_topic
})

# === Merge with sentiment data on reconstructed_text ===
df_sentiment['reconstructed_text'] = df_sentiment['reconstructed_text'].str.strip().str.lower()
topic_df['processed_text'] = topic_df['processed_text'].str.strip().str.lower()

merged_df = pd.merge(df_sentiment, topic_df, left_on='reconstructed_text', right_on='processed_text', how='inner')

# === Assign Topic Labels (only for valid 0–3) ===
topic_labels = {
    0: "Technological Superiority",
    1: "Infrastructure & Transit Compatibility",
    2: "Human Driving vs Automation Concerns",
    3: "Initial Experiences & Public Reaction"
}

merged_df['topic_label'] = merged_df['dominant_topic'].map(topic_labels)

# === Save Final Output ===
merged_df.to_excel(output_path, index=False)
print(f"✅ Final LDA + Sentiment file saved:\n{output_path}")
print(f"📊 Total Rows in Final File: {len(merged_df)}")
