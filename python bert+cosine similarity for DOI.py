import pickle
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util

# === STEP 1: Load Preprocessed Tokenized Comments ===
with open(r"C:\Users\aishw\OneDrive\Desktop\PBL Autonomous buses\texts_filtered.pkl", "rb") as f:
    texts = pickle.load(f)

comments = [" ".join(tokens) for tokens in texts]

# === STEP 2: Load Sentence-BERT Model ===
model = SentenceTransformer('all-MiniLM-L6-v2')

# === STEP 3: Define DOI Reference Sentences ===
doi_sentences = {
    "Relative Advantage": "Autonomous buses are safer, more efficient, and better than traditional transportation.",
    "Compatibility": "Autonomous buses fit into our lifestyle and daily routines easily.",
    "Complexity": "Autonomous buses are confusing and difficult to understand or operate.",
    "Trialability": "Autonomous buses can be tested or tried before fully using them.",
    "Observability": "The benefits of autonomous buses can be easily seen and shared publicly."
}

doi_embeddings = {label: model.encode(sentence) for label, sentence in doi_sentences.items()}

# === STEP 4: Generate Comment Embeddings ===
comment_embeddings = model.encode(comments, show_progress_bar=True)

# === STEP 5: Classify and Record Scores ===
labels = []
similarity_scores = []

for comment_emb in tqdm(comment_embeddings):
    similarities = {
        label: float(util.cos_sim(comment_emb, doi_embeddings[label]))
        for label in doi_sentences
    }
    best_label = max(similarities, key=similarities.get)
    labels.append(best_label)
    similarity_scores.append(similarities[best_label])

# === STEP 6: Save Results to CSV with Scores ===
df = pd.DataFrame({
    "comment_text": comments,
    "doi_label": labels,
    "similarity_score": similarity_scores
})
df.to_csv(r"C:\Users\aishw\OneDrive\Desktop\PBL Autonomous buses\bert_doi_classified_with_scores.csv", index=False)

print("✅ Classification complete. CSV with scores saved.")
