import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.pyplot as plt

# Load the common file
df = pd.read_csv("lda_doi_common_comments.csv")

# Step 1: Create text per LDA topic
lda_topic_texts = df.groupby('topic_label')['comment_clean'].apply(lambda x: ' '.join(x)).to_dict()

# Step 2: Create text per DOI label
doi_label_texts = df.groupby('doi_label')['comment_clean'].apply(lambda x: ' '.join(x)).to_dict()

# Step 3: Combine all text groups for vectorization
all_groups = list(lda_topic_texts.keys()) + list(doi_label_texts.keys())
all_texts = list(lda_topic_texts.values()) + list(doi_label_texts.values())

# Step 4: Vectorize using TF-IDF
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(all_texts)

# Step 5: Compute cosine similarity matrix
similarity_matrix = cosine_similarity(tfidf_matrix)

# Step 6: Create a labeled heatmap
lda_labels = list(lda_topic_texts.keys())
doi_labels = list(doi_label_texts.keys())
cosine_df = pd.DataFrame(
    similarity_matrix[:len(lda_labels), len(lda_labels):],  # only LDA ↔ DOI block
    index=lda_labels,
    columns=doi_labels
)

# Step 7: Plot heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(cosine_df, annot=True, cmap="YlGnBu")
plt.title("Cosine Similarity: LDA Topics vs DOI Labels (TF-IDF)")
plt.xlabel("DOI Label")
plt.ylabel("LDA Topic")
plt.tight_layout()
plt.show()

