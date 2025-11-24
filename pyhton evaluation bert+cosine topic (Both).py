import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.pyplot as plt

# Load your data
df = pd.read_csv("lda_doi_common_comments.csv")

# Step 1: Join comments per LDA topic and DOI label
lda_topic_texts = df.groupby('topic_label')['comment_clean'].apply(lambda x: ' '.join(x)).to_dict()
doi_label_texts = df.groupby('doi_label')['comment_clean'].apply(lambda x: ' '.join(x)).to_dict()

# Step 2: Encode using BERT
model = SentenceTransformer('all-MiniLM-L6-v2')  # Fast and good for semantic similarity

lda_embeddings = model.encode(list(lda_topic_texts.values()), convert_to_tensor=True)
doi_embeddings = model.encode(list(doi_label_texts.values()), convert_to_tensor=True)

# Step 3: Compute cosine similarity
similarity_matrix = cosine_similarity(lda_embeddings.cpu(), doi_embeddings.cpu())

# Step 4: Create heatmap
lda_labels = list(lda_topic_texts.keys())
doi_labels = list(doi_label_texts.keys())

bert_cosine_df = pd.DataFrame(similarity_matrix, index=lda_labels, columns=doi_labels)

plt.figure(figsize=(10, 6))
sns.heatmap(bert_cosine_df, annot=True, cmap="YlOrBr")
plt.title("Cosine Similarity: LDA Topics vs DOI Labels (BERT Embeddings)")
plt.xlabel("DOI Label")
plt.ylabel("LDA Topic")
plt.tight_layout()
plt.show()
