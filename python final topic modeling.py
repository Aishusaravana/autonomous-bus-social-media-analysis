import pandas as pd
import nltk
from nltk.corpus import stopwords
import spacy
import pickle
import langdetect
from gensim import corpora
from gensim.models import LdaModel, CoherenceModel
import pyLDAvis
import pyLDAvis.gensim
import csv

# ✅ Download NLTK Stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# ✅ Custom Stopwords (positive sentiment words kept)
custom_stopwords = [
    # General
    "the", "be", "to", "and", "in", "it", "that", "for", "do", "with", "so",
    "but", "we", "from", "if", "as", "there", "would", "or", "no", "all", "more", 
    "make", "well", "have", "this", "you", "on", "its", "they", "can", "system", "service",

    # Geo-political
    "china", "chinese", "india", "indian", "america", "american", "country", "city",
    "world", "place", "government", "korea", "japan", "pakistan", "guangzhou", "shanghai",
    "beijing", "states", "united", "western", "asian",

    # People
    "people", "citizen", "guy", "man", "woman", "child", "everyone", "someone", "nobody",
    "user", "customer", "passenger",

    # Social Media
    "video", "watch", "subscribe", "channel", "like", "comment", "click", "share", "upload",
    "youtube", "views",

    # Religion / Emotion (positive words kept)
    "jesus", "christ", "lord", "bless", "pray", "amen", "hope", "love", "hate", "god", "faith",
    "thank", "thanks",

    # Vague / Filler
    "thing", "stuff", "look", "something", "everything", "anything", "nothing", "some", "someone",
    "much", "many", "time", "year", "day", "life", "long", "short", "come", "go", "keep", "make",
    "take", "get", "know", "need", "want", "work", "use"
]
stop_words = stop_words.union(custom_stopwords)

# ✅ Load SpaCy
nlp = spacy.load('en_core_web_sm')

# ✅ Preprocessing
def preprocess_text(text):
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if token.is_alpha]
    tokens = [word.lower() for word in tokens if word.lower() not in stop_words and len(word) > 3]
    return tokens

def is_english(tokens):
    try:
        return langdetect.detect(" ".join(tokens)) == 'en'
    except:
        return False

# ✅ Main Execution
if __name__ == "__main__":
    # ✅ Load Dataset
    file_path = r"C:\Users\aishw\OneDrive\Desktop\PBL Autonomous buses\cleaned_youtube_comments_no_blanks.xlsx"
    df = pd.read_excel(file_path, engine="openpyxl")
    df = df.dropna(subset=['translated_comment'])

    # ✅ Preprocess Comments
    df['processed_text'] = df['translated_comment'].apply(preprocess_text)
    df = df[df['processed_text'].apply(lambda x: len(x) > 0)]
    texts = df['processed_text'].tolist()
    texts = [text for text in texts if is_english(text)]

    # ✅ Save Processed Texts
    with open(r"C:\Users\aishw\OneDrive\Desktop\PBL Autonomous buses\texts_filtered.pkl", "wb") as f:
        pickle.dump(texts, f, protocol=pickle.HIGHEST_PROTOCOL)

    # ✅ Dictionary & Corpus
    dictionary = corpora.Dictionary(texts)
    dictionary.filter_extremes(no_below=5, no_above=0.6)
    corpus = [dictionary.doc2bow(text) for text in texts]

    # ✅ Train LDA Model (6 Topics + More Passes/Iterations)
    lda_model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=6,
        passes=30,
        iterations=500,
        random_state=42,
        alpha='auto',
        eta='auto',
        per_word_topics=True
    )
    print("✅ LDA Model trained with 6 topics.")

    # ✅ Save Topic Keywords
    topics = lda_model.show_topics(num_topics=-1, num_words=10, formatted=False)
    with open(r"C:\Users\aishw\OneDrive\Desktop\PBL Autonomous buses\topic_keywords_6.csv", "w", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Topic", "Top Keywords"])
        for topic_num, words in topics:
            keyword_list = [word for word, _ in words]
            keywords = ", ".join(keyword_list)
            writer.writerow([topic_num, keywords])
    print("✅ Topic keywords saved.")

    # ✅ Coherence Scores
    coherence_model_cv = CoherenceModel(model=lda_model, texts=texts, dictionary=dictionary, coherence='c_v')
    coherence_cv = coherence_model_cv.get_coherence()

    coherence_model_umass = CoherenceModel(model=lda_model, texts=texts, dictionary=dictionary, coherence='u_mass')
    coherence_umass = coherence_model_umass.get_coherence()

    print(f"📊 Coherence Score (c_v): {coherence_cv:.4f}")
    print(f"📊 Coherence Score (u_mass): {coherence_umass:.4f}")

    with open(r"C:\Users\aishw\OneDrive\Desktop\PBL Autonomous buses\coherence_score_6topics.txt", "w") as f:
        f.write(f"Coherence Score (c_v): {coherence_cv:.4f}\n")
        f.write(f"Coherence Score (u_mass): {coherence_umass:.4f}")

    # ✅ Save Model & Artifacts
    lda_model.save(r"C:\Users\aishw\OneDrive\Desktop\PBL Autonomous buses\lda_model_6topics.gensim")
    dictionary.save(r"C:\Users\aishw\OneDrive\Desktop\PBL Autonomous buses\lda_dictionary_6topics.gensim")
    corpora.MmCorpus.serialize(r"C:\Users\aishw\OneDrive\Desktop\PBL Autonomous buses\lda_corpus_6topics.mm", corpus)

    # ✅ Assign Dominant Topics
    document_topic = []
    for doc in corpus:
        topic_distribution = lda_model.get_document_topics(doc)
        most_relevant_topic = max(topic_distribution, key=lambda x: x[1]) if topic_distribution else None
        document_topic.append(most_relevant_topic)

    for i, topic in enumerate(document_topic[:10]):
        if topic:
            print(f"Document {i}: Topic {topic[0]} with probability {topic[1]:.4f}")
        else:
            print(f"Document {i}: No topic assigned")

    # ✅ Visualization
    try:
        vis = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary)
        pyLDAvis.save_html(vis, r"C:\Users\aishw\OneDrive\Desktop\PBL Autonomous buses\lda_visualization_6topics.html")
        print("✅ pyLDAvis visualization saved.")
    except Exception as e:
        print(f"⚠️ Visualization failed: {e}")
