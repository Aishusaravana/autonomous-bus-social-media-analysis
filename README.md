# 🚌 Autonomous Bus — Public Perception Analysis
### NLP & Sentiment Analysis of YouTube Data | Personal Portfolio Project

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)
![NLP](https://img.shields.io/badge/NLP-LDA%20%7C%20BERT%20%7C%20VADER-green)
![Sentiment](https://img.shields.io/badge/Sentiment-Google%20Cloud%20NLP-yellow?logo=google-cloud)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)
![Type](https://img.shields.io/badge/Type-Research%20%7C%20Portfolio-blueviolet)

---

## 📌 Project Overview

This project investigates **how the public perceives autonomous buses** by analysing YouTube comments at scale using Natural Language Processing (NLP). Rather than relying on surveys or controlled studies, it taps into organic, unsolicited public discourse — capturing genuine sentiment, recurring themes, and evolving opinions over time.

> **Disclaimer:** This is a self-initiated research and portfolio project. All YouTube data was collected via the YouTube Data API v3 in accordance with Google's Terms of Service. No personal data was collected or stored.

---

## 🎯 Research Questions

1. What **topics and themes** dominate public discussion around autonomous buses?
2. Is public sentiment toward autonomous buses generally **positive, negative, or neutral**?
3. How do **unsupervised methods** (pure LDA) compare to **theory-driven methods** (DOI-seeded classification) in capturing meaningful themes?
4. Do **Google Cloud NLP** and **VADER** agree on sentiment — and where do they diverge?

---

## 🧠 Methodology

The pipeline follows four main stages:

```
YouTube Data Collection
        ↓
Text Preprocessing & Cleaning
        ↓
Topic Modeling (LDA + DOI-Seeded)
        ↓
Sentiment Analysis (Google NLP + VADER)
        ↓
Evaluation & Visualisation
```

### 1. 🗂️ Data Collection & Preprocessing
- Collected YouTube comments using the **YouTube Data API v3**
- Cleaned text: removed stopwords, punctuation, URLs, emojis, and duplicates
- Applied tokenisation and lemmatisation for modelling readiness

### 2. 🧩 Topic Modeling — Two Approaches

| Approach | Method | Description |
|---|---|---|
| **Unsupervised** | LDA (Latent Dirichlet Allocation) | Discovers topics purely from data patterns |
| **Theory-Driven** | DOI-Based Semi-Supervised | Seeds topics using Diffusion of Innovation theory categories |

- Used **BERT + Cosine Similarity** to evaluate how well LDA topics align with DOI theory categories
- Identified **dominant topics** per comment and tracked topic distributions

### 3. 💬 Sentiment Analysis — Dual Approach

| Tool | Type | Strength |
|---|---|---|
| **Google Cloud Natural Language API** | ML-based | Handles nuance, sarcasm, and complex sentences |
| **VADER** | Rule-based | Fast, lightweight, strong on social media text |

- Compared both tools using **Pearson Correlation Coefficient**
- Evaluated **sentiment consistency** across topic clusters

### 4. 📊 Evaluation Metrics
- **TF-IDF + Cosine Similarity** — measures topic coherence
- **BERT + Cosine Similarity** — semantic alignment of topics to DOI categories
- **Pearson Coefficient** — agreement between Google NLP and VADER scores
- **Histogram & Distribution Plots** — sentiment and topic visualisations

---

## 📂 Repository Structure

```
autonomous-bus-social-media-analysis/
│
├── src/
│   ├── preprocessing/
│   │   ├── youtube_cleaning.py          # Data collection & text cleaning
│   │   └── stopwords.py                 # Custom stopword list
│   │
│   ├── topic_modeling/
│   │   ├── final_topic_modeling.py      # LDA unsupervised topic modeling
│   │   ├── dominant_topic.py            # Assigns dominant topic per comment
│   │   ├── reconstructed_dominant.py    # Reconstructed dominant topic file
│   │   └── filtered_merge_dominant.py   # Merges & filters dominant topic output
│   │
│   ├── sentiment/
│   │   ├── final_sentiment_analysis.py          # Google NLP + VADER sentiment
│   │   ├── merged_DOI_google_nlp_vader.py       # Merges DOI + sentiment results
│   │   └── histogram_sentiment.py               # Sentiment distribution plots
│   │
│   ├── evaluation/
│   │   ├── bert_cosine_topic.py                 # BERT + Cosine for topic similarity
│   │   ├── bert_cosine_DOI.py                   # BERT + Cosine for DOI alignment
│   │   ├── tfidf_cosine_similarity.py           # TF-IDF evaluation metric
│   │   ├── sentiment_consistency.py             # Sentiment consistency check
│   │   ├── pearson_google_vs_vader.py           # Pearson correlation comparison
│   │   └── seeded_list_evaluate.py              # DOI seed list evaluation
│   │
│   └── visualisation/
│       └── unsupervised_visualization.py        # Final unsupervised visual output
│
├── figures/                   # All charts, plots, and visual outputs
└── README.md
```

## 🔍 Key Findings

- 📌 Public discourse clusters around **safety concerns, technology curiosity, and policy skepticism**
- 📌 **DOI-seeded classification** produced more interpretable and theory-aligned topics than pure LDA
- 📌 Google Cloud NLP and VADER showed **moderate-to-strong agreement** on overall sentiment polarity
- 📌 Sentiment varies significantly by **topic cluster** — safety topics skew negative; innovation topics skew positive
- 📌 BERT-based similarity confirmed that LDA topics **partially align** with DOI innovation categories

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.10 |
| NLP & ML | BERT (HuggingFace), LDA (Gensim), NLTK, VADER |
| Sentiment | Google Cloud Natural Language API |
| Evaluation | TF-IDF, Cosine Similarity, Pearson Correlation |
| Visualisation | Matplotlib, Seaborn |
| Data Collection | YouTube Data API v3 |
| Theory Framework | Diffusion of Innovation (DOI) — Rogers, 2003 |

---

## ▶️ How to Run

```bash
# 1. Clone the repository
git clone https://github.com/Aishusaravana/autonomous-bus-social-media-analysis.git
cd autonomous-bus-social-media-analysis

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run preprocessing
python src/preprocessing/youtube_cleaning.py

# 4. Run topic modeling
python src/topic_modeling/final_topic_modeling.py

# 5. Run sentiment analysis
python src/sentiment/final_sentiment_analysis.py

# 6. Run evaluation
python src/evaluation/bert_cosine_topic.py
```

> ⚠️ You will need a **Google Cloud API key** with Natural Language API enabled and a **YouTube Data API v3 key** to replicate data collection.

---

## 👩‍💻 About the Author

Aishwarya Saravanan — Business Analyst | NLP Researcher | Aspiring Data Analyst

🎓 Based in **Ottawa, Ontario, Canada**
📜 Passionate about using data to understand human behaviour at scale
🔗 [GitHub](https://github.com/Aishusaravana)

---

