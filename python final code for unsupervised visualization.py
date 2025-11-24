import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Load Data
@st.cache_data

def load_data():
    return pd.read_excel("final_lda_sentiment_data_cleaned.xlsx")

df = load_data()

st.title("🚍 Autonomous Bus Public Opinion Explorer")

# --- Sidebar Filters ---
st.sidebar.header("🔍 Filters")
topic_options = df['topic_label'].dropna().unique().tolist()
selected_topics = st.sidebar.multiselect("Select Topics:", topic_options, default=topic_options)

min_score, max_score = st.sidebar.slider("Sentiment Score Range:", -1.0, 1.0, (-1.0, 1.0), 0.01)

# --- Filter Data ---
filtered_df = df[(df['topic_label'].isin(selected_topics)) &
                 (df['sentiment_score'] >= min_score) &
                 (df['sentiment_score'] <= max_score)]

st.markdown(f"### Showing {len(filtered_df)} comments")

# --- Average Sentiment ---
st.subheader("📊 Average Sentiment Score per Topic")
avg_sentiment = filtered_df.groupby("topic_label")["sentiment_score"].mean().sort_values()
fig, ax = plt.subplots(figsize=(8, 4))
avg_sentiment.plot(kind="barh", ax=ax, color="skyblue")
plt.xlabel("Average Sentiment Score")
st.pyplot(fig)

# --- Sentiment Score Distribution ---
st.subheader("📈 Sentiment Score Distribution")
fig2, ax2 = plt.subplots(figsize=(10, 5))
sns.boxplot(data=filtered_df, x="topic_label", y="sentiment_score", palette="pastel", ax=ax2)
plt.xticks(rotation=30, ha="right")
st.pyplot(fig2)

# --- Word Clouds ---
st.subheader("☁️ Word Cloud per Topic")
for topic in selected_topics:
    topic_text = " ".join(filtered_df[filtered_df['topic_label'] == topic]['reconstructed_text'].dropna())
    if topic_text.strip():
        wc = WordCloud(width=800, height=400, background_color="white").generate(topic_text)
        st.markdown(f"**{topic}**")
        st.image(wc.to_array(), use_column_width=True)
    else:
        st.warning(f"No text available for topic: {topic}")
