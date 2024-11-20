import streamlit as st
import spacy
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from textblob import TextBlob

# Attempt to load the spaCy model, or download it if not found
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Function to extract primary topics from the story
def identify_topics(story_text, num_topics=3):
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform([story_text])

    lda_model = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda_model.fit(matrix)

    topic_list = []
    terms = vectorizer.get_feature_names_out()
    for idx, topic in enumerate(lda_model.components_):
        important_words = [terms[i] for i in topic.argsort()[-5:]]
        topic_list.append(f"Topic {idx + 1}: {' '.join(important_words)}")

    # Ensure topics remain distinct
    return list(dict.fromkeys(topic_list))[:num_topics]

# Function to derive thematic insights from the story
def derive_insights(story_text):
    doc = nlp(story_text)
    insights = []

    # Themes or concepts to look for
    themes = [
        "mystery", "adventure", "love", "discovery", "growth", "conflict",
        "journey", "character", "emotions", "challenges", "success", "failure"
    ]

    for sentence in doc.sents:
        if any(theme in sentence.text.lower() for theme in themes) and sentence.text not in insights:
            insights.append(sentence.text)

    return insights

# Function to analyze the sentiment of the story
def analyze_sentiment(story_text):
    analysis = TextBlob(story_text)
    return analysis.sentiment.polarity, analysis.sentiment.subjectivity


# Function to create and display a word cloud
def display_wordcloud(story_text):
    # Generate the word cloud
    wc = WordCloud(width=800, height=400, background_color="white").generate(story_text)

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")  # Hide axes
    st.pyplot(fig)  # Pass the figure object to st.pyplot


# Streamlit app setup
st.set_page_config(page_title="Story Analyzer", page_icon="📚", layout="wide")

# Title and description
st.title("📚 Story Analyzer")
st.markdown("""
Uncover hidden themes, analyze sentiments, and visualize key elements of your story. 
Just paste your story below and let the tool provide a detailed breakdown!
""")

# Sidebar with usage instructions
with st.sidebar:
    st.header("How to Use")
    st.write("""
    1. Paste your story into the text box.
    2. Adjust the slider to choose how many topics to uncover.
    3. Click **Analyze Story** to get results.
    """)

# Story input area
story_input = st.text_area(
    "Paste or type your story here:",
    placeholder="Enter your story to analyze its themes, sentiments, and word importance.",
    height=250
)

# Slider to select the number of topics
num_topics = st.slider("Number of topics to identify", 1, 5, 3)

# Button to perform analysis
if st.button("Analyze Story"):
    if story_input.strip():
        with st.spinner("Analyzing your story, please wait..."):
            # Identify topics
            topics = identify_topics(story_input, num_topics)
            st.subheader("Identified Topics:")
            for topic in topics:
                st.write(f"- {topic}")

            # Derive insights
            insights = derive_insights(story_input)
            st.subheader("Thematic Insights:")
            if insights:
                for insight in insights:
                    st.write(f"• {insight}")
            else:
                st.write("No specific insights detected from the story.")

            # Perform sentiment analysis
            polarity, subjectivity = analyze_sentiment(story_input)
            st.subheader("Sentiment Analysis:")
            st.write(f"**Polarity:** {polarity:.2f} (positive > 0, negative < 0)")
            st.write(f"**Subjectivity:** {subjectivity:.2f} (objective = 0, subjective = 1)")

            # Display word cloud
            st.subheader("Word Cloud Visualization:")
            display_wordcloud(story_input)
    else:
        st.warning("Please provide a story to analyze.")
