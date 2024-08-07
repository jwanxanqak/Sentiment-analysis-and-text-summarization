import streamlit as st
import gensim
from gensim import corpora
from nltk.corpus import stopwords
import nltk
nltk.download('punkt')
from gensim.models import CoherenceModel
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer

st.title("Sentiment analysis and text summarization")
st.sidebar.title("sentiment analysis and text summarization")
st.markdown("This application is a Streamlit dashboard used "
            "to analyze sentiments and text summarization 🦅 ")
st.sidebar.markdown("This application is a Streamlit dashboard used "
            "to analyze sentiments and text summarization 🦅 ")


########################################
#tweet summary
########################################

st.subheader("tweet summary")
tweet_to_summarize = st.text_area("Enter the tweet to summarize")
num_sentences = st.sidebar.slider("Number of sentences in the summary", 1, 10, 2)

if tweet_to_summarize:
    parser = PlaintextParser.from_string(tweet_to_summarize, Tokenizer("english"))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, num_sentences)
    summary_text = " ".join([str(sentence) for sentence in summary])
    
    st.write("### Summary:")
    st.write(summary_text)

########################################
#sentiment analysis with VADER
########################################

# Download the VADER lexicon
nltk.download('vader_lexicon')
sid = SentimentIntensityAnalyzer()

# Home page title
st.header("sentiment analysis with VADER (NTLK)")

# Sidebar Settings
st.sidebar.header("sentiment analysis with VADER (NTLK)")
tweet_to_analyze = st.text_area("Enter the tweet to analyze")

# Additional parameters
st.sidebar.subheader("Additional parameters")
show_scores = st.sidebar.checkbox("Show Detailed Ratings", value=True)
show_summary = st.sidebar.checkbox("Show Sentiment Summary", value=True)

# sentiment analysis
if tweet_to_analyze:
    sentiment_scores = sid.polarity_scores(tweet_to_analyze)
    
    # Show results on the main page
    st.header("Sentiment Analysis Results")
    if show_scores:
        st.subheader("Sentiment Scores:")
        st.write(sentiment_scores)
    
    if show_summary:
        if sentiment_scores['compound'] >= 0.05:
            st.write("General Feeling: Positive 😊")
        elif sentiment_scores['compound'] <= -0.05:
            st.write("General Feeling: Negative 😞")
        else:
            st.write("General Feeling: Neutral 😐")

# Show instructions on the main page
st.sidebar.subheader("Instructions")
st.sidebar.write("""
1. Enter the tweet in the text area.
2. Adjust the additional parameters according to your preferences.
3. The results will be displayed on the main page.
""")
