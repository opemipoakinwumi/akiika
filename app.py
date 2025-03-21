import streamlit as st
import feedparser
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from bs4 import BeautifulSoup
from urllib.parse import quote  # For URL encoding
import os

# Set page config FIRST
st.set_page_config(page_title="AI-Powered News Sentiment Analyzer", page_icon="📰")

# Inject custom CSS for modern styling
st.markdown(
    """
    <style>
    /* Main page styling */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa, #c3cfe2);
        color: #2c3e50;
    }
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #2c3e50 !important;
        color: #ffffff !important;
    }
    /* Button styling */
    .stButton button {
        background-color: #3498db;
        color: white;
        border-radius: 12px;
        padding: 10px 24px;
        font-size: 16px;
        border: none;
        transition: background-color 0.3s ease;
    }
    .stButton button:hover {
        background-color: #2980b9;
    }
    /* Input box styling */
    .stTextInput input {
        background-color: #ecf0f1;
        border-radius: 8px;
        border: 1px solid #bdc3c7;
        padding: 10px;
    }
    /* Header styling */
    h1 {
        color: #3498db;
        font-size: 36px;
        font-weight: bold;
    }
    h2 {
        color: #2c3e50;
        font-size: 28px;
        font-weight: bold;
    }
    h3 {
        color: #2c3e50;
        font-size: 22px;
        font-weight: bold;
    }
    /* Word cloud container styling */
    .stPlot {
        border-radius: 12px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Function to remove HTML tags
def remove_html_tags(text):
    """Remove HTML tags from a string using BeautifulSoup."""
    if text:
        return BeautifulSoup(text, "html.parser").get_text()
    return ""

# Function to fetch news articles
def fetch_news(topic):
    # URL-encode the topic to handle spaces and special characters
    encoded_topic = quote(topic)
    url = f"https://news.google.com/rss/search?q={encoded_topic}"
    try:
        feed = feedparser.parse(url)
        if not feed.entries:
            st.warning("No articles found. Please try a different topic.")
            return []
        articles = []
        for entry in feed.entries:
            title = entry.title if 'title' in entry else ''
            summary = remove_html_tags(entry.summary) if 'summary' in entry else ''
            if title:  # Skip articles with empty titles
                articles.append({
                    'title': title,
                    'summary': summary
                })
        return articles
    except Exception as e:
        st.error(f"Error fetching news: {e}")
        return []

# Function to analyze sentiment
def analyze_sentiment(text):
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0:
        return 'Positive'
    elif analysis.sentiment.polarity < 0:
        return 'Negative'
    else:
        return 'Neutral'

# Function to generate a summary
def generate_summary(articles):
    summary = ""
    sentiment_count = {'Positive': 0, 'Negative': 0, 'Neutral': 0}
    for article in articles:
        sentiment = analyze_sentiment(article['title'] + " " + article['summary'])
        sentiment_count[sentiment] += 1
        summary += f"**Headline:** {article['title']}\n\n**Sentiment:** {sentiment}\n\n**Summary:** {article['summary']}\n\n---\n\n"
    total = len(articles)
    if total > 0:
        sentiment_distribution = {k: (v / total) * 100 for k, v in sentiment_count.items()}
        summary += f"**Sentiment Distribution:**\n\n- Positive: {sentiment_distribution['Positive']:.2f}%\n- Negative: {sentiment_distribution['Negative']:.2f}%\n- Neutral: {sentiment_distribution['Neutral']:.2f}%\n"
    else:
        summary += "No valid articles found for sentiment analysis.\n"
    return summary

# Function to generate a word cloud
def generate_wordcloud(text):
    if text.strip():
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(plt)
    else:
        st.warning("No text available to generate word cloud.")

# Function to create an interactive word cloud
def interactive_wordcloud(text):
    if text.strip():
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        word_list = wordcloud.words_
        fig = go.Figure(data=[go.Scatter(
            x=[1] * len(word_list),
            y=list(word_list.keys()),
            text=list(word_list.keys()),
            mode='text',
            textfont=dict(size=list(word_list.values()))
        )])
        fig.update_layout(title="Interactive Word Cloud")
        st.plotly_chart(fig)
    else:
        st.warning("No text available to generate interactive word cloud.")

# Main function for Streamlit app
def main():
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "About Us"])

    if page == "Home":
        st.title("AI-Powered News Sentiment Analyzer")
        st.write("Enter a topic to analyze the sentiment of recent news articles.")

        # User input for topic
        topic = st.text_input("Enter the topic you want to search for:", "technology")

        if st.button("Analyze"):
            articles = fetch_news(topic)
            if articles:
                summary = generate_summary(articles)
                st.markdown(summary)

                all_text = " ".join([article['title'] + " " + article['summary'] for article in articles])
                st.subheader("Word Cloud")
                generate_wordcloud(all_text)

                st.subheader("Interactive Word Cloud")
                interactive_wordcloud(all_text)

    elif page == "About Us":
        st.title("About Us")
        st.write("Welcome to the **AI-Powered News Sentiment Analyzer**! This tool is designed to help you understand the sentiment behind recent news articles on any topic of your choice.")

        st.write("### Our Mission")
        st.write("Our mission is to provide users with a quick and easy way to analyze the sentiment of news articles, helping them stay informed about public opinion and media trends.")

        st.write("### How It Works")
        st.write("1. **Fetch News Articles**: The app fetches recent news articles from Google News RSS feeds based on your chosen topic.")
        st.write("2. **Analyze Sentiment**: Using Natural Language Processing (NLP), the app determines whether each article has a positive, negative, or neutral sentiment.")
        st.write("3. **Generate Insights**: The app provides a summary of the articles, including sentiment distribution and a word cloud for visual analysis.")

        st.write("### Meet the Team")
        st.write("This project was created by a dedicated team of developers and data enthusiasts:")
        st.write("- **Opemipo Akinwumi**: Founder and Lead Developer")
       

        st.write("### Contact Us")
        st.write("Have questions or feedback? We'd love to hear from you!")
        st.write("📧 Email: support@newsanalyzer.com")
        st.write("🌐 Website: [www.newsanalyzer.com](https://www.newsanalyzer.com)")

        st.write("### Disclaimer")
        st.write("This tool is for educational and informational purposes only. The sentiment analysis is based on automated algorithms and may not always reflect human judgment.")

        st.write("### Shoutout")
        st.write("Shoutout to [Ayodeji](https://ayodejiades.vercel.app/) for helping me out with errors.")
# Run the Streamlit app
if __name__ == "__main__":
    main()
