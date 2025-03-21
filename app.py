import streamlit as st
import feedparser
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from bs4 import BeautifulSoup
from urllib.parse import quote
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import time
import plotly.express as px
import io

# Set page config FIRST
st.set_page_config(
    page_title="AI-Powered News Sentiment Analyzer",
    page_icon="📰",
    layout="wide"
)

# Initialize session state variables if they don't exist
if 'articles' not in st.session_state:
    st.session_state.articles = []
if 'last_topic' not in st.session_state:
    st.session_state.last_topic = ""
if 'sentiment_df' not in st.session_state:
    st.session_state.sentiment_df = None
if 'progress' not in st.session_state:
    st.session_state.progress = 0

# Cache the function to improve performance
@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_news(topic, max_articles=20):
    """Fetch news articles based on the provided topic."""
    encoded_topic = quote(topic)
    url = f"https://news.google.com/rss/search?q={encoded_topic}"
    
    try:
        feed = feedparser.parse(url)
        if not feed.entries:
            return []
            
        articles = []
        for entry in feed.entries[:max_articles]:  # Limit the number of articles
            title = entry.title if 'title' in entry else ''
            summary = remove_html_tags(entry.summary) if 'summary' in entry else ''
            link = entry.link if 'link' in entry else ''
            published = entry.published if 'published' in entry else ''
            
            if title:  # Skip articles with empty titles
                articles.append({
                    'title': title,
                    'summary': summary,
                    'link': link,
                    'published': published
                })
        return articles
    except Exception as e:
        st.error(f"Error fetching news: {e}")
        return []

def remove_html_tags(text):
    """Remove HTML tags from a string using BeautifulSoup."""
    if text:
        return BeautifulSoup(text, "html.parser").get_text()
    return ""

def analyze_sentiment(text):
    """Analyze the sentiment of the provided text."""
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    subjectivity = analysis.sentiment.subjectivity
    
    if polarity > 0.1:
        sentiment = 'Positive'
    elif polarity < -0.1:
        sentiment = 'Negative'
    else:
        sentiment = 'Neutral'
        
    return {
        'sentiment': sentiment,
        'polarity': polarity,
        'subjectivity': subjectivity
    }

def process_article(article, idx, total):
    """Process a single article for sentiment analysis."""
    full_text = article['title'] + " " + article['summary']
    sentiment_data = analyze_sentiment(full_text)
    article.update(sentiment_data)
    
    # Update progress
    st.session_state.progress = (idx + 1) / total
    
    return article

def create_sentiment_dataframe(articles):
    """Create a pandas DataFrame from the analyzed articles."""
    df = pd.DataFrame(articles)
    
    # Extract date from published field if available
    if 'published' in df.columns:
        df['date'] = pd.to_datetime(df['published'], errors='coerce').dt.date
    
    return df

def generate_word_cloud(text):
    """Generate a word cloud from the provided text."""
    if not text.strip():
        return None
        
    wordcloud = WordCloud(
        width=800, 
        height=400, 
        background_color='white',
        max_words=100
    ).generate(text)
    
    return wordcloud

def plot_sentiment_distribution(df):
    """Plot the sentiment distribution."""
    sentiment_counts = df['sentiment'].value_counts().reset_index()
    sentiment_counts.columns = ['Sentiment', 'Count']
    
    fig = px.pie(
        sentiment_counts, 
        values='Count', 
        names='Sentiment', 
        title='Sentiment Distribution'
    )
    return fig

def plot_sentiment_over_time(df):
    """Plot sentiment over time if date information is available."""
    if 'date' not in df.columns or df['date'].isna().all():
        return None
        
    # Group by date and calculate sentiment counts
    sentiment_by_date = pd.crosstab(df['date'], df['sentiment'])
    sentiment_by_date = sentiment_by_date.reset_index()
    
    # Convert to long format for Plotly
    sentiment_by_date_long = pd.melt(
        sentiment_by_date, 
        id_vars=['date'], 
        value_vars=['Positive', 'Neutral', 'Negative'],
        var_name='Sentiment',
        value_name='Count'
    )
    
    fig = px.line(
        sentiment_by_date_long, 
        x='date', 
        y='Count', 
        color='Sentiment',
        title='Sentiment Trends Over Time'
    )
    return fig

def plot_polarity_subjectivity(df):
    """Create a scatter plot of polarity vs subjectivity."""
    fig = px.scatter(
        df, 
        x='polarity', 
        y='subjectivity', 
        color='sentiment',
        hover_data=['title'],
        title='Polarity vs Subjectivity'
    )
    
    fig.update_layout(
        xaxis_title="Polarity (Negative ⟷ Positive)",
        yaxis_title="Subjectivity (Factual ⟷ Opinion)"
    )
    
    return fig

def display_article_list(df):
    """Display a list of articles with their sentiment."""
    st.subheader("News Articles")
    
    # Create sentiment badges
    sentiment_badges = {
        'Positive': '🟢 Positive',
        'Neutral': '🔵 Neutral',
        'Negative': '🔴 Negative'
    }
    
    # Display each article with an expander
    for i, row in df.iterrows():
        with st.expander(f"{row['title']} [{sentiment_badges[row['sentiment']]}]"):
            st.markdown(f"**Summary:** {row['summary']}")
            st.markdown(f"**Polarity:** {row['polarity']:.2f} | **Subjectivity:** {row['subjectivity']:.2f}")
            if row['link']:
                st.markdown(f"[Read full article]({row['link']})")

def main():
    """Main function for the Streamlit app."""
    # Page header
    st.title("📰 AI-Powered News Sentiment Analyzer")
    
    # Sidebar configuration
    with st.sidebar:
        st.title("Settings")
        
        topic = st.text_input("Search Topic:", value="technology")
        max_articles = st.slider("Max Articles:", min_value=5, max_value=1000, value=20)
        
        # Advanced options in an expander
        with st.expander("Advanced Options"):
            sentiment_threshold = st.slider(
                "Sentiment Threshold:", 
                min_value=0.0, 
                max_value=0.5, 
                value=0.1,
                help="Threshold for classifying sentiment as positive/negative"
            )
        
        # About section
        st.markdown("---")
        st.subheader("About")
        st.info("""
        This tool analyzes the sentiment of recent news articles on your chosen topic.
        
        **Features:**
        - Real-time news fetching
        - Sentiment analysis
        - Interactive visualizations
        - Trend analysis
        
        Created by Opemipo Akinwumi with help from [Ayodeji](https://ayodejiades.vercel.app/).
        """)
        
        st.markdown("---")
        st.caption("© 2025 News Analyzer | [About Us](https://www.newsanalyzer.com)")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("Enter a topic to analyze the sentiment of recent news articles.")
        
        analyze_button = st.button("Analyze News", use_container_width=True)
        
        # If topic has changed or analyze button is clicked
        if analyze_button or (topic and topic != st.session_state.last_topic):
            with st.spinner("Fetching news articles..."):
                articles = fetch_news(topic, max_articles)
                
            if not articles:
                st.warning("No articles found. Please try a different topic.")
            else:
                # Process articles with progress bar
                st.session_state.progress = 0
                progress_bar = st.progress(0)
                
                # Process articles with parallelization
                total_articles = len(articles)
                with ThreadPoolExecutor(max_workers=4) as executor:
                    processed_articles = list(executor.map(
                        lambda x: process_article(x[1], x[0], total_articles),
                        enumerate(articles)
                    ))
                
                # Update progress bar (complete)
                progress_bar.progress(1.0)
                time.sleep(0.5)  # Small delay for visual feedback
                progress_bar.empty()  # Remove progress bar
                
                # Store results in session state
                st.session_state.articles = processed_articles
                st.session_state.last_topic = topic
                st.session_state.sentiment_df = create_sentiment_dataframe(processed_articles)
                
                st.success(f"Analyzed {len(processed_articles)} news articles!")
    
    with col2:
        if st.session_state.sentiment_df is not None and not st.session_state.sentiment_df.empty:
            # Summary statistics
            df = st.session_state.sentiment_df
            total = len(df)
            positive = sum(df['sentiment'] == 'Positive')
            negative = sum(df['sentiment'] == 'Negative')
            neutral = sum(df['sentiment'] == 'Neutral')
            
            # Create metrics
            st.subheader("Summary")
            metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
            metrics_col1.metric("Positive", f"{positive} ({positive/total*100:.1f}%)")
            metrics_col2.metric("Neutral", f"{neutral} ({neutral/total*100:.1f}%)")
            metrics_col3.metric("Negative", f"{negative} ({negative/total*100:.1f}%)")
    
    # Display visualizations if we have data
    if st.session_state.sentiment_df is not None and not st.session_state.sentiment_df.empty:
        df = st.session_state.sentiment_df
        
        # Tabs for different visualizations
        tab1, tab2, tab3, tab4 = st.tabs(["Articles", "Sentiment Analysis", "Word Cloud", "Trends"])
        
        with tab1:
            display_article_list(df)
        
        with tab2:
            # Sentiment distribution and polarity/subjectivity plot
            pie_col, scatter_col = st.columns(2)
            
            with pie_col:
                fig_pie = plot_sentiment_distribution(df)
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with scatter_col:
                fig_scatter = plot_polarity_subjectivity(df)
                st.plotly_chart(fig_scatter, use_container_width=True)
        
        with tab3:
            # Word cloud visualization with download option
            all_text = " ".join([article['title'] + " " + article['summary'] for article in st.session_state.articles])
            wordcloud = generate_word_cloud(all_text)
            
            if wordcloud:
                st.subheader("Word Cloud")
                
                # Create a figure for the word cloud
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                
                # Display the word cloud
                st.pyplot(fig)
                
                # Create a download button for the word cloud image
                buf = io.BytesIO()
                fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
                buf.seek(0)
                
                st.download_button(
                    label="Download Word Cloud",
                    data=buf,
                    file_name=f"wordcloud_{topic}_{time.strftime('%Y%m%d')}.png",
                    mime="image/png"
                )
            else:
                st.warning("Not enough text to generate word cloud.")
        
        with tab4:
            # Sentiment trends over time
            fig_trend = plot_sentiment_over_time(df)
            if fig_trend:
                st.plotly_chart(fig_trend, use_container_width=True)
            else:
                st.info("Time-based analysis not available for these articles.")
            
            # Allow downloading the DataFrame
            st.subheader("Export Data")
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download Data as CSV",
                csv,
                f"news_sentiment_{topic}_{time.strftime('%Y%m%d')}.csv",
                "text/csv",
                key='download-csv'
            )

# Run the Streamlit app
if __name__ == "__main__":
    main()
