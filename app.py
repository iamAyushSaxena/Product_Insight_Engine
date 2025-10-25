import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import os
import streamlit as st
from datetime import datetime
from collections import Counter
import altair as alt 

# --- NLTK Data Download ---
@st.cache_resource
def download_nltk_resources():
    print("Checking/Downloading NLTK resources...")
    nltk.download('vader_lexicon')
    nltk.download('punkt')
    nltk.download('punkt_tab') 
    nltk.download('stopwords')
    print("NLTK resources are ready.")

# Run the download function
download_nltk_resources()


# --- Core Analysis Functions ---
# We use @st.cache_data to speed up the app by not reloading data
# every time the user interacts with it.

@st.cache_data
def load_data(filepath):
    """Loads review data from a CSV file."""
    try:
        df = pd.read_csv(filepath)
        if 'review_text' not in df.columns or df['review_text'].isnull().all():
            raise ValueError("CSV must contain a non-empty 'review_text' column.")
        df.dropna(subset=['review_text'], inplace=True)
        return df
    except FileNotFoundError:
        st.error(f"Error: The file {filepath} was not found.")
        return None

@st.cache_data
def analyze_sentiment(df):
    """Analyzes the sentiment of each review and categorizes it."""
    sia = SentimentIntensityAnalyzer()
    
    # Get sentiment scores
    df['sentiment_score'] = df['review_text'].apply(lambda x: sia.polarity_scores(x)['compound'])
    
    # Categorize sentiment based on score
    def categorize_sentiment(score):
        if score >= 0.05:
            return 'Positive'
        elif score <= -0.05:
            return 'Negative'
        else:
            return 'Neutral'
            
    df['sentiment_category'] = df['sentiment_score'].apply(categorize_sentiment)
    return df

# --- Visualization Functions ---

# --- REMOVED THE visualize_sentiment_distribution FUNCTION ---

def generate_negative_keywords_wordcloud(df):
    """Generates and returns a word cloud from negative reviews."""
    negative_reviews = df[df['sentiment_category'] == 'Negative']['review_text']
    
    if negative_reviews.empty:
        return None
        
    text = " ".join(review for review in negative_reviews)
    stop_words = set(stopwords.words('english'))
    stop_words.update(['app', 'use', 'get', 'like', 'even', 'would', 'fix', 'please', 'bug'])
    
    wordcloud = WordCloud(
        stopwords=stop_words,
        background_color="white",
        width=800,
        height=400,
        colormap='Reds'
    ).generate(text)
    
    # Create the figure
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    ax.set_title("Common Keywords in Negative Reviews")
    
    # Return the figure object
    return fig

@st.cache_data
def get_negative_keyword_frequency(df, num_keywords=10):
    """
    Extracts and ranks common keywords from negative reviews by frequency.
    Assigns frequency levels (High, Medium, Low).
    """
    negative_reviews = df[df['sentiment_category'] == 'Negative']['review_text']

    if negative_reviews.empty:
        return pd.DataFrame(columns=['Keyword', 'Frequency Count', 'Frequency Level'])

    text = " ".join(review for review in negative_reviews)
    
    stop_words = set(stopwords.words('english'))
    # Add common app-related but non-insightful words
    stop_words.update(['app', 'use', 'get', 'like', 'even', 'would', 'fix', 'please', 'bug']) 

    # Tokenize and remove stopwords
    words = word_tokenize(text.lower())
    filtered_words = [word for word in words if word.isalnum() and word not in stop_words]

    # Count word frequencies
    word_counts = Counter(filtered_words)

    # Convert to DataFrame
    keywords_df = pd.DataFrame(word_counts.most_common(num_keywords), columns=['Keyword', 'Frequency Count'])

    if keywords_df.empty:
        return pd.DataFrame(columns=['Keyword', 'Frequency Count', 'Frequency Level'])

    # Determine frequency levels
    total_unique_keywords = len(word_counts)
    if total_unique_keywords > 0:
        high_threshold = num_keywords // 3
        medium_threshold = (num_keywords // 3) * 2

        def assign_frequency_level(row_index):
            if row_index < high_threshold:
                return 'High'
            elif row_index < medium_threshold:
                return 'Medium'
            else:
                return 'Low'
        
        keywords_df['Frequency Level'] = keywords_df.index.to_series().apply(assign_frequency_level)
    else:
        keywords_df['Frequency Level'] = 'N/A' # No keywords to rank

    # Reset the index to start from 1 instead of 0
    keywords_df.index = pd.RangeIndex(start=1, stop=len(keywords_df) + 1, step=1)

    return keywords_df


# --- Main Application UI ---

# Set up the file paths
DATA_FILE = os.path.join('data', 'app_reviews.csv')

# --- 1. Load Data ---
reviews_df = load_data(DATA_FILE)

# --- 2. Build the UI ---
st.set_page_config(layout="wide")

# Center-align the title and subtitle using HTML
st.markdown("""
    <div style='text-align: center;'>
        <h1>Product Manager's Insight Engine 📊</h1>
        <p>This dashboard analyzes customer app reviews to extract sentiment and key topics.</p>
    </div>
    """, unsafe_allow_html=True)

if reviews_df is not None:
    # --- 3. Analyze Data ---
    reviews_df = analyze_sentiment(reviews_df)
    
    # --- 4. Display Summary Metrics ---
    st.header("Analysis Summary")
    
    sentiment_counts = reviews_df['sentiment_category'].value_counts()
    total_reviews = len(reviews_df)
    
    # Get counts for each category, defaulting to 0 if none exist
    positive_count = sentiment_counts.get('Positive', 0)
    negative_count = sentiment_counts.get('Negative', 0)
    neutral_count = sentiment_counts.get('Neutral', 0)
    
    # Calculate percentages
    positive_perc = (positive_count / total_reviews) * 100
    negative_perc = (negative_count / total_reviews) * 100 # <-- This was the typo fix
    neutral_perc = (neutral_count / total_reviews) * 100 

    # Display metrics in columns
    col1, col2, col3, col4 = st.columns(4) 
    
    col1.metric("Total Reviews", f"{total_reviews}")
    
    col2.metric("Positive Reviews", f"{positive_count} ({positive_perc:.1f}%)", 
               delta=f"{positive_perc:.1f}%", delta_color="normal") 
    
    col3.metric("Negative Reviews", f"{negative_count} ({negative_perc:.1f}%)", 
               delta=f"{negative_perc:.1f}%", delta_color="inverse")
    
    col4.metric("Neutral Reviews", f"{neutral_count} ({neutral_perc:.1f}%)", 
               delta=f"{neutral_perc:.1f}%", delta_color="off") 

    st.markdown("---") # Adds a horizontal line

    # --- 5. Display Visualizations ---
    st.header("Visual Insights")
    
    col_viz1, col_viz2 = st.columns(2)
    
    with col_viz1:
        st.subheader("Sentiment Distribution")
        
        # --- THIS IS THE NEW INTERACTIVE CHART ---
        # Get the data and reset index
        sentiment_data = reviews_df['sentiment_category'].value_counts().reset_index()
        sentiment_data.columns = ['Sentiment Category', 'Number of Reviews']
        
        # Define the colors and order
        domain_ = ['Positive', 'Negative', 'Neutral']
        range_ = ['green', 'red', 'grey']

        # Create the Altair chart
        chart = alt.Chart(sentiment_data).mark_bar().encode(
            # --- THIS IS THE FIX for horizontal labels ---
            x=alt.X('Sentiment Category', sort=None, axis=alt.Axis(labelAngle=0)), # Set label angle to 0
            y=alt.Y('Number of Reviews'), # Y-axis
            color=alt.Color('Sentiment Category', 
                            scale=alt.Scale(domain=domain_, range=range_)), # Color bars
            tooltip=['Sentiment Category', 'Number of Reviews'] # Hover tooltip
        ).interactive() # Make it interactive

        st.altair_chart(chart, use_container_width=True)
        # --- END OF NEW CHART CODE ---


    with col_viz2:
        st.subheader("Negative Feedback Keywords")
        
        # Display the Word Cloud first
        fig_wc = generate_negative_keywords_wordcloud(reviews_df)
        if fig_wc is not None:
            st.pyplot(fig_wc, use_container_width=True) 

        else:
            st.write("No negative reviews to generate a word cloud from.")

    st.markdown("---") # Optional separator

    # --- 5b. Display Keyword Frequency Table (Full Width) ---
    st.subheader("Top Negative Keyword Frequency") 
    keywords_table_df = get_negative_keyword_frequency(reviews_df, num_keywords=15) # Get top 15 keywords
    if not keywords_table_df.empty:
        st.dataframe(keywords_table_df, use_container_width=True)
    else:
        st.info("No common negative keywords found.")

    # --- THIS IS THE ADDED LINE ---
    st.markdown("---") 

    # --- 6. (Optional) Show Raw Data ---
    if st.checkbox("Show raw data"):
        st.subheader("All Reviews")
        st.dataframe(reviews_df)

else:
    st.warning("Could not load data. Please check the 'data/app_views.csv' file.")


# --- 7. Footer ---
# --- REVERTED TO THE SIMPLE, WORKING, CENTERED FOOTER ---
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #6b7280; padding: 20px;'>
        <p>📊<strong>Product Insight Engine</strong> - Sentiment Analysis of Customer Reviews</p>
        <p>Built with <strong>Streamlit</strong> | <strong>Python</strong> | <strong>Pandas</strong> | <strong>NLTK (VADER)</strong> | <strong>WordCloud</strong> ⏱️<strong>Last Updated:</strong> {}</p>
        <p>© 2025 <strong>Ayush Saxena</strong>. All rights reserved.</p>
    </div>
""".format(datetime.now().strftime("%d-%b-%Y At %I:%M %p")), unsafe_allow_html=True)