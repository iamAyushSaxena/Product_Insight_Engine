import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import os

# --- Pre-analysis Setup: Download NLTK data ---
# This needs to be done once.
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
    nltk.data.find('tokenizers/punkt.zip')
    nltk.data.find('corpus/stopwords.zip')
except nltk.downloader.DownloadError:
    print("Downloading required NLTK data...")
    nltk.download('vader_lexicon')
    nltk.download('punkt')
    nltk.download('stopwords')
    print("Download complete.")

# --- Core Analysis Functions ---

def load_data(filepath):
    """Loads review data from a CSV file."""
    print(f"Loading data from {filepath}...")
    try:
        df = pd.read_csv(filepath)
        # Ensure the 'review_text' column exists and is not empty
        if 'review_text' not in df.columns or df['review_text'].isnull().all():
            raise ValueError("CSV must contain a non-empty 'review_text' column.")
        df.dropna(subset=['review_text'], inplace=True)
        return df
    except FileNotFoundError:
        print(f"Error: The file {filepath} was not found.")
        return None

def analyze_sentiment(df):
    """Analyzes the sentiment of each review and categorizes it."""
    print("Analyzing sentiment...")
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
    print("Sentiment analysis complete.")
    return df

def generate_summary(df):
    """Prints a summary of the sentiment analysis results."""
    print("\n--- Product Feedback Analysis Summary ---")
    sentiment_counts = df['sentiment_category'].value_counts()
    print("Overall Review Sentiment Distribution:")
    print(sentiment_counts)
    
    # Calculate percentages for better context
    total_reviews = len(df)
    for category, count in sentiment_counts.items():
        percentage = (count / total_reviews) * 100
        print(f"- {category}: {count} reviews ({percentage:.2f}%)")
    
    print("\n--- Actionable Insights for Product Team ---")
    if 'Negative' in sentiment_counts:
        negative_percentage = (sentiment_counts['Negative'] / total_reviews) * 100
        if negative_percentage > 20:
             print("🔴 High Alert: Over 20% of reviews are negative. Prioritize bug fixes and user issue investigation.")
        else:
             print("🟢 Healthy: Negative feedback is within an acceptable range. Continue monitoring.")
    
    if 'Positive' in sentiment_counts:
        positive_percentage = (sentiment_counts['Positive'] / total_reviews) * 100
        if positive_percentage > 70:
            print("✅ Strong Positive Feedback: Users are happy! Identify what they love and double down on it in marketing.")

def visualize_sentiment_distribution(df, output_path):
    """Creates and saves a bar chart of sentiment distribution."""
    print("Generating sentiment distribution chart...")
    sentiment_counts = df['sentiment_category'].value_counts()
    
    plt.figure(figsize=(8, 6))
    sentiment_counts.plot(kind='bar', color=['green', 'red', 'grey'])
    plt.title('Distribution of Customer Review Sentiment')
    plt.xlabel('Sentiment Category')
    plt.ylabel('Number of Reviews')
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save the plot
    plt.savefig(output_path)
    print(f"Chart saved to {output_path}")
    plt.close()

def generate_negative_keywords_wordcloud(df, output_path):
    """Generates a word cloud from negative reviews to identify pain points."""
    print("Generating word cloud for negative feedback...")
    negative_reviews = df[df['sentiment_category'] == 'Negative']['review_text']
    
    if negative_reviews.empty:
        print("No negative reviews to generate a word cloud from.")
        return
        
    # Combine all negative reviews into one text block
    text = " ".join(review for review in negative_reviews)
    
    # Clean text: remove common stop words
    stop_words = set(stopwords.words('english'))
    # Add common app-related but non-insightful words
    stop_words.update(['app', 'use', 'get', 'like', 'even', 'would'])
    
    wordcloud = WordCloud(
        stopwords=stop_words,
        background_color="white",
        width=800,
        height=400,
        colormap='Reds'
    ).generate(text)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title("Common Keywords in Negative Reviews")
    
    # Save the plot
    plt.savefig(output_path)
    print(f"Word cloud saved to {output_path}")
    plt.close()

# --- Main Execution ---

if __name__ == "__main__":
    # Define file paths
    DATA_FILE = os.path.join('data', 'app_reviews.csv')
    OUTPUT_DIR = 'output'
    
    # Create output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    DISTRIBUTION_CHART_PATH = os.path.join(OUTPUT_DIR, 'sentiment_distribution.png')
    WORDCLOUD_PATH = os.path.join(OUTPUT_DIR, 'negative_keywords_wordcloud.png')

    # 1. Load Data
    reviews_df = load_data(DATA_FILE)
    
    if reviews_df is not None:
        # 2. Analyze Sentiment
        reviews_df = analyze_sentiment(reviews_df)
        
        # 3. Generate and Print Summary
        generate_summary(reviews_df)
        
        # 4. Create and Save Visualizations
        visualize_sentiment_distribution(reviews_df, DISTRIBUTION_CHART_PATH)
        generate_negative_keywords_wordcloud(reviews_df, WORDCLOUD_PATH)
        
        print("\n✅ Analysis complete. Check the 'output' folder for your charts.")
