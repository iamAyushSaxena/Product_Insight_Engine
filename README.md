# Product Manager's Insight Engine: Sentiment Analysis of Customer Reviews

## 🚀 Problem Statement

As a product grows, so does the volume of user feedback. For a Product Manager, manually sifting through thousands of app store reviews to find actionable insights is inefficient and prone to bias. This project, the **Product Manager's Insight Engine**, is a Python tool that automates the analysis of customer reviews to provide a clear, data-driven overview of user sentiment and key pain points.

The goal is to transform raw, unstructured feedback into a strategic asset that can inform the product roadmap, prioritize bug fixes, and measure user satisfaction over time.

## ✨ Key Features & Analysis

* **Automated Sentiment Scoring:** Uses NLTK's VADER (Valence Aware Dictionary and sEntiment Reasoner) to assign a sentiment score (Positive, Negative, Neutral) to each review.
* **Quantitative Summary:** Provides a clear terminal output with the total count and percentage of reviews in each sentiment category.
* **Sentiment Distribution Visualization:** Generates and saves a bar chart (`sentiment_distribution.png`) for an at-a-glance understanding of overall user happiness.
* **Pain Point Identification:** Creates a word cloud (`negative_keywords_wordcloud.png`) from negative reviews to visually highlight recurring complaints and keywords like "crash," "slow," "login," and "bug."

## 🛠️ Tech Stack

* **Language:** Python 3.x
* **Libraries:**
    * Pandas: For data manipulation and analysis.
    * NLTK (VADER): For sentiment analysis.
    * Matplotlib: For generating static plots.
    * WordCloud: For creating word cloud visualizations.

## ⚙️ Setup & Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/product_insight_engine.git](https://github.com/your-username/product_insight_engine.git)
    cd product_insight_engine
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required libraries:**
    ```bash
    pip install -r requirements.txt
    ```
    The first time you run the script, it will automatically download necessary NLTK data.

## 🏃‍♀️ How to Run the Project

Ensure your review data is in `data/app_reviews.csv`. Then, simply run the main script from the root directory:

```bash
python main.py