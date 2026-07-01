import pandas as pd
import re
import string
from textblob import TextBlob
import emoji

TEXT_COLUMNS = ['tweet_text', 'text', 'tweet', 'content', 'full_text']
FOLLOWER_COLUMNS = ['followers_count', 'followers', 'user_followers']
DESCRIPTION_COLUMNS = ['user_description', 'description', 'bio']

def first_matching_column(df, candidates):
    normalized = {str(column).strip().lower(): column for column in df.columns}
    for candidate in candidates:
        column = normalized.get(candidate.lower())
        if column is not None:
            return column
    return None

def load_tweet_export(path='crypto-query-tweets.csv'):
    df = pd.read_csv(path, encoding='utf-8')
    text_column = first_matching_column(df, TEXT_COLUMNS)
    if text_column is None:
        raise ValueError("CSV must include tweet_text, text, tweet, content, or full_text.")

    normalized = df.copy()
    normalized['tweet_text'] = normalized[text_column].fillna('').astype(str)

    follower_column = first_matching_column(normalized, FOLLOWER_COLUMNS)
    if follower_column is None:
        normalized['followers_count'] = 11
    elif follower_column != 'followers_count':
        normalized['followers_count'] = pd.to_numeric(normalized[follower_column], errors='coerce').fillna(0)

    description_column = first_matching_column(normalized, DESCRIPTION_COLUMNS)
    if description_column is None:
        normalized['user_description'] = ''
    elif description_column != 'user_description':
        normalized['user_description'] = normalized[description_column].fillna('').astype(str)

    return normalized

df = load_tweet_export()

# Filter tweets out from accounts with < 10 followers
trimmed_tweets = df[df['followers_count'] > 10].copy()

# Define stop words
stop_words = {
    'a', 'an', 'the', 'and', 'or', 'but', 'if', 'in', 'on', 'to', 'for', 'of', 'at', 'by',
    'with', 'without', 'is', 'was', 'are', 'were', 'be', 'been', 'being', 'this', 'that',
    'these', 'those', 'it', 'its', 'as', 'from', 'they', 'them', 'their', 'you', 'your',
    'i', 'me', 'my', 'we', 'us', 'our', 'he', 'him', 'his', 'she', 'her', 'what', 'which',
    'who', 'whom', 'do', 'does', 'did', 'have', 'has', 'had', 'can', 'could', 'will', 'would',
    'shall', 'should', 'may', 'might', 'must', 'not', 'so', 'no', 'yes', 'up', 'down', 'out',
    'about', 'how', 'when', 'where', 'why'
}

def clean_text(text):
    if pd.isna(text):
        return ""
    # Remove URLs
    text = re.sub(r'http\S+|www.\S+', '', text)
    # Remove mentions and hashtags
    text = re.sub(r'@\w+|#\w+', '', text)
    # Remove emojis
    text = emoji.replace_emoji(text, replace='')
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Convert to lowercase
    text = text.lower()
    # Remove stopwords
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

trimmed_tweets['cleaned_text'] = trimmed_tweets['tweet_text'].apply(clean_text)

# Drop original uncleaned text and bio to improve readability in Tableau
cleaned_tweets = trimmed_tweets.drop(columns=['tweet_text', 'user_description'])

# Sentiment analysis with TextBlob
def get_sentiment(text):
    return TextBlob(text).sentiment.polarity

cleaned_tweets['sentiment_polarity'] = cleaned_tweets['cleaned_text'].apply(get_sentiment)

sentiment = cleaned_tweets

sentiment.to_csv('tweet_sentiment_data.csv', index=False, encoding='utf-8', quoting=1)

print("Sentiment analysis complete. Data saved to 'tweet_sentiment_data.csv'.")
