import pandas as pd

df = pd.read_csv('tweet_sentiment_data.csv')

# Split the cleaned text into individual words
df['cleaned_text'] = df['cleaned_text'].astype(str)
df_exploded = df.assign(word=df['cleaned_text'].str.split()).explode('word')

# Remove any empty strings or nulls
df_exploded = df_exploded[df_exploded['word'].str.len() > 1]

df_exploded.to_csv('tweet_wordcloud_data.csv', index=False)