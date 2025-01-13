import pandas as pd
import re
import pycountry

df = pd.read_csv('tweet_sentiment_data.csv')

country_list = [country.name.lower() for country in pycountry.countries]
# Add common alternative country names
country_list += [
    'usa', 'uk', 'u.s.a', 'u.k', 'united states', 'united kingdom',
    'south korea', 'north korea', 'russia', 'venezuela'
]

# Clean location data removing emojis and special characters
def clean_location(text):
    if pd.isna(text):
        return ""
    # Remove emojis and non-ASCII characters
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    # Remove punctuation and extra spaces
    text = re.sub(r'[^\w\s,]', '', text).strip().lower()
    return text

# Return country
def extract_country(text):
    for word in text.split(','):
        for country in country_list:
            if country in word.strip():
                return country.title()  # Return in title case for Tableau
    return None  # Return None if no country is found

df['cleaned_location'] = df['user_location'].apply(clean_location)
df['country'] = df['cleaned_location'].apply(extract_country)

df.to_csv('tweet_sentiment_with_country.csv', index=False)