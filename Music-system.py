import random
import pandas as pd
from textblob import TextBlob

def classify_sentiment(user_text):
    """Classify sentiment based on polarity score."""
    polarity = TextBlob(user_text).sentiment.polarity
    if polarity > 0.3:
        return "Happy"
    elif polarity < -0.3:
        return "Sad"
    elif polarity > -0.3 and polarity < 0:
        return "Relaxed"
    else:
        return "Motivated"

def recommend_song(sentiment, df):
    """Recommend a song based on sentiment."""
    filtered_df = df[df['Sentiment_Label'] == sentiment]
    if not filtered_df.empty:
        return filtered_df.sample(1).to_dict(orient='records')[0]
    return None

# Load dataset
file_path = "music_sentiment_dataset.csv"
df = pd.read_csv(file_path)

def get_music_recommendation(user_text):
    sentiment = classify_sentiment(user_text)
    song = recommend_song(sentiment, df)
    return sentiment, song

# Example usage
user_input = "I feel sad to even go out."
sentiment, recommended_song = get_music_recommendation(user_input)
print(f"Detected Sentiment: {sentiment}")
print("Recommended Song:", recommended_song)
