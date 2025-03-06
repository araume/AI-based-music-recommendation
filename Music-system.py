import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv('music_sentiment_dataset.csv')

# Preprocessing
data['Sentiment_Label'] = data['Sentiment_Label'].map({'Happy': 0, 'Sad': 1, 'Relaxed': 2, 'Motivated': 3})

# Feature Extraction
tfidf = TfidfVectorizer(max_features=1000)
X = tfidf.fit_transform(data['User_Text']).toarray()
y = data['Sentiment_Label']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Model Evaluation
y_pred = model.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')

# Recommendation Function
def recommend_songs(user_text):
    sentiment = model.predict(tfidf.transform([user_text]))[0]
    recommended_songs = data[data['Sentiment_Label'] == sentiment]
    return recommended_songs[['Song_Name', 'Artist', 'Genre', 'Tempo (BPM)', 'Mood', 'Energy', 'Danceability']]

# Example Usage
user_input = input("Enter your current mood: ")
recommendations = recommend_songs(user_input)
print(recommendations)