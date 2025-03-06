import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv('music_sentiment_dataset.csv')

# Preprocessing
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['User_Text'])
y = df['Sentiment_Label']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Sentiment Classification Model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

# Recommendation Function
def recommend_songs(user_text):
    sentiment = model.predict(vectorizer.transform([user_text]))[0]
    recommended_songs = df[df['Sentiment_Label'] == sentiment]
    return recommended_songs[['Song_Name', 'Artist', 'Genre', 'Tempo (BPM)', 'Mood', 'Energy', 'Danceability']]

# Example Usage
user_input = "I feel like I can do anything"
recommendations = recommend_songs(user_input)
print(recommendations)