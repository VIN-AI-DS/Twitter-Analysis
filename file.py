pip install emoji gradio

import pandas as pd
import re
import string
import emoji
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Load the datasets
train_data = pd.read_csv('/content/twitter_training.csv', header=None)
val_data = pd.read_csv('/content/twitter_validation.csv', header=None)

# Assign proper column names since dataset lacks headers
train_data.columns = ['Tweet_ID', 'Entity', 'Sentiment', 'Tweet_description']
val_data.columns = ['Tweet_ID', 'Entity', 'Sentiment', 'Tweet_description']

# Combine training and validation datasets
data = pd.concat([train_data, val_data]).reset_index(drop=True)

# Text Cleaning Function
def clean_text(text):
    if not isinstance(text, str):  # Handle NaN or non-string values
        return ""
    text = re.sub(r'@\w+', '', text)  # Remove mentions
    text = re.sub(r'#\w+', '', text)  # Remove hashtags
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = emoji.replace_emoji(text, replace='')  # Remove emojis
    text = re.sub(f"[{string.punctuation}]", "", text)  # Remove punctuation
    text = text.lower().strip()  # Lowercase and strip extra spaces
    return text

# Clean the text
data['cleaned_text'] = data['Tweet_description'].apply(clean_text)

# Normalize Sentiment column to lowercase and strip spaces
data['Sentiment'] = data['Sentiment'].str.strip().str.lower()

# Correct mapping with lowercase labels
label_mapping = {'positive': 1, 'negative': 0, 'neutral': 2, 'irrelevant': 3}
data['Sentiment'] = data['Sentiment'].map(label_mapping)

# Drop rows where sentiment mapping failed (i.e., NaN values)
data.dropna(subset=['Sentiment'], inplace=True)

print("Unique Sentiment Values After Mapping:", data['Sentiment'].unique())

# Continue with TF-IDF Vectorization and Model Training
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(data['cleaned_text']).toarray()
y = data['Sentiment']

# Split data with stratify to balance classes
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train Model
model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred, target_names=label_mapping.keys()))


#Gradio Interface

def predict_sentiment(tweet):
    cleaned_tweet = clean_text(tweet)
    vectorized_tweet = vectorizer.transform([cleaned_tweet]).toarray()
    prediction = model.predict(vectorized_tweet)[0]
    sentiment_labels = {1: 'Positive', 0: 'Negative', 2: 'Neutral', 3: 'Irrelevant'}
    return sentiment_labels[prediction]

# Gradio Interface
demo = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(label="Enter Tweet"),
    outputs=gr.Textbox(label="Predicted Sentiment"),
    title="Sentiment Analysis using Logistic Regression",
    description="Enter a tweet to predict its sentiment."
)

demo.launch()
