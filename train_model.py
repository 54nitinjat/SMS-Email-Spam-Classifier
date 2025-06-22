# This script is used to train the TF-IDF vectorizer and Naive Bayes spam classifier.
# It generates 'model.pkl' and 'vectorizer.pkl' files used in app.py

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

# 1. Load dataset
df = pd.read_csv('spam.csv', encoding='latin-1')[['v1', 'v2']]
df = df.rename(columns={'v1': 'label', 'v2': 'text'})
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# 2. Vectorization
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(df['text']).toarray()
y = df['label']

# 3. Train model
model = MultinomialNB()
model.fit(X, y)

# 4. Save fitted vectorizer and model
pickle.dump(tfidf, open('vectorizer.pkl', 'wb'))
pickle.dump(model, open('model.pkl', 'wb'))

print("✅ Model and Vectorizer saved successfully.")
