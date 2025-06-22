# SMS-Email-Spam-Classifier
This project is a web-based spam detection system that classifies messages (SMS or emails) as Spam or Not Spam using machine learning techniques.  Built with Python, Scikit-learn, NLTK, and Streamlit, it leverages natural language processing (NLP) to analyze and process raw text before making predictions.

# Features:

1.Predicts whether a message is spam or not in real-time.
2.Simple and clean Streamlit UI.
3.Uses TF-IDF Vectorization for text representation.
4.Trained using Multinomial Naive Bayes algorithm.
5.Includes data cleaning, stemming, stopword removal, and tokenization.

# Technologies Used:
Python
Scikit-learn
NLTK
Pandas
Streamlit

# How to Run Locally:
1.Clone the repository
2.Install requirements:
  pip install -r requirements.txt
3.Run the app:
  streamlit run app.py

# Files:
1.app.py → Main Streamlit application
2.vectorizer.pkl → Trained TF-IDF vectorizer
3.model.pkl → Trained spam detection model
4.spam.csv → Dataset used for training
