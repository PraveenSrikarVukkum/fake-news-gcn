import pandas as pd
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('punkt')

def load_data(path):
    df = pd.read_csv(path)
    df['label'] = df['label'].map({'REAL': 0, 'FAKE': 1})
    return df

def tokenize_sentences(text):
    from nltk.tokenize import sent_tokenize
    return sent_tokenize(text)

def tfidf_features(df, max_features=5000):
    tfidf = TfidfVectorizer(stop_words='english', max_features=max_features)
    features = tfidf.fit_transform(df['text']).toarray()
    return features, df['label'].values

def split_data(features, labels):
    return train_test_split(features, labels, test_size=0.2, random_state=42)
