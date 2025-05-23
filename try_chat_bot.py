# Chatbot on the History of PlayStation using Streamlit and NLP

## 1. Required Libraries

import streamlit as st
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re

nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize


## 2. Load and Preprocess Text

# Load PlayStation history text
with open(r"C:\Users\DELL User\Desktop\Playstation_History.txt", "r") as file:
    raw_text = file.read().replace('\n', ' ')

# Sentence tokenization
sentences = sent_tokenize(raw_text)

# Preprocessing function
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)

# Preprocess all sentences
processed_sentences = [preprocess(s) for s in sentences]


## 3. Similarity Function

def get_most_relevant_sentence(query):
    processed_query = preprocess(query)
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([processed_query] + processed_sentences)
    similarity = cosine_similarity(vectors[0:1], vectors[1:])
    index = np.argmax(similarity)
    return sentences[index]  # Return original sentence


## 4. Chatbot Function

def chatbot(query):
    response = get_most_relevant_sentence(query)
    return response


## 5. Streamlit App

def main():
    st.title("PlayStation History Chatbot")
    st.write("Ask me anything about the history of PlayStation!")
    user_input = st.text_input("Your question:", "When was the first PlayStation released?")
    if user_input:
        response = chatbot(user_input)
        st.write("**Chatbot:**", response)

if __name__ == '__main__':
    main()

