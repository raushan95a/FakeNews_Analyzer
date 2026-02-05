# app.py

import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pandas as pd
import numpy as np

# Download NLTK data if not already present
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Initialize the Porter Stemmer and stopwords
port_stem = PorterStemmer()
english_stopwords = set(stopwords.words('english'))

@st.cache_resource
def load_model():
    """Load the pre-trained model and vectorizer"""
    try:
        with open('model.pkl', 'rb') as model_file:
            model = pickle.load(model_file)
        with open('vectorizer.pkl', 'rb') as vectorizer_file:
            vectorizer = pickle.load(vectorizer_file)
        return model, vectorizer
    except FileNotFoundError:
        st.error("Model files not found! Ensure 'model.pkl' and 'vectorizer.pkl' are in this directory.")
        return None, None

def stemming(content):
    """Preprocess text: remove non-letters, lowercase, remove stopwords, stem"""
    text = re.sub('[^a-zA-Z]', ' ', content)
    text = text.lower().split()
    text = [port_stem.stem(word) for word in text if word not in english_stopwords]
    return ' '.join(text)

def predict_news(news_text, model, vectorizer):
    """Predict whether news is real (0) or fake (1)"""
    if not news_text.strip():
        return None, None
    processed = stemming(news_text)
    vect = vectorizer.transform([processed])
    pred = model.predict(vect)[0]
    proba = model.predict_proba(vect)[0]
    return pred, proba

def main():
    st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="wide")
    st.markdown(
        """
        <style>
        .main-header {font-size:3rem; font-weight:bold; color:#1f77b4; text-align:center;}
        .sub-header {font-size:1.2rem; color:#555; text-align:center; margin-bottom:2rem;}
        .prediction-box {padding:1rem; border-radius:8px; text-align:center; font-size:1.2rem;}
        .real {background:#d4edda; color:#155724; border:2px solid #c3e6cb;}
        .fake {background:#f8d7da; color:#721c24; border:2px solid #f5c6cb;}
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<h1 class="main-header">📰 Fake News Detector</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Enter or upload a news article to check authenticity</p>', unsafe_allow_html=True)

    model, vectorizer = load_model()
    if model is None:
        st.stop()

    col1, col2 = st.columns([2, 1])

    with col1:
        input_method = st.radio("Input method:", ["Type/Paste Text", "Upload .txt File"])
        news_text = ""
        if input_method == "Type/Paste Text":
            news_text = st.text_area("Enter the news article:", height=250)
        else:
            uploaded = st.file_uploader("Upload a .txt file", type=["txt"])
            if uploaded:
                news_text = uploaded.read().decode("utf-8")
                st.text_area("File content:", news_text, height=200, disabled=True)

        if st.button("🔍 Analyze News"):
            pred, proba = predict_news(news_text, model, vectorizer)
            if pred is not None:
                with col2:
                    if pred == 0:
                        st.markdown('<div class="prediction-box real">✅ REAL NEWS</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="prediction-box fake">❌ FAKE NEWS</div>', unsafe_allow_html=True)
                    st.write(f"Real Confidence: {proba[0]*100:.1f}%")
                    st.write(f"Fake Confidence: {proba[1]*100:.1f}%")
            else:
                st.error("Please enter or upload some text to analyze!")

    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; color:#888;'>Built with ❤️ using Streamlit | Model trained on WELFake Dataset</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()