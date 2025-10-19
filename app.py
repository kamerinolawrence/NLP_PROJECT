import streamlit as st
import joblib
import re
import string
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import warnings
warnings.filterwarnings("ignore", message="Some weights of the model checkpoint")

# Streamlit page config
st.set_page_config(page_title="Sentiment Classifier", page_icon="🧩", layout="centered")
st.markdown(
    """
    <h1 style='text-align: center; color: #4B9CD3;'>
        🎯 Sentiment Classifier
    </h1>
    """,
    unsafe_allow_html=True
)
# NLTK setup 
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

# Text cleaning for SVM model
def clean_tweet(text):
    text = text.lower()
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\d+', '', text)
    
    negation_patterns = {
        r'\b(am|are) not a fan\b': 'dislike',
        r'\b(do|did|does) not like\b': 'dislike',
        r'\b(do|did|does|is) not good\b': 'bad',
        r'\b(do|did|does|is) not great\b': 'poor',
        r'\b(am|are) not happy\b': 'unhappy',
        r'\b(do|did|does) not worth\b': 'worthless',
        r'\b(do|did|does) not recommend\b': 'avoid',
        r'\b(is|was|are) not bad\b': 'good',
        r'\b(is|was|are) not terrible\b': 'good',
        r'\b(am|are|do|did|does) not unhappy\b': 'happy',
        r'\b(am|is|are) not poor\b': 'good',
        r'\b(do|did|does) not want\b': 'dislike',
        r'\b(do|did|does) not need\b': 'dislike'
    }

    for pattern, replacement in negation_patterns.items():
        text = re.sub(pattern, replacement, text)
    
    # General negation (keep as last)
    text = re.sub(r'\b(do|did|does|am|is|are|was|were) not (\w+)\b', r'not_\2', text)

    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    return text.strip()

# Load SVM model 
@st.cache_resource
def load_svm_model():
    return joblib.load("sentiment_pipeline.pkl")

svm_model = load_svm_model()

# Load BERT model
@st.cache_resource
def load_bert_model():
    model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

tokenizer, bert_model = load_bert_model()


st.title("Sentiment Classifier")
st.write("Compare predictions from your **SVM** and **BERT** models.")

st.markdown("---")

# User input 
user_input = st.text_area("Enter a tweet or text:", placeholder="e.g. I do not like the new iPhone")

# Buttons
col1, col2 = st.columns(2)

with col1:
    svm_clicked = st.button("Analyze with SVM")
with col2:
    bert_clicked = st.button("Analyze with BERT")

# Run SVM model 
if svm_clicked and user_input.strip():
    cleaned_text = clean_tweet(user_input)
    prediction = svm_model.predict([cleaned_text])[0]

    if "neg" in prediction.lower() or prediction.lower() in ["negative", "bad", "dislike"]:
        st.error(f"**SVM Sentiment:** {prediction}")
    elif "pos" in prediction.lower() or prediction.lower() in ["positive", "good", "happy"]:
        st.success(f"**SVM Sentiment:** {prediction}")
    else:
        st.info(f"**SVM Sentiment:** {prediction}")

# Run BERT model 
if bert_clicked and user_input.strip():
    inputs = tokenizer(user_input, return_tensors="pt", truncation=True)
    outputs = bert_model(**inputs)
    scores = torch.softmax(outputs.logits, dim=1)[0]
    labels = ["Negative", "Neutral", "Positive"]

    sentiment = labels[torch.argmax(scores).item()]
    confidence = scores.max().item()

    st.markdown(f"**BERT Sentiment:** {sentiment}")
    st.write(f"**Confidence:** {confidence:.2f}")

    if sentiment == "Positive":
        st.success("😊 Positive")
    elif sentiment == "Negative":
        st.error("😠 Negative")
    else:
        st.warning("😐 Neutral")

elif not user_input.strip():
    st.warning("Please enter text to analyze.")

st.markdown("---")
st.caption("Built with Streamlit • SVM • and Transformers")