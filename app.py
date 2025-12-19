import streamlit as st
from transformers import pipeline

# Load sentiment analysis pipeline
classi = pipeline("sentiment-analysis")

# Streamlit UI
st.title("Sentiment Analysis App")

text = st.text_area("Enter text for sentiment analysis")

if st.button("Analyze Sentiment"):
    if text.strip() == "":
        st.warning("Please enter some text")
    else:
        result = classi(text)
        st.success(f"Sentiment: {result[0]['label']}")
        st.info(f"Confidence Score: {result[0]['score']:.2f}")
