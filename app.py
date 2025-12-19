import streamlit as st
from transformers import pipeline

# Page configuration
st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="🧠",
    layout="centered"
)

# Custom CSS for better design
st.markdown("""
<style>
.main {
    background-color: #f7f9fc;
}
.result-box {
    padding: 20px;
    border-radius: 12px;
    margin-top: 15px;
    font-size: 18px;
    text-align: center;
}
.positive {
    background-color: #e6fffa;
    color: #065f46;
}
.negative {
    background-color: #fee2e2;
    color: #7f1d1d;
}
</style>
""", unsafe_allow_html=True)

# Load sentiment analysis pipeline
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis")

classi = load_model()

# App Header
st.markdown("<h1 style='text-align: center;'>🧠 Sentiment Analysis App</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Analyze text sentiment using Transformer model</p>", unsafe_allow_html=True)

st.divider()

# Text input
text = st.text_area(
    "✍️ Enter your text below:",
    height=120,
    placeholder="Example: The product quality is terrible. I'm disappointed."
)

# Button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    analyze_btn = st.button("🔍 Analyze Sentiment", use_container_width=True)

# Prediction
if analyze_btn:
    if text.strip() == "":
        st.warning("⚠️ Please enter some text")
    else:
        with st.spinner("Analyzing sentiment..."):
            result = classi(text)[0]
            label = result["label"]
            score = result["score"]

        if label.upper() == "POSITIVE":
            st.markdown(
                f"<div class='result-box positive'>😊 <b>POSITIVE</b><br>Confidence: {score:.2f}</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<div class='result-box negative'>😞 <b>NEGATIVE</b><br>Confidence: {score:.2f}</div>",
                unsafe_allow_html=True
            )

st.divider()
st.markdown(
    "<p style='text-align: center; font-size: 14px;'>Built with ❤️ using Streamlit & Transformers</p>",
    unsafe_allow_html=True
)
