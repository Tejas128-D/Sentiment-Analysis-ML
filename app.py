import streamlit as st
import string
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ------------------------------
# Load Model & Vectorizer
# ------------------------------
with open('model_and_vectorizer.pkl', 'rb') as f:
    bundle = pickle.load(f)

model = bundle['model']
vectorizer = bundle['vectorizer']

# ------------------------------
# NLTK Setup
# ------------------------------
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
stop_words.discard("not")
stop_words.discard("no")

# ------------------------------
# Text Cleaning Function
# ------------------------------
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)

# ------------------------------
# Custom Styling
# ------------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #1f1c2c, #928dab);
    color: white;
}
.main-container {
    background-color: rgba(0, 0, 0, 0.65);
    padding: 2rem;
    border-radius: 18px;
    max-width: 700px;
    margin: auto;
    box-shadow: 0 0 25px rgba(0,0,0,0.6);
}
textarea {
    background-color: #f5f5f5 !important;
    color: black !important;
    border-radius: 10px;
}
.stButton>button {
    background-color: #ff4b2b;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-weight: bold;
}
.stButton>button:hover {
    background-color: #ff416c;
}
</style>
""", unsafe_allow_html=True)

# ------------------------------
# UI
# ------------------------------
st.markdown('<div class="main-container">', unsafe_allow_html=True)

st.title("🛍️ Amazon Review Sentiment Analyzer")
st.write("💬 Enter a product review and let AI analyze its sentiment instantly.")

user_input = st.text_area("✍️ Type your review here:")

if st.button("🚀 Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a review to analyze.")
    else:
        cleaned_text = clean_text(user_input)
        vectorized_text = vectorizer.transform([cleaned_text])

        prediction = model.predict(vectorized_text)[0]
        probability = max(model.predict_proba(vectorized_text)[0])

        if prediction.lower() == "positive":
            st.success(f"😊 Predicted Sentiment: **{prediction}**")
        else:
            st.error(f"😞 Predicted Sentiment: **{prediction}**")

        st.info(f"📊 Confidence Score: {probability:.2f}")

st.markdown("</div>", unsafe_allow_html=True)