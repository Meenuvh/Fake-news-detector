import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk

# ✅ Only download stopwords
nltk.download('stopwords')

# ✅ Set of stopwords
stop_words = set(stopwords.words('english'))

# ✅ Clean function with NO word_tokenize
def clean_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    tokens = text.split()  # ✅ This avoids word_tokenize & punkt
    cleaned = [word for word in tokens if word not in stop_words]
    return ' '.join(cleaned)

# ✅ Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# ✅ Streamlit UI
st.title("📰 Fake News Detection App")
input_text = st.text_area("Paste a news article or headline here")

if st.button("Predict"):
    cleaned = clean_text(input_text)
    transformed = vectorizer.transform([cleaned])
    prediction = model.predict(transformed)[0]
    probability = model.predict_proba(transformed)[0][prediction]

    if prediction == 1:
        st.error(f"🟥 Prediction: Fake News ({round(probability * 100, 2)}% confidence)")
    else:
        st.success(f"🟩 Prediction: Real News ({round(probability * 100, 2)}% confidence)")
