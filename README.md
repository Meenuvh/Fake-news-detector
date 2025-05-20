# Fake-news-detector
A Streamlit app for detecting fake news using NLP and ML
# 📰 Fake News Detection App (NLP + Streamlit)

This project is a machine learning web app that classifies whether a news article is **real** or **fake** based on its content.

## 🔍 Overview
- Built using **TF-IDF Vectorization** + **Logistic Regression**
- Preprocessed text with stopword removal and punctuation filtering
- Deployed locally using **Streamlit**

## 🧠 Model Performance
- Achieved ~94% accuracy on test set
- Limitations include dataset bias and poor context awareness (classic problem with TF-IDF)
- Example issue: it may classify real modern articles as fake due to word associations

## ⚙️ Tech Stack
- Python, Pandas, Scikit-learn, NLTK
- Streamlit for frontend
- Trained on [Fake and Real News Dataset (Kaggle)](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

## 📦 Run Locally
1. Clone this repo
2. Install dependencies:
pip install -r requirements.txt
3. Run:
streamlit run app.py

## 🚧 Future Work
- Replace TF-IDF with BERT or other deep NLP models
- Add explainability (SHAP, confidence scores)
- Host the app publicly using Streamlit Cloud

