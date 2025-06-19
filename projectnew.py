import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Title and UI
st.set_page_config(page_title="Movie Review Sentiment Analyzer", layout="centered")
st.title("🎬 Movie Review Sentiment Analyzer")
st.markdown("Analyze whether a movie review is **Positive** or **Negative** using Machine Learning!")

# Load data
@st.cache_data
def load_data():
    data = pd.read_csv("imdb.csv")
    return data

data = load_data()
x = data['review']
y = data['sentiment']

# Preprocess
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
cv = CountVectorizer(stop_words='english')
x_train_vec = cv.fit_transform(x_train)
x_test_vec = cv.transform(x_test)

# Train model
model = MultinomialNB()
model.fit(x_train_vec, y_train)

# Input area
st.subheader("📝 Enter your movie review:")
user_input = st.text_area("Write your review below...", height=150)

if st.button("Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter a valid review.")
    else:
        transformed_input = cv.transform([user_input])
        prediction = model.predict(transformed_input)[0]
        if prediction.lower() == "positive":
            st.success("🎉 The review is **Positive**!")
        else:
            st.error("😞 The review is **Negative**.")

# Optional: Show accuracy
if st.checkbox("Show model accuracy"):
    y_pred = model.predict(x_test_vec)
    acc = accuracy_score(y_test, y_pred)
    st.write(f"📊 Accuracy of the model: **{acc*100:.2f}%**")
