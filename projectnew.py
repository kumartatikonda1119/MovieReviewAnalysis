import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# ------------------------------------------
# 🎨 Custom Styling with HTML + CSS
st.markdown("""
    <style>
        body {
            font-family: 'Segoe UI', sans-serif;
        }
        .title {
            font-size: 40px;
            font-weight: bold;
            color: #00FFD1;
            text-align: center;
            text-shadow: 2px 2px #000000;
        }
        .subtitle {
            font-size: 18px;
            color: #AAAAAA;
            text-align: center;
            margin-bottom: 30px;
        }
        .stTextArea textarea {
            font-size: 16px;
            border-radius: 10px;
            padding: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# ------------------------------------------
# 🧠 App Title & Description
st.markdown("<div class='title'>🎬 Movie Review Sentiment Analyzer</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>✨ Let the ML model decode your review as Positive or Negative!</div>", unsafe_allow_html=True)

# ------------------------------------------
# 🔍 Load Data & Train Model
@st.cache_data
def load_model():
    data = pd.read_csv("imdb.csv")
    x = data['review']
    y = data['sentiment']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
    cv = CountVectorizer(stop_words='english')
    x_train_vec = cv.fit_transform(x_train)
    x_test_vec = cv.transform(x_test)
    model = MultinomialNB()
    model.fit(x_train_vec, y_train)
    accuracy = accuracy_score(y_test, model.predict(x_test_vec))
    return cv, model, accuracy

cv, model, accuracy = load_model()

# ------------------------------------------
# 📝 User Input
st.subheader("📝 Write your movie review here:")
user_review = st.text_area("For example: *This movie was absolutely amazing! A must watch.*", height=150)

min_length = 20

# ------------------------------------------
# 🎯 Predict Button
if st.button("🔍 Analyze Review"):
    if user_review.strip() == "":
        st.warning("⚠️ Please enter a review before submitting.")
    elif len(user_review.strip()) < min_length:
        st.warning(f"Please enter at least {min_length} characters in your review.")
    else:
        transformed = cv.transform([user_review])
        prediction = model.predict(transformed)[0]

        if prediction.lower() == "positive":
            st.success("✅ **Positive Review!** 🎉 Great vibes, amazing energy! 💖")
            st.balloons()
        else:
            st.error("❌ **Negative Review!** 😢 Maybe not a good one to watch...")
            st.markdown("👉 *Next time check IMDb before watching!*")

# ------------------------------------------
# 📊 Accuracy Display
with st.expander("📈 Show Model Accuracy"):
    st.info(f"Model is **{accuracy * 100:.2f}%** accurate based on the IMDb review dataset.")

# ------------------------------------------
# 🔗 Footer
st.markdown("---")
st.markdown("Made with ❤️ by **Kumar Swamy Tatikonda**")
st.markdown(
    "<p style='text-align: center;'>🚀 Powered by Python, Streamlit & Scikit-learn</p>",
    unsafe_allow_html=True
)
