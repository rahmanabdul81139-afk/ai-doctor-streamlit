import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

st.set_page_config(page_title="AI Doctor", page_icon="🩺")
st.title("🩺 AI Doctor – Medical Test Recommendation")

# ---- LOAD DATASET ----
@st.cache_data
def load_data():
    return pd.read_excel("symptoms based medical test recommendations (2).xlsx")

data = load_data()

data = data.rename(columns={
    "Questions": "symptoms",
    "Recommending medical tests": "test"
})

# ---- TRAIN MODEL ----
X = data["symptoms"]
y = data["test"]

vectorizer = CountVectorizer()
X_vec = vectorizer.fit_transform(X)

model = MultinomialNB()
model.fit(X_vec, y)

# ---- USER INPUT ----
user_input = st.text_area(
    "Enter your symptoms 👇",
    placeholder="eg: fever, headache, body pain"
)

if st.button("Recommend Medical Test"):
    if user_input.strip() == "":
        st.warning("Please enter symptoms ⚠️")
    else:
        user_vec = vectorizer.transform([user_input])
        prediction = model.predict(user_vec)

        st.success("✅ Recommended Medical Test")
        st.write(f"🧪 **{prediction[0]}**")
