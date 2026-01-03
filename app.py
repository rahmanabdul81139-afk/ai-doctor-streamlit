import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

st.set_page_config(page_title="AI Doctor", page_icon="🩺")

st.title("🩺 AI Doctor – Disease & Medical Test Recommendation (FINAL)")

# -------- LOAD DATA --------
data = pd.read_excel("symptoms based medical test recommendations (2).xlsx")

data = data.rename(columns={
    "Questions": "symptoms",
    "Disease": "disease",
    "Recommending medical tests": "test"
})

# -------- VECTORIZATION --------
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data["symptoms"])

# -------- MODELS --------
disease_model = MultinomialNB()
disease_model.fit(X, data["disease"])

test_model = MultinomialNB()
test_model.fit(X, data["test"])

# -------- USER INPUT --------
user_input = st.text_area(
    "Enter your symptoms 👇",
    placeholder="eg: fever, headache, body pain"
)

if st.button("Get Diagnosis"):
    if user_input.strip() == "":
        st.warning("Please enter symptoms ⚠️")
    else:
        user_vec = vectorizer.transform([user_input])

        disease_pred = disease_model.predict(user_vec)[0]
        test_pred = test_model.predict(user_vec)[0]

        st.success("Diagnosis Result")
        st.write("🦠 **Possible Disease:**", disease_pred)
        st.write("🧪 **Recommended Medical Test:**", test_pred)

       
