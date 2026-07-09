import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Spam Classifier", page_icon="📩", layout="centered")

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("ℹ️ About this model")
    st.markdown(
        """
        **Algorithm:** Multinomial Naive Bayes  
        **Vectorizer:** TF-IDF (max 3000 features)  
        **Dataset:** UCI SMS Spam Collection (~5,169 messages)  
        **Preprocessing:** lowercasing, tokenization,
        stopword removal, Porter stemming
        """
    )
    st.divider()
    st.subheader("📊 Model Performance")
    col1, col2 = st.columns(2)
    col1.metric("Accuracy", "97.1%")
    col2.metric("Precision", "100%")
    st.caption(
        "Naive Bayes was chosen over Random Forest and XGBoost "
        "for its perfect precision — it never mislabels a real "
        "message as spam."
    )
    with st.expander("Compare all models tested"):
        st.markdown(
            """
            | Model | Accuracy | Precision |
            |---|---|---|
            | **Naive Bayes** ✅ | 97.10% | **100%** |
            | Random Forest | 97.39% | 98.26% |
            | XGBoost | 96.81% | 94.87% |
            """
        )

# ---------------- MAIN ----------------
st.title("📩 Email/SMS Spam Classifier")
st.write("Paste a message below and check whether it's spam or not, "
         "along with the model's confidence.")

input_sms = st.text_area("Enter the message", height=120)

if st.button('Predict', type="primary"):

    if not input_sms.strip():
        st.warning("Please enter a message first.")
    else:
        # 1. preprocess
        transformed_sms = transform_text(input_sms)
        # 2. vectorize
        vector_input = tfidf.transform([transformed_sms])
        # 3. predict
        result = model.predict(vector_input)[0]

        # confidence — MultinomialNB supports predict_proba
        confidence = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(vector_input)[0]
            confidence = proba[int(result)] * 100

        # 4. Display result
        st.divider()
        if result == 1:
            st.error("🚨 **Spam**")
        else:
            st.success("✅ **Not Spam**")

        if confidence is not None:
            st.metric("Model confidence", f"{confidence:.1f}%")

        with st.expander("🔍 See preprocessed text (what the model actually sees)"):
            st.code(transformed_sms if transformed_sms else "(empty after preprocessing)")