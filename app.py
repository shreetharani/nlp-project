# =================== Imports =====================
import streamlit as st
import torch
from sentence_transformers import util
import pickle
from tensorflow.keras.layers import TextVectorization
import numpy as np
from tensorflow import keras

# ========== Page Config ===========
st.set_page_config(page_title="Research Paper Assistant", page_icon="📚", layout="centered")

# ========== Load Saved Files ===========
embeddings = pickle.load(open('models/embeddings.pkl','rb'))
sentences = pickle.load(open('models/sentences.pkl','rb'))
rec_model = pickle.load(open('models/rec_model.pkl','rb'))

loaded_model = keras.models.load_model("models/model.h5")

with open("models/text_vectorizer_config.pkl", "rb") as f:
    saved_text_vectorizer_config = pickle.load(f)

loaded_text_vectorizer = TextVectorization.from_config(saved_text_vectorizer_config)

with open("models/text_vectorizer_weights.pkl", "rb") as f:
    weights = pickle.load(f)
    loaded_text_vectorizer.set_weights(weights)

with open("models/vocab.pkl", "rb") as f:
    loaded_vocab = pickle.load(f)

# ========== Functions ===========

def recommendation(input_paper):
    cosine_scores = util.cos_sim(embeddings, rec_model.encode(input_paper))
    top_similar_papers = torch.topk(cosine_scores, dim=0, k=5, sorted=True)
    papers_list = [sentences[i.item()] for i in top_similar_papers.indices]
    return papers_list

def invert_multi_hot(encoded_labels):
    hot_indices = np.argwhere(encoded_labels == 1.0)[..., 0]
    return np.take(loaded_vocab, hot_indices)

def predict_category(abstract, model, vectorizer, label_lookup):
    preprocessed_abstract = vectorizer([abstract])
    predictions = model.predict(preprocessed_abstract)
    predicted_labels = label_lookup(np.round(predictions).astype(int)[0])
    return predicted_labels

# ========== UI ===========

st.markdown("<h1 style='text-align: center; color: #2E86C1;'>📘 Research Paper Assistant</h1>", unsafe_allow_html=True)
st.markdown("### 🤖 Get paper recommendations and predict the subject area from an abstract")
st.markdown("---")

with st.form("paper_form"):
    st.markdown("#### 📝 Input Details")
    input_paper = st.text_input("Enter Paper Title:")
    new_abstract = st.text_area("Paste Paper Abstract:")
    submitted = st.form_submit_button("🔍 Recommend & Predict")

    if submitted:
        if input_paper.strip() == "" or new_abstract.strip() == "":
            st.warning("⚠️ Please provide both the paper title and abstract.")
        else:
            with st.spinner("🔎 Generating recommendations and predictions..."):
                # Recommendation
                recommend_papers = recommendation(input_paper)
                st.markdown("### 📄 Recommended Papers:")
                for i, paper in enumerate(recommend_papers, 1):
                    st.markdown(f"**{i}.** {paper}")

                st.markdown("---")

                # Prediction
                predicted_categories = predict_category(new_abstract, loaded_model, loaded_text_vectorizer, invert_multi_hot)
                st.markdown("### 🧠 Predicted Subject Area(s):")
                for label in predicted_categories:
                    st.success(f"✅ {label}")


# ========== Footer ===========
st.markdown("""
    <hr style='margin-top:50px;'>
    <p style='text-align: center; color: #555;'>Made with ❤️ by [Shobika and Shree Tharani] |</p>
""", unsafe_allow_html=True)
