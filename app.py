import streamlit as st
import pickle
import numpy as np
import re

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# =========================
# Chargement des fichiers
# =========================

tokenizer = pickle.load(
    open("tokenizer.pkl", "rb")
)

label_encoder = pickle.load(
    open("label_encoder.pkl", "rb")
)

model_cnn = load_model("cnn_model.h5")

model_bilstm = load_model("bilstm_model.h5")

# =========================
# Fonction nettoyage
# =========================

def clean_text(text):

    text = text.lower()

    text = re.sub(r"http\S+", "", text)

    text = re.sub(r"@\w+", "", text)

    text = re.sub(r"#", "", text)

    text = re.sub(r"[^\w\s]", "", text)

    return text

# =========================
# Interface Streamlit
# =========================

st.set_page_config(
    page_title="Tweet Classification",
    layout="centered"
)

st.title("🧠 Classification des Tweets")

st.write(
    "Projet Deep Learning pour la classification des tweets de catastrophes."
)

# =========================
# Choix du modèle
# =========================

model_choice = st.selectbox(
    "Choisir un modèle",
    [
        "CNN1D",
        "BiLSTM"
    ]
)

# =========================
# Zone de texte
# =========================

tweet = st.text_area(
    "Entrer un tweet"
)

# =========================
# Bouton prédiction
# =========================

if st.button("Prédire"):

    cleaned = clean_text(tweet)

    sequence = tokenizer.texts_to_sequences(
        [cleaned]
    )

    padded = pad_sequences(
        sequence,
        maxlen=50
    )

    if model_choice == "CNN1D":

        prediction = model_cnn.predict(
            padded
        )

    else:

        prediction = model_bilstm.predict(
            padded
        )

    predicted_class = np.argmax(
        prediction
    )

    label = label_encoder.inverse_transform(
        [predicted_class]
    )

    confidence = np.max(prediction)

    st.success(
        f"Classe prédite : {label[0]}"
    )

    st.info(
        f"Confiance : {confidence:.2f}"
    )