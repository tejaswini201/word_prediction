import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

model = load_model("lstm_model.h5")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

max_len = 6

def predict(text):
    token_list = tokenizer.texts_to_sequences([text])[0]
    token_list = pad_sequences([token_list], maxlen=max_len-1, padding='pre')
    predicted = np.argmax(model.predict(token_list), axis=1)

    for word, index in tokenizer.word_index.items():
        if index == predicted:
            return word

st.title("Next Word Prediction")

text = st.text_input("Enter text")

if st.button("Predict"):
    st.write(predict(text))