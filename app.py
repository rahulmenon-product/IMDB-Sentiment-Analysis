import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

#Load the trained model
model = load_model('Simple_RNN_IMDB.h5')

#converting words into vectors
word_index = imdb.get_word_index()
reverse_word_index = {value:key for key,value in word_index.items() }

#Function to decode the review
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i-3,'?') for i in encoded_review])

#preprocess user input
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word,2)+3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review],maxlen=500)
    return padded_review

#Streamlit Build
st.title("IMDB Movie Review Sentiment Analysis")
st.write('Enter a movie to classify as positive or negative')

#UserInput
user_input = st.text_area("Movie Review")

if st.button('Classify'):
    preprocessed_input = preprocess_text(user_input)
    prediction = model.predict(preprocessed_input)
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    #Display the resuts
    st.write(f'Sentiment : {sentiment}')
    st.write(f'Prediction score: {prediction[0][0]}')
else:
    st.write("Please enter a movie")