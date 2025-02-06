import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Load the LSTM model
model = load_model('Dense_Spam_Detection.h5')

# Load the tokenizer (assuming it was saved during training)
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)


# Define the function to preprocess the user's input message
def preprocess_message(message):
    # Convert the message to lowercase
    message = message.lower()
    # Tokenize the message using the loaded tokenizer
    sequence = tokenizer.texts_to_sequences([message])
    # Pad the sequence with zeros so that it has the same length as the sequences used to train the model
    padded_sequence = pad_sequences(sequence, maxlen=50)
    return padded_sequence


# Define the Streamlit app
def app():
    st.title("Spam Detector")

    # Custom background with gradient and text colors
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(to right, #ff7e5f, #feb47b);  /* Gradient from pink to orange */
            background-attachment: fixed;
            background-size: cover;
        }
        .title {
            color: #FF6347;  /* Tomato color */
            font-size: 36px;
            text-align: center;
        }
        .input-field input {
            background-color: #fff8dc;  /* Cornsilk background */
            color: #333;
            border: 2px solid #ff6347;  /* Tomato border color */
            padding: 10px;
            font-size: 18px;
        }
        .input-field input:focus {
            outline: none;
            border-color: #32CD32;  /* Lime green on focus */
        }
        .prediction {
            font-size: 20px;
            font-weight: bold;
        }
        .spam {
            color: red;
        }
        .ham {
            color: green;
        }
        </style>
        """, unsafe_allow_html=True
    )

    # Ask the user to input a message
    message = st.text_input("Enter a message:", key="input_message")

    # Preprocess the message and make a prediction
    if message:
        processed_message = preprocess_message(message)
        prediction = model.predict(processed_message)

        # Display the prediction with color-coded output
        if prediction > 0.5:
            st.markdown(
                f"<p class='prediction spam'>This message is spam with a probability of {prediction[0][0] * 100:.2f}%.</p>",
                unsafe_allow_html=True)
        else:
            st.markdown(
                f"<p class='prediction ham'>This message is ham with a probability of {(1 - prediction[0][0]) * 100:.2f}%.</p>",
                unsafe_allow_html=True)


if __name__ == '__main__':
    app()
