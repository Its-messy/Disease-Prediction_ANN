#Importing Libraries
import streamlit as st 
from sklearn.model_selection import train_test_split
import pandas as pd
#pickle Importing
import joblib

#Loading the  pre-trained model

model = joblib.load('Keras ANN.pkl')

#Loading Display and Accuracy
with open('accuracy.txt', 'r') as file:
    accuracy = file.read()

st.title("Model Accuracy and Real-time Prediction")
st.write(f"model {accuracy}")

#User Inputs for real-time predictions
st.header("Real-Time Prediction")