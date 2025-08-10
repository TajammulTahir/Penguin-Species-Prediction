# Penguin Species Prediction
# Predict penguin species based on physical measurements.

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from seaborn import load_dataset

# Load Data
@st.cache_data
def load_data():
    penguins = load_dataset("penguins").dropna()
    X = penguins[["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]]
    y = penguins["species"]
    return X, y, penguins["species"].unique()

X, y, species_names = load_data()

# Train Model
model = RandomForestClassifier()
model.fit(X, y)

# Sidebar Input
st.sidebar.title("Penguin Features")
bill_length = st.sidebar.slider("Bill Length (mm)", float(X["bill_length_mm"].min()), float(X["bill_length_mm"].max()))
bill_depth = st.sidebar.slider("Bill Depth (mm)", float(X["bill_depth_mm"].min()), float(X["bill_depth_mm"].max()))
flipper_length = st.sidebar.slider("Flipper Length (mm)", float(X["flipper_length_mm"].min()), float(X["flipper_length_mm"].max()))
body_mass = st.sidebar.slider("Body Mass (g)", float(X["body_mass_g"].min()), float(X["body_mass_g"].max()))

# Prediction
prediction = model.predict([[bill_length, bill_depth, flipper_length, body_mass]])[0]

st.write("Prediction")
st.write(f"The predicted penguin species is {prediction}")
