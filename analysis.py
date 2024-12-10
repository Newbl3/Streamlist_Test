import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix
import streamlit as st
from io import StringIO

def classify_uploaded_csv(uploaded_files, model, scaler, imputer):
    true_labels = []
    predicted_labels = []

    if not uploaded_files:
        st.warning("No files uploaded. Please upload CSV files.")
        return

    for uploaded_file in uploaded_files:
        # Read file name and true label
        file_name = uploaded_file.name
        true_label = standardize_label(file_name).lower()  # Replace this with your label standardization function
        true_labels.append(true_label)

        # Load new data from uploaded file
        string_data = StringIO(uploaded_file.getvalue().decode("utf-8"))
        new_data = pd.read_csv(string_data)

        # Remove 'time' column if it exists
        if 'time' in new_data.columns:
            new_data = new_data.drop(columns=['time'])

        # Remove first and last 30 frames
        new_data = new_data.iloc[60:-60].reset_index(drop=True)

        # Create moving window data
        window_size = 30
        X_new = [new_data.iloc[i:i + window_size].values.flatten()
                 for i in range(0, len(new_data) - window_size + 1)]
        X_new = np.array(X_new)

        # Handle missing values
        X_new = imputer.transform(X_new)

        # Standardize the data
        X_new = scaler.transform(X_new)

        # Make predictions for each frame
        predictions = model.predict(X_new)

        # Count occurrences of each label and decide final prediction using majority vote
        unique, counts = np.unique(predictions, return_counts=True)
        final_prediction_counts = dict(zip(unique, counts))
        final_prediction = unique[np.argmax(counts)].lower()  # Convert to lowercase
        predicted_labels.append(final_prediction)

        # Display results for the file
        st.write(f"File: {file_name} | True Label: {true_label} | Label Counts: {final_prediction_counts} | Final Prediction: {final_prediction}")

# Streamlit app
st.title("Movement Classification")

st.write("Upload your CSV files to classify movements using the trained model.")

# File uploader
uploaded_files = st.file_uploader("Upload CSV files", accept_multiple_files=True, type=['csv'])

# Dummy placeholders for model, scaler, and imputer (replace with your trained objects)
# Example: Load your model, scaler, and imputer here.
svm_model = None  # Replace with your trained model
scaler = None     # Replace with your scaler
imputer = None    # Replace with your imputer

def standardize_label(file_name):
    # Replace this with your logic to extract and standardize labels from file names
    return file_name.split("_")[0]

if uploaded_files and svm_model and scaler and imputer:
    classify_uploaded_csv(uploaded_files, svm_model, scaler, imputer)
else:
    st.warning("Please upload files and ensure the model, scaler, and imputer are properly loaded.")
