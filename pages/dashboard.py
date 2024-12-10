import streamlit as st
from services import calculate, file_service
import os
import numpy as np
import pandas as pd
from io import StringIO
import pickle

# モデルとスケーラーの読み込み
model = pickle.load(open(f'./model/svm_model_rbf_best.pkl', 'rb'))
scaler = pickle.load(open(f'./model/scaler.pkl', 'rb'))
imputer = pickle.load(open(f'./model/imputer.pkl', 'rb'))

def classify_movement(uploaded_file, model, scaler, imputer):
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
    return final_prediction_counts, final_prediction

def display():
    st.header("Upload CSV for Calculation")

    # ファイルアップロード
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file:
        # データの読み込みと処理
        st.write("ファイルがアップロードされました。処理を開始します...")
        prediction_counts, prediction = classify_movement(uploaded_file, model, scaler, imputer)
        
        # 結果表示
        st.write(f"動作分類結果: {prediction}")
        st.write(f"分類の詳細: {prediction_counts}")
    else:
        st.write("CSVファイルをアップロードしてください。")

# メイン実行
if __name__ == "__main__":
    display()
