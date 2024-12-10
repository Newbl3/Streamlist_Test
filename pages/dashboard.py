import streamlit as st
from services import calculate, file_service
import os
import numpy as np
import pandas as pd
from io import StringIO
import pickle
from sklearn import datasets
#import matplotlib

dataset_name = st.sidebar.selectbox("select Dataset", ("Iris", "Breast Cancer", "Wine dataset"))
classifier_name= st.sidebar.selectbox("select Classifier", ("KNN", "SVM", "RandomForest"))

def get_dataset(dataset_name):
    if dataset_name == "Iris":
        data = dataset.load_iris()
    elif dataset_name =="Breast Cancer":
        data = datasets.load_breast_cancer()
    else:
        data = datasets.load_wine()
    X = data.data
    y = data.target
    return X,y
X,y = get_dataset(dataset_name)
st.write("shape of dataset", X.shape)
st.write("number of classes", len(np.unique(y)))

def add_parameter_ui(clf_name):
    params = dict()
    if clf_name =="KNN":
        K = st.sidebar.slider("K",1,15)
        params["K"] = K
    return params


# フォルダ階層に基づいてモデルファイルへの絶対パスを取得
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # pagesフォルダの親ディレクトリ
model_path = os.path.join(base_dir, "model", "svm_model_rbf_best.pkl")
#scaler_path = os.path.join(base_dir, "model", "scaler.pkl")
#imputer_path = os.path.join(base_dir, "model", "imputer.pkl")

# モデルとスケーラーの読み込み
model = pickle.load(open(model_path, "rb"))
#scaler = pickle.load(open(scaler_path, "rb"))
#imputer = pickle.load(open(imputer_path, "rb"))

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
