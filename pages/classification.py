import streamlit as st
from services import calculate, file_service
import numpy as np
import pandas as pd
from io import StringIO
import matplotlib.pyplot as plt
from joblib import load

####################################
#       使用するモデル読み込み       #
####################################
model = load(open("./model/svm_model_rbf_best.pkl", "rb"))
scaler = load(open("./model/scaler.pkl", "rb"))
imputer = load(open("./model/imputer.pkl", "rb"))

####################################
#             動作分類部            #
####################################
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

####################################
#             表示部分              #
####################################
def classification_display():
    #サイドバー部分
    st.sidebar.header("SideBar")
    st.sidebar.write("あああ")
    st.sidebar.button("Click me!",key = "button2")
    st.sidebar.selectbox("Choose an option:", ["Option 1", "Option 2", "Option 3"],key = "selectbox2")

    #メイン部分
    st.header("Upload CSV for Classification")
    # ファイルアップロード
    uploaded_file_2 = st.file_uploader("Choose a CSV file", type="csv",key = "file2")
    if uploaded_file_2:
        # データの読み込みと処理
        st.write("File uploaded. Processing start...")
        prediction_counts, prediction = classify_movement(uploaded_file_2, model, scaler, imputer)
        
        
        # 結果表示(省略できる部分)
        st.write(f"動作分類結果: {prediction}")
        st.write("分類の詳細（テーブル表示）:")
        
        # dict を DataFrame に変換
        prediction_counts_df = pd.DataFrame(list(prediction_counts.items()), columns=["動作種目", "予測数"])
        st.dataframe(prediction_counts_df)

        # 動作種目ごとの予測数をグラフ化
        st.write("動作種目ごとの予測数（グラフ表示）:")

        # Matplotlibを用いた棒グラフの描画
        fig, ax = plt.subplots()
        ax.bar(prediction_counts_df["動作種目"], prediction_counts_df["予測数"], color='skyblue')
        ax.set_xlabel("Motion name")
        ax.set_ylabel("Prediction count")
        ax.set_title("Motion Prediction Count")
        ax.set_xticks(range(len(prediction_counts_df["動作種目"])))
        ax.set_xticklabels(prediction_counts_df["動作種目"], rotation=45, ha="right")
        st.pyplot(fig)
    else:
        st.write("Please upload a CSV file.")


####################################
#             メイン部分            #
####################################
if __name__ == "__main__":
    classification_display()
