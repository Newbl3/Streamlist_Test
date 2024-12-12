import streamlit as st
from services import calculate, file_service
import pandas as pd
import matplotlib.pyplot as plt
from joblib import load
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import json

####################################
#       使用するモデル読み込み       #
####################################
randomforest_model = load(open("./model/random_forest_model.joblib", "rb"))

####################################
#            ファイル処理部         #
####################################
def extract_data(new_data):
    # 関節角度データと筋張力データを抽出する処理を記述
    new_data['glmax_total_r'] = new_data[['glmax1_r', 'glmax2_r', 'glmax3_r']].sum(axis=1)
    new_data['glmax_total_l'] = new_data[['glmax1_l', 'glmax2_l', 'glmax3_l']].sum(axis=1)

    with open("./pages/columns.json", "r") as file:
        columns = json.load(file)
    
    #列名リスト取得
    joint_angle_columns = columns["joint_angles"]
    muscle_tension_columns = columns["muscle_tensions"]

    #必要なデータを抽出
    joint_angles = new_data[joint_angle_columns]
    muscle_tensions = new_data[muscle_tension_columns]

    return joint_angles, muscle_tensions

####################################
#             筋張力予測部          #
####################################
def predict_with_new_data(model, new_data_file):
    new_data = pd.read_csv(new_data_file)  # 新しいデータの読み込み
    new_data = new_data.iloc[200:-30]  # 最初と最後の30フレームを無視

    new_joint_angles, actual_muscle_tensions = extract_data(new_data)  # 新しい関節角度データと筋張力データの抽出

    # 新しいデータを使用して予測
    new_muscle_tensions_pred = model.predict(new_joint_angles)

    # 結果をデータフレームに格納
    new_muscle_tensions_df = pd.DataFrame(new_muscle_tensions_pred, columns=actual_muscle_tensions.columns)
    actual_muscle_tensions_df = actual_muscle_tensions.reset_index(drop=True)  # インデックスをリセット

    # 各筋肉のR², MAE, MSEを計算して出力
    st.write("Evaluation Metrics (R², MAE, MSE) for Each Muscle Tension:")
    for target in actual_muscle_tensions.columns:
        r2 = r2_score(actual_muscle_tensions_df[target].values, new_muscle_tensions_df[target].values)
        mae = mean_absolute_error(actual_muscle_tensions_df[target].values, new_muscle_tensions_df[target].values)
        mse = mean_squared_error(actual_muscle_tensions_df[target].values, new_muscle_tensions_df[target].values)

        st.write(f"{target} - R²: {r2:.2f}, MAE: {mae:.2f}, MSE: {mse:.2f}")

    # プロット処理
    num_columns = new_data.shape[1]
    
    # 35以上の列数の場合、予測と実際の筋張力をプロット
    if num_columns > 35:
        st.write("Actual vs Predicted Muscle Tension:")
        fig, ax = plt.subplots(len(actual_muscle_tensions.columns), 1, figsize=(10, 30))
        for i, target in enumerate(actual_muscle_tensions.columns):
            ax[i].plot(actual_muscle_tensions_df[target].values, label='Actual', marker='o')
            ax[i].plot(new_muscle_tensions_df[target].values, label='Predicted', marker='x')
            ax[i].set_title(f'Actual vs Predicted Muscle Tension for {target}')
            ax[i].set_xlabel('Sample Index')
            ax[i].set_ylabel('Muscle Tension')
            ax[i].legend()

        st.pyplot(fig)
    
    # 100以下の列数の場合、予測された筋張力のみをプロット
    else:
        st.write("Predicted Muscle Tension:")
        fig, ax = plt.subplots(len(new_muscle_tensions_pred.columns), 1, figsize=(10, 30))
        for i, target in enumerate(new_muscle_tensions_pred.columns):
            ax[i].plot(new_muscle_tensions_df[target].values, label='Predicted', marker='x')
            ax[i].set_title(f'Predicted Muscle Tension for {target}')
            ax[i].set_xlabel('Sample Index')
            ax[i].set_ylabel('Muscle Tension')
            ax[i].legend()

        st.pyplot(fig)


####################################
#             表示部分              #
####################################
def prediction_display():
    #サイドバー部分
    st.sidebar.header("SideBar")
    st.sidebar.write("あああ")
    st.sidebar.button("Click me!",key = "button1")
    st.sidebar.selectbox("Choose an option:", ["Option 1", "Option 2", "Option 3"],key = "selectbox1")
   
    #メイン部分
    st.header("Upload CSV for Classification")
    # ファイルアップロード
    uploaded_file_1 = st.file_uploader("Choose a CSV file", type="csv",key = "file1")
    if uploaded_file_1:
        # データの読み込みと処理
        st.write("File uploaded. Processing start...")
        # モデルのロード
        predict_with_new_data(randomforest_model, uploaded_file_1)

# メイン実行
if __name__ == "__main__":
    prediction_display()
