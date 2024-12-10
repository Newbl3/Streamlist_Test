import streamlit as st
import pandas as pd
from services import calculate, file_service

def display():
    st.header("Upload CSV for Calculation")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file:
        # アップロードしたデータを読み込む
        data = file_service.read_csv(uploaded_file)

        if data is not None:
            st.write("アップロードデータ:", data.head())

            # 計算対象のデータを動的に取得して処理
            selected_columns = st.multiselect(
                "Select the columns to visualize",
                options=data.columns[1:],  # 1列目以外を選択肢として表示
                default=data.columns[1:6]  # デフォルトで最初の5列を選択状態にする
            )

            if selected_columns:
                # 計算部分に渡して処理
                calc_df = calculate.calculate(data, selected_columns)

                # 計算結果をチャートで表示
                st.line_chart(data=calc_df, x="time", y=selected_columns)

        else:
            st.error("データが正しく読み込めていません。")
