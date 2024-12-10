import pandas as pd
import streamlit as st

st.title("Hello, Streamlit!")
st.write("これはとても簡単な Streamlit アプリです！")

# ↓以下を追加
file_buffer = st.file_uploader("ファイルをアップロードしてください")

# Excelファイルの読み込み
df = pd.read_excel("Data1.xlsx")

# 表を表示
st.write("データ", df)
