import pandas as pd
import streamlit as st

# タイトル
st.title("売上データの表示")

# Excelファイルの読み込み
df = pd.read_excel("Data1.xlsx")

# 表を表示
st.write("データ", df)

