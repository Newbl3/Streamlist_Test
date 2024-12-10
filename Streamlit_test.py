import pandas as pd
import streamlit as st

# タイトル
st.title("売上データの表示")

# ファイル読み込みのエラーハンドリング
try:
    # Excelファイルの読み込み
    df = pd.read_excel("Data1.xlsx")
    # 表を表示
    st.write("データ", df)
except FileNotFoundError:
    st.error("エラー: 指定されたファイル 'Data1.xlsx' が見つかりませんでした。")
except Exception as e:
    st.error(f"エラー: ファイルを読み込む際に問題が発生しました。詳細: {e}")

