import streamlit as st

st.title("Hello, Streamlit!")
st.write("これはとても簡単な Streamlit アプリです！")

# ↓以下を追加
file_buffer = st.file_uploader("ファイルをアップロードしてください")
