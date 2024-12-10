import streamlit as st

from pages import dashboard


def main():
    # ページ設定
st.set_page_config(
    page_title="Streamlit Dashboard",
    page_icon="🌟",
    layout="wide",
)

# ページ装飾 CSS
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f0f8ff;
    }
    .title {
        font-size: 3em;
        text-align: center;
        color: #4CAF50;
        margin: 20px 0;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ダッシュボード表示へナビゲーション
st.markdown('<div class="title">Streamlit Dashboard アプリ</div>', unsafe_allow_html=True)

menu = st.sidebar.selectbox("メニュー選択", ["Dashboard", "データ表示"])

if menu == "Dashboard":
    from pages.dashboard import dashboard_page
    dashboard_page()
elif menu == "データ表示":
    st.write("データ表示ページへ")
    st.title("CSV Data Processing and Graph Display")
    dashboard.display()

if __name__ == "__main__":
    main()
