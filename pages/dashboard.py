import streamlit as st

from services import calculate, file_service

def display():
    st.header("Upload CSV for Calculation")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file:
        data = file_service.read_csv(uploaded_file)

        calc_df = calculate.calculate(data, "sales", "tax")
        st.line_chart(data=calc_df, x="date", y="sales_with_tax")

def dashboard_page():
    st.subheader("ダッシュボード")
    
    # サンプルデータ表示
    data = {
        "A列": [10, 20, 30],
        "B列": [15, 25, 35],
        "C列": [20, 30, 40],
    }
    df = pd.DataFrame(data)
    st.write("データフレーム表示", df)

