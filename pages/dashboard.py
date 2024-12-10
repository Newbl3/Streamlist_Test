import streamlit as st

from services import calculate, file_service

def display():
    st.header("Upload CSV for Calculation")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file:
        data = file_service.read_csv(uploaded_file)

        calc_df = calculate.calculate(data, "sales", "tax")
        st.line_chart(data=calc_df, x="date", y="sales_with_tax")
