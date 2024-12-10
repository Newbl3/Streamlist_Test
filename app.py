import streamlit as st

from pages import dashboard

def main():
    st.title("CSV Data Processing and Graph Display")
    dashboard.display()

if __name__ == "__main__":
    main()
