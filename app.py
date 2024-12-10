import streamlit as st

from pages import dashboard

def main():
    st.title("CSV Data Processing and Graph Display")

    model = pickle.load(open(f'{path}/pkl/clf.pkl', 'rb'))
    pred = model.predict(datasets)
    st.write("動作分類結果",:pred)
    dashboard.display()

if __name__ == "__main__":
    main()
