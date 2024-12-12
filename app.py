import streamlit as st

from pages import classification, prediction

def main():
    st.title("Motion Classification")    
    prediction.prediction_display()
    classification.classification_display()

if __name__ == "__main__":
    main()
