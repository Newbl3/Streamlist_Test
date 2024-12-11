import streamlit as st
import pickle
import numpy as np

# サイドバー設定
st.sidebar.title("動作分類アプリ")
st.sidebar.write("SVM モデルを使用して動作を分類します。")

# モデルのロード
def load_model(model_path):
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"モデルのロード中にエラーが発生しました: {e}")
        return None

model_path = "svm_model_rbf_best.pkl"
model = load_model(model_path)

if model is not None:
    # 入力値の設定
    st.title("動作分類")
    st.write("以下のフィールドに入力してください。")

    # サンプル用特徴量の入力 (変更可能)
    feature_1 = st.number_input("特徴量 1", value=0.0, format="%.2f")
    feature_2 = st.number_input("特徴量 2", value=0.0, format="%.2f")
    feature_3 = st.number_input("特徴量 3", value=0.0, format="%.2f")
    feature_4 = st.number_input("特徴量 4", value=0.0, format="%.2f")

    # 入力値を配列に変換
    features = np.array([[feature_1, feature_2, feature_3, feature_4]])

    # 分類ボタン
    if st.button("分類を実行"):
        try:
            # 予測実行
            prediction = model.predict(features)
            st.success(f"予測されたクラス: {prediction[0]}")
        except Exception as e:
            st.error(f"分類中にエラーが発生しました: {e}")
else:
    st.error("モデルが正しくロードされていません。アプリを終了して、モデルを確認してください。")
