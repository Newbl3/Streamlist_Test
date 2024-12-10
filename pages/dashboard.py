import streamlit as st

from services import calculate, file_service

def display():
    st.header("Upload CSV for Calculation")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file:
        #ファイルアップローダー
        data = file_service.read_csv(uploaded_file)
        # 修正後
        calc_df = calculate.calculate(
            data, 
            ["pelvis_tilt", "pelvis_list", "pelvis_rotation", "pelvis_tx", "pelvis_ty", "pelvis_tz",
                "hip_flexion_r", "hip_adduction_r", "hip_rotation_r", "knee_angle_r", "ankle_angle_r",
                "subtalar_angle_r", "mtp_angle_r", "hip_flexion_l", "hip_adduction_l", "hip_rotation_l","knee_angle_l", 
                "ankle_angle_l", "subtalar_angle_l", "mtp_angle_l", "lumbar_extension","lumbar_bending", "lumbar_rotation", 
                "arm_flex_r", "arm_add_r", "arm_rot_r", "elbow_flex_r","pro_sup_r", "arm_flex_l", "arm_add_l", 
                "arm_rot_l", "elbow_flex_l", "pro_sup_l"])
        
        # UIでX軸とY軸の選択肢を作成
        columns_list = calc_df.columns.tolist()
        x_column = st.selectbox("Select X-axis column", options=calc_df.columns)
        y_column = st.selectbox("Select Y-axis column", options=calc_df.columns)

        # グラフプロット
        st.line_chart(data=calc_df, x=x_column, y=y_column)


        #モデルとスケーラーの読み込み
        model = pickle.load(open(f'/pkl/svm_model_rbf_best.pkl', 'rb'))
        scaler = pickle.load(open(f'/pkl/scaler.pkl', 'rb'))
        imputer = pickle.load(open(f'/pkl/imputer.pkl', 'rb'))
        pred = classify_movement_in_folder(training_folder, svm_model, scaler, imputer, labels)
        st.write("動作分類結果",:pred)
