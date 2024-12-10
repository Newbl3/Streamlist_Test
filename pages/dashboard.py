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

def classify_uploaded_csv(uploaded_files, model, scaler, imputer):
    #モデルとスケーラーの読み込み
    model = pickle.load(open(f'/model/svm_model_rbf_best.pkl', 'rb'))
    scaler = pickle.load(open(f'/model/scaler.pkl', 'rb'))
    imputer = pickle.load(open(f'/model/imputer.pkl', 'rb'))

    # Load new data from uploaded file
    string_data = StringIO(uploaded_file.getvalue().decode("utf-8"))
    new_data = pd.read_csv(string_data)

    # Remove 'time' column if it exists
    if 'time' in new_data.columns:
        new_data = new_data.drop(columns=['time'])
        # Remove first and last 30 frames
        new_data = new_data.iloc[60:-60].reset_index(drop=True)
        # Create moving window data
        window_size = 30
        X_new = [new_data.iloc[i:i + window_size].values.flatten()
                 for i in range(0, len(new_data) - window_size + 1)]
        X_new = np.array(X_new)
        # Handle missing values
        X_new = imputer.transform(X_new)
        # Standardize the data
        X_new = scaler.transform(X_new)
        # Make predictions for each frame
        predictions = model.predict(X_new)
        # Count occurrences of each label and decide final prediction using majority vote
        unique, counts = np.unique(predictions, return_counts=True)
        final_prediction_counts = dict(zip(unique, counts))
        final_prediction = unique[np.argmax(counts)].lower()  # Convert to lowercase
        predicted_labels.append(final_prediction)
        # Display results for the file
        st.write(f"File: {file_name} | True Label: {true_label} | Label Counts: {final_prediction_counts} | Final Prediction: {final_prediction}")

        
    pred = classify_movement_in_folder(training_folder, svm_model, scaler, imputer, labels)
    st.write("動作分類結果",:pred)
