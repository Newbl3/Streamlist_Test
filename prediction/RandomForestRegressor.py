import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import IsolationForest
from collections import defaultdict
import joblib

########################################################################################################
#
#　ファイル処理
#
########################################################################################################
# ディレクトリ内のすべてのCSVファイルを読み込み、結合する
def load_and_combine_data(directory):
    all_data = []
    subject_counts = defaultdict(int)  # 被験者ごとの行数を集計する辞書
    exercise_counts = defaultdict(int)  # 種目ごとの行数を集計する辞書
    date_counts = defaultdict(int)  # 日付ごとの行数を集計する辞書

    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            file_path = os.path.join(directory, filename)
            data = pd.read_csv(file_path)
            all_data.append(data)

            # ファイル名とサイズを出力
            print(f"ファイル名: {filename}, 行数: {data.shape[0]}, 列数: {data.shape[1]}")

            # 被験者名を取得し、行数を加算
            try:
                subject_name = filename.split('_')[0]  # 被験者名はファイル名の最初の部分
                subject_counts[subject_name] += data.shape[0]

                # 種目名を取得し、行数を加算（大文字小文字を区別しない）
                exercise_name = filename.split('_')[1].lower()  # 小文字に変換
                exercise_counts[exercise_name] += data.shape[0]

                # 日付を取得し、行数を加算
                date = filename.split('_')[2]  # ファイル名の3番目の部分が日付
                date_counts[date] += data.shape[0]
            except IndexError:
                print(f"ファイル名 {filename} は予期しない形式です。")

    # 被験者ごとの行数の合計を出力
    for subject, total_rows in subject_counts.items():
        print(f"被験者 {subject} のデータサイズ合計（行数）: {total_rows}")

    # 種目ごとの行数の合計を出力
    for exercise, total_rows in exercise_counts.items():
        print(f"{exercise} のデータサイズ合計（行数）: {total_rows}")

    # 日付ごとの行数の合計を出力
    for date, total_rows in date_counts.items():
        print(f"日付 {date} のデータサイズ合計（行数）: {total_rows}")

    combined_data = pd.concat(all_data, ignore_index=True)  # データを結合
    return combined_data

########################################################################################################
#
#　関節角度データと筋張力データの抽出
#
########################################################################################################
def extract_data(dataset):
    #dataset = remove_outliers_iqr(dataset)
    print(dataset.columns)

    joint_angles = dataset[['pelvis_tilt', 'pelvis_list', 'pelvis_rotation', 'pelvis_tx',
                             'pelvis_ty', 'pelvis_tz', 'hip_flexion_r', 'hip_adduction_r',
                             'hip_rotation_r', 'knee_angle_r', 'ankle_angle_r',
                             'subtalar_angle_r', 'mtp_angle_r', 'hip_flexion_l',
                             'hip_adduction_l', 'hip_rotation_l', 'knee_angle_l',
                             'ankle_angle_l', 'subtalar_angle_l', 'mtp_angle_l',
                             'lumbar_extension', 'lumbar_bending', 'lumbar_rotation',
                             'arm_flex_r', 'arm_add_r', 'arm_rot_r', 'elbow_flex_r',
                             'pro_sup_r', 'arm_flex_l', 'arm_add_l', 'arm_rot_l',
                             'elbow_flex_l', 'pro_sup_l']]
    # 右側と左側の glmax の合計張力を新しい列に追加
    dataset['glmax_total_r'] = dataset[['glmax1_r', 'glmax2_r', 'glmax3_r']].sum(axis=1)
    dataset['glmax_total_l'] = dataset[['glmax1_l', 'glmax2_l', 'glmax3_l']].sum(axis=1)
    # 合算後の筋張力データを取得(Squat)
    #muscle_tensions = dataset[['glmax_total_r', 'glmax_total_l', 'vaslat_r', 'vaslat_l','addmagDist_r','addmagDist_l','recfem_r','recfem_l','bflh_r','bflh_l']]
    muscle_tensions = dataset[['glmax_total_r', 'glmax_total_l', 'vaslat_r', 'vaslat_l','recfem_r','recfem_l','bflh_r','bflh_l']]

    #NaN値の補完
    muscle_tensions = muscle_tensions.ffill().bfill()
    joint_angles = joint_angles.ffill().bfill()

    # 筋張力データに移動平均を適用
    moving_average = muscle_tensions.rolling(window=10, min_periods=1).mean()  # フレーム数で割る処理を排除
    normalized_muscle_tensions = moving_average

    return joint_angles, normalized_muscle_tensions

########################################################################################################
#
#　モデルの学習
#
########################################################################################################
def train_and_evaluate(joint_angles, muscle_tensions):
    # データの分割
    X_train, X_test, y_train, y_test = train_test_split(joint_angles, muscle_tensions,
                                                        test_size=0.2, random_state=1000)

    # モデルの構築
    model = RandomForestRegressor(n_estimators=100, random_state=1000, oob_score=True, verbose=2)

    model.fit(X_train, y_train)

    # トレーニングセットとテストセットでの予測
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # 評価
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)  # トレーニングセットのR²スコア
    test_r2 = r2_score(y_test, y_test_pred)      # テストセットのR²スコア
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)

    print(f'Training R^2 Score (各筋張力): {train_r2}')  # トレーニングセットのR²スコアを表示
    print(f'Test R^2 Score (各筋張力): {test_r2}')        # テストセットのR²スコアを表示
    print(f'Training Mean Absolute Error (各筋張力): {train_mae}')  # トレーニングセットのMAEを表示
    print(f'Test Mean Absolute Error (各筋張力): {test_mae}')        # テストセットのMAEを表示
    print(f'Training Mean Squared Error (各筋張力): {train_mse}')  # トレーニングセットのMAEを表示
    print(f'Test Mean Squared Error (各筋張力): {test_mse}')        # テストセットのMAEを表示

    # グラフの作成
    y_test_df = pd.DataFrame(y_test, columns=muscle_tensions.columns)
    y_test_pred_df = pd.DataFrame(y_test_pred, columns=muscle_tensions.columns)

    plt.figure(figsize=(100, 30))
    for i, target in enumerate(muscle_tensions.columns):
        plt.subplot(len(muscle_tensions.columns), 1, i + 1)
        plt.plot(y_test_df[target].values, label='Actual', marker='o')
        plt.plot(y_test_pred_df[target].values, label='Predicted', marker='x')
        plt.title(f'Actual vs Predicted for {target}')
        plt.xlabel('Sample Index')
        plt.ylabel('Muscle Tension')
        plt.legend()

    plt.tight_layout()
    plt.show()

    # モデルをjoblibで保存
    joblib.dump(model, 'random_forest_model.joblib')
    print("モデルが保存されました: random_forest_model.joblib")

    return model  # 学習したモデルを返す

########################################################################################################
#
#　メイン処理
#
########################################################################################################
# 1. データを格納するディレクトリの指定
data_directory = './prediction/dataset'  # CSVファイルが保存されているディレクトリのパス
# 3. データの読み込み
dataset = load_and_combine_data(data_directory)
# 4. データの形状を確認
print("結合されたデータの形状:")
print(dataset.shape)
# 6. データの取得
joint_angles, muscle_tensions = extract_data(dataset)
print("関節角度データの形状:", joint_angles.shape)
print("筋張力データの形状:", muscle_tensions.shape)
# 8. 学習
print("データセットの学習:")
model = train_and_evaluate(joint_angles, muscle_tensions)
