import pandas as pd


def calculate(data, selected_columns):
    """
    計算処理を行う関数
    :param data: アップロードされたCSVデータ
    :param selected_columns: ユーザーが選択した表示対象のカラム名
    :return: 計算後のデータフレーム
    """
    # 存在する列のみを抽出
    valid_columns = [col for col in selected_columns if col in data.columns]

    if not valid_columns:
        return pd.DataFrame()  # 空のDataFrameを返してエラーハンドリング

    # CSVから必要な列のみを抽出し、計算対象として返す
    calc_df = data[["time"] + valid_columns].copy()

    # ここで計算処理を行いたい場合は記述する
    return calc_df
