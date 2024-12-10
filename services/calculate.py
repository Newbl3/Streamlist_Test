import pandas as pd


def calculate(data, selected_columns):
    """
    計算処理を行う関数
    :param data: アップロードされたCSVデータ
    :param selected_columns: ユーザーが選択した表示対象のカラム名
    :return: 計算後のデータフレーム
    """
    # CSVから必要な列のみを抽出し、計算対象として返す
    # ここでは単純に選択列データをそのまま返しているだけですが、
    # 計算処理を行いたい場合はこの部分に適切な計算処理を記述します。
    calc_df = data[["time"] + selected_columns].copy()
    
    # 例えば、何らかの処理を加えたい場合は以下のような操作を追加できます
    # calc_df["adjusted_column"] = calc_df["pelvis_tilt"] * 1.5  # 加工例

    return calc_df
