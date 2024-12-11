import pandas as pd


def calculate(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    複数のカラム名を動的に受け取り、それらの積を計算して新しい列 'total' に格納します。

    Args:
        df (pd.DataFrame): 入力データ
        columns (list): 計算対象のカラム名リスト

    Returns:
        pd.DataFrame: 計算後のデータフレーム
    """
    calc_df = df.copy()
    
    # 入力されたカラムリスト内のすべての列を順番に掛け合わせる
    if all(col in calc_df.columns for col in columns):  # 全てのカラムが存在していることを確認
        calc_df['total'] = calc_df[columns].prod(axis=1)  # 各行におけるすべての列の積を計算
    else:
        missing_cols = [col for col in columns if col not in calc_df.columns]
        raise KeyError(f"Missing required columns: {missing_cols}")

    return calc_df

