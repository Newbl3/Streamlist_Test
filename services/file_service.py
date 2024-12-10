import pandas as pd

#修正前
#def read_csv(filename: str) -> pd.DataFrame:
 #   return pd.read_csv(filename)

#修正後
def read_csv(filename: str) -> pd.DataFrame:
    """
    CSVファイルを安全に読み込む関数
    :param filename: 読み込むCSVファイルのパス
    :return: pandas DataFrame
    """
    try:
        # CSVを読み込んで返す。エンコーディングと区切り文字を指定
        data = pd.read_csv(filename, encoding="utf-8", sep=",")
        
        # 読み込んだデータのヘッダー確認ログ出力
        print("Columns in uploaded CSV:", data.columns)

        return data
    except FileNotFoundError:
        raise FileNotFoundError(f"ファイルが見つかりません: {filename}")
    except pd.errors.EmptyDataError:
        raise ValueError("アップロードしたCSVデータが空です。")
    except pd.errors.ParserError:
        raise ValueError("CSVデータの形式が不正です。")
    except Exception as e:
        raise Exception(f"予期せぬエラーが発生しました: {e}")

