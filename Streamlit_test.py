# ワイドレイアウト
st.set_page_config(layout="wide")

# テキスト(マークダウンで書けます。)
st.write("# title")

# テーブル
import pandas as pd
df = pd.DataFrame(
        {
            "first column": [1, 2, 3, 4],
            "second column": [10, 20, 30, 40],
        }
    )
st.write(df)
