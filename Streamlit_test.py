# テキスト(マークダウンで書けます。)
st.write("# title")

# 注釈
st.caption("注釈")

# テーブル
import pandas as pd
df = pd.DataFrame(
        {
            "first column": [1, 2, 3, 4],
            "second column": [10, 20, 30, 40],
        }
    )
st.write(df)

# チャート
st.line_chart(df)
