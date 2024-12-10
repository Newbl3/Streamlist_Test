import pandas as pd

def calculate(df: pd.DataFrame, pelvis: str, pelvis_list: str) -> pd.DataFrame:
    calc_df = df.copy()
    calc_df['sales_with_tax'] = calc_df[pelvis] * calc_df[pelvis_list]
    return calc_df
