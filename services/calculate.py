import pandas as pd

def calculate(df: pd.DataFrame, pelvis: str, pelvis_list: str) -> pd.DataFrame:
    calc_df = df.copy()
    calc_df['total'] = calc_df[pelvis] * calc_df[pelvis_list]
    return calc_df
