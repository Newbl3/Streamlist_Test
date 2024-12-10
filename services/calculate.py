import pandas as pd

def calculate(df: pd.DataFrame, pelvis_tilt: str, pelvis_list: str) -> pd.DataFrame:
    calc_df = df.copy()
    calc_df['total'] = calc_df[pelvis_tilt] * calc_df[pelvis_list]
    return calc_df
