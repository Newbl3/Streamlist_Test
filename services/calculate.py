import pandas as pd

def calculate(df: pd.DataFrame, hip_flexion_r: str, hip_flexion_l: str) -> pd.DataFrame:
    calc_df = df.copy()
    calc_df['sales_with_tax'] = calc_df[hip_flexion_r] * calc_df[hip_flexion_l]
    return calc_df
