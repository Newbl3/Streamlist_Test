import pandas as pd

def calculate(df: pd.DataFrame, sales_column: str, tax_column: str) -> pd.DataFrame:
    calc_df = df.copy()
    calc_df['sales_with_tax'] = calc_df[sales_column] * calc_df[tax_column]
    return calc_df