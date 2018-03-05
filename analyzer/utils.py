import os
import pandas as pd
import numpy as np
import datetime as dt


def df_col_to_type(df, col, data_type):
    if data_type == 'TEXT':
        df[col] = df[col].replace(np.nan, 'None')
        df[col] = df[col].astype('U')
    if data_type == 'REAL':
        df[col] = df[col].replace(np.nan, 0)
        df[col] = df[col].astype(float)
    if data_type == 'DATE':
        df[col] = pd.to_datetime(df[col], errors='coerce')
        df[col] = df[col].replace(pd.NaT, None)
        df[col] = df[col].replace(pd.NaT, dt.datetime.today())
        df[col] = df[col].dt.date
    if data_type == 'INT':
        df[col] = df[col].replace(np.nan, 0)
        df[col] = df[col].astype(int)
    return df


def get_right_df(df, merge_col):
    cols = [x for x in df.columns if x[-2:] == '_y']
    df = df[[merge_col] + cols]
    df.columns = ([merge_col] + [x[:-2] for x in df.columns if x[-2:] == '_y'])
    return df

def dir_check(directory):
    if not os.path.isdir(directory):
        os.makedirs(directory)
