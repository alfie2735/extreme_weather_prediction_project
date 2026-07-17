import datetime as dt
import meteostat as ms
import numpy as np

def engineer_features(df):

    df = df.copy()
    
    df['pres_delta_1d'] = df['pres'].diff()

    print(df.groupby('extreme_rain')['pres_delta_1d'].mean())

    df["pres_lag_1"] = df["pres"].shift(1)
    df["pres_lag_2"] = df["pres"].shift(2)

    df['pres_delta_3d_rolling'] = df['pres_delta_1d'].rolling(window=3).mean()

    df['wspd_max_3d'] = df['wspd'].rolling(window=3).max()

    df = df.dropna(subset=['pres_lag_1', 'pres_lag_2', 'pres_delta_3d_rolling', 'wspd_max_3d']).reset_index(drop = True)

    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    df = df.drop(["prcp", "month", "snwd", "tsun", "cldc", "wpgt", "time", "monthly_threshold", "tmax", "tmin"], axis = "columns").dropna()

    return df