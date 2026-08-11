import datetime as dt
import meteostat as ms
import numpy as np

def load_and_preprocess(start, end):

    point = ms.Point(51.4776, -0.4619, 25)

    station_info = ms.stations.nearby(point, limit = 1, radius = 50000000).reset_index().iloc[0]

    station_id, station_name = station_info["id"], station_info["name"]

    station = ms.Station(id = station_id)

    df = ms.daily(station, start, end).fetch()

    df = df.dropna(subset=['prcp'])

    df = df.reset_index()

    return df

def compute_extreme_thresholds(df, risk_threshold):

    df = df.copy()
    
    df["month"] = df["time"].dt.month

    monthly_thresholds = df.groupby("month")["prcp"].apply(lambda x: np.percentile(x[x > 0], risk_threshold)).to_dict()

    df["monthly_threshold"] = df["month"].map(monthly_thresholds)

    df['extreme_rain'] = np.where(df['prcp'] > df['monthly_threshold'], 1, 0)

    return df
