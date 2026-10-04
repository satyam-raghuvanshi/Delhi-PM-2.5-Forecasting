import numpy as np
import pandas as pd
COVARIATES = ["PM10", "NO2", "SO2"]
POLLUTION =["PM2.5", "pm25_lag_1", "pm25_lag_2", "pm25_lag_3", "pm25_lag_7",
"pm25_roll3","pm25_roll7", "PM10", "NO2","SO2"]
WEATHER = ["temperature_mean", "temperature_min", "temperature_max",
"humidity_mean","dew_point_mean", "rain_sum", "pressure_mean",
"pressure_change", "cloud_cover_mean", "wind_speed_mean",
"wind_speed_max", "wind_u_mean", "wind_v_mean"]
CALENDAR = ["month_sin", "month_cos", "is_weekend"]
FEATURE_SETS ={
"A_pollution":POLLUTION + CALENDAR,
"B_weather": WEATHER + CALENDAR,
"C_combined": POLLUTION + WEATHER +CALENDAR,
}
def build_features(daily):
    """daily: complete daily calendar withDate, PM2.5, PM10, NO2, SO2 and weather columns.
    Every featurein row t uses onlyinformation from dayt or earlier."""
    d = daily.sort_values("Date").reset_index(drop=True).copy()
    d["Date"] =pd.to_datetime(d["Date"])
    if not (d["Date"].diff().dropna() == pd.Timedelta(days=1)).all():
        raise ValueError("Dates must be a complete daily calendar (no gaps/duplicates).")
    for c in COVARIATES: # causal, short-gap fill only
        d[c] = d[c].ffill(limit=3)
    for lag in(1, 2, 3, 7):
        d[f"pm25_lag_{lag}"] = d["PM2.5"].shift(lag)
    d["pm25_roll3"]= d["PM2.5"].rolling(3, min_periods=2).mean()
    d["pm25_roll7"]= d["PM2.5"].rolling(7, min_periods=4).mean()
    d["pressure_change"] = d["pressure_mean"].diff()
    month = d["Date"].dt.month
    d["month_sin"]= np.sin(2 * np.pi *month / 12)
    d["month_cos"]= np.cos(2 * np.pi *month / 12)
    d["is_weekend"]= (d["Date"].dt.dayofweek >= 5).astype(int)
    d["target_next_day"] = d["PM2.5"].shift(-1)
    d["target_date"]= d["Date"].shift(-1)
    return d