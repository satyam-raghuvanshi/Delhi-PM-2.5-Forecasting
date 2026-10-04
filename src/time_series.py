import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
def walk_forward_arima(series, order, start, end, refit_every=30, log=False):
    """Honest one-step-ahead ARIMA forecastsfor every date in [start, end].
    series : regular daily Series (NaN allowed) containing ALLhistory up to ‘end‘.
    Pass the series truncated at ‘end‘so later data cannever be touched.
    Returns a DataFrame indexed by thedate being PREDICTED.
    """
    y = series.asfreq("D").astype(float)
    ym = np.log1p(y) if log else y
    eval_dates =ym.loc[start:end].index
    preds = []
    for i in range(0, len(eval_dates), refit_every):
        block = eval_dates[i:i + refit_every]
        history = ym.loc[: block[0]- pd.Timedelta(days=1)] # strictly earlier
        res = ARIMA(history, order=order).fit()
        applied = res.apply(ym.loc[: block[-1]],refit=False) # fixed params
        pred = applied.get_prediction(start=block[0],end=block[-1],
        dynamic=False).predicted_mean
        preds.append(pred)
    pred = pd.concat(preds)
    if log:
        pred = np.expm1(pred) # note: back-transformis biased low (median-type)
    return pd.DataFrame({"Actual": y.loc[start:end], "Predicted": pred})