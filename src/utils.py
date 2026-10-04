import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
def month_to_season(m):
    """Winter Dec-Feb, Summer Mar-May, Monsoon Jun-Sep, Post-Monsoon Oct-Nov."""
    if m in (12, 1, 2):
        return "Winter"
    if m in (3, 4, 5):
        return "Summer"
    if m in (6, 7, 8, 9):
        return "Monsoon"
    return "Post-Monsoon"
def evaluate(actual, pred):
    """MAE, RMSE, R2 on rows where BOTH actual and pred are available."""
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    ok = ~(np.isnan(actual) | np.isnan(pred))
    a, p = actual[ok], pred[ok]
    return {
        "MAE": mean_absolute_error(a, p),
        "RMSE": float(np.sqrt(mean_squared_error(a, p))),
        "R2": r2_score(a, p),
        "n": int(ok.sum()),
}