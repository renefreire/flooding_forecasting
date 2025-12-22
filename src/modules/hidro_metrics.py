import numpy as np

def rmse(y_true, y_pred):
    return np.sqrt(np.nanmean((y_true - y_pred) ** 2))

def mae(y_true, y_pred):
    return np.nanmean(np.abs(y_true - y_pred))

def nse(y_true, y_pred):
    # Nash-Sutcliffe Efficiency
    denom = np.nanmean((y_true - np.nanmean(y_true))**2)
    if denom == 0:
        return np.nan
    return 1.0 - (np.nanmean((y_true - y_pred)**2) / denom)

def kge(y_true, y_pred):
    # Kling-Gupta Efficiency (original formulation)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    # remove nan aligned
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if mask.sum() == 0:
        return np.nan
    y_true = y_true[mask]; y_pred = y_pred[mask]
    r = np.corrcoef(y_true, y_pred)[0,1]
    if np.isnan(r):
        return np.nan
    alpha = np.std(y_pred)/np.std(y_true) if np.std(y_true) != 0 else np.nan
    beta = np.mean(y_pred)/np.mean(y_true) if np.mean(y_true) != 0 else np.nan
    if np.isnan(alpha) or np.isnan(beta):
        return np.nan
    return 1 - np.sqrt((r-1)**2 + (alpha-1)**2 + (beta-1)**2)