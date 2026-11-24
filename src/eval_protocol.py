"""
src/eval_protocol.py

Makes the accuracy table defensible:
  * every model + the TSO forecast scored on the SAME timestamps,
  * mean ± std over seeds instead of one run,
  * Diebold-Mariano test (with Harvey-Leybourne-Newbold small-sample
    correction) so "model A beats model B" is a tested claim.
"""

import numpy as np
import pandas as pd
from scipy import stats


def align_on_time(pred_dict, y_true, times_dict, tso=None, tso_times=None):
    """
    pred_dict  : {model_name: (n_i,) predictions}
    times_dict : {model_name: (n_i,) timestamps of those predictions}
    y_true     : pd.Series of actuals indexed by timestamp
    tso        : optional pd.Series of the TSO forecast indexed by timestamp
    Returns a DataFrame indexed by the timestamps ALL models share.
    """
    frames = {name: pd.Series(np.asarray(p), index=pd.to_datetime(times_dict[name], utc=True))
              for name, p in pred_dict.items()}
    df = pd.DataFrame(frames)
    if tso is not None:
        df["TSO"] = tso.reindex(df.index)
    df["y"] = y_true.reindex(df.index)
    df = df.dropna()
    return df


def scores(df_aligned):
    """MAE / RMSE / MAPE for every column vs 'y', plus skill vs TSO if present."""
    rows = []
    y = df_aligned["y"].to_numpy()
    for c in df_aligned.columns.drop("y"):
        e = df_aligned[c].to_numpy() - y
        rows.append({"model": c, "MAE": np.abs(e).mean(), "RMSE": np.sqrt((e ** 2).mean()),
                     "MAPE_%": 100 * np.abs(e / y).mean()})
    out = pd.DataFrame(rows).set_index("model")
    if "TSO" in out.index:
        out["skill_vs_TSO_%"] = 100 * (1 - out["MAE"] / out.loc["TSO", "MAE"])
    return out


def seed_summary(per_seed_tables):
    """per_seed_tables: list of scores() outputs (one per seed) -> mean ± std table."""
    stacked = pd.concat(per_seed_tables, keys=range(len(per_seed_tables)))
    g = stacked.groupby(level=1)
    mean, std = g.mean(), g.std(ddof=1)
    out = mean.copy()
    for c in ["MAE", "RMSE"]:
        out[f"{c}_std"] = std[c]
    out["n_seeds"] = len(per_seed_tables)
    return out.sort_values("MAE")


def diebold_mariano(e1, e2, h=1, power=1):
    """
    H0: equal predictive accuracy of two forecasts with errors e1, e2.
    power=1 -> absolute loss (matches MAE), 2 -> squared loss.
    h = forecast horizon in steps; the HAC variance uses h-1 autocovariances.
    For hourly day-ahead use h=24 (errors within a day are correlated).
    Returns (DM statistic, two-sided p-value). Negative DM -> model 1 better.
    """
    e1, e2 = np.asarray(e1), np.asarray(e2)
    d = np.abs(e1) ** power - np.abs(e2) ** power
    n = len(d)
    d_mean = d.mean()
    gamma = [np.mean((d[k:] - d_mean) * (d[:n - k] - d_mean)) for k in range(h)]
    var = (gamma[0] + 2 * sum(gamma[1:])) / n
    if var <= 0:
        return np.nan, np.nan
    dm = d_mean / np.sqrt(var)
    hln = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)   # Harvey et al. 1997
    dm *= hln
    p = 2 * stats.t.sf(np.abs(dm), df=n - 1)
    return dm, p


def dm_matrix(df_aligned, h=24, power=1):
    """Pairwise DM p-values; cell (A, B) < 0.05 and DM<0 means A significantly beats B."""
    models = [c for c in df_aligned.columns if c != "y"]
    y = df_aligned["y"].to_numpy()
    stat = pd.DataFrame(index=models, columns=models, dtype=float)
    pval = stat.copy()
    for a in models:
        for b in models:
            if a == b:
                continue
            s, p = diebold_mariano(df_aligned[a].to_numpy() - y, df_aligned[b].to_numpy() - y, h, power)
            stat.loc[a, b], pval.loc[a, b] = s, p
    return stat, pval
