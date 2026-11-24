"""
src/sequence_utils.py  (v2 — leakage fixes + unified horizon protocol)

What changed vs v1, and why
---------------------------
1. ROLLING-MEAN LEAKAGE (bug, affects 02/03/04/05/06)
   v1:  df["target_roll_mean24"] = df[target].rolling(24).mean()
   A pandas rolling window ENDS at the current row, so the feature at row t
   contains y_t itself (1/24 of it, 1/168 for the weekly one). XGBoost uses
   row t directly -> contemporaneous target leakage at BOTH horizons.
   v2: every target-derived feature is shifted by the horizon's min_lag.

2. target_roll_std6 IN THE DAY-AHEAD SET (bug)
   It is .shift(1), i.e. uses y_{t-1}, but the 'target_roll' prefix filter in
   get_feature_sets let it through into the day-ahead features.
   v2: feature lists are built explicitly per horizon, never by prefix.

3. SEQUENCE WINDOWS DROPPED THE TARGET ROW (bug, affects 04/05/06)
   v1 window [i, i+L) predicted y at row i+L, so the sequence models never saw
   the target hour's own calendar/weather/holiday features or its lag24.
   Tabular models (XGBoost) did. So LSTM-vs-XGBoost was not like-for-like.
   v2: window [t-L+1, t] predicts y_t. Safe because every feature on row t is
   already lagged >= min_lag (enforced by audit_leakage below).

4. ONE SPLIT FOR EVERY MODEL
   v1 split sequence arrays by position, and sequence arrays are shorter than
   the tabular frame, so test timestamps differed between model families.
   v2 splits by TIMESTAMP (make_time_splits); every model and the TSO
   forecast are scored on exactly the same hours.

Horizons
--------
    "nowcast"   min_lag=1   (1h ahead)
    "h24"       min_lag=24  (rolling 24h ahead; what v1 called day-ahead)
    "da_strict" min_lag=36  (true day-ahead: forecasts issued ~12:00 on D-1
                              for all hours of D, so the freshest actual is
                              12-35h old; 36 covers every hour of D)
Compare against the TSO forecast with "da_strict"; that is the fair one.

Weather: actual observed weather is used as a PERFECT forecast (oracle).
Say so in the paper, or pass weather_noise_std>0 to simulate forecast error.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

HORIZONS = {"nowcast": 1, "h24": 24, "da_strict": 36}

WEATHER_COLS = ["temp", "humidity", "wind_speed", "pressure", "clouds_all"]
CALENDAR_COLS = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend", "is_holiday",
                 "heating_degree", "cooling_degree", "doy_sin", "doy_cos",
                 "temp_day_mean", "temp_day_max", "temp_day_min", "pre_holiday", "post_holiday"]

# the two near-empty columns 02/03 already dropped
_DROP_COLS = ["generation hydro pumped storage aggregated",
              "forecast wind offshore eday ahead"]


# ---------------------------------------------------------------------------
# 1. Raw load + target-independent features
# ---------------------------------------------------------------------------

def _spain_holiday_flags(times):
    import holidays
    years = range(times.dt.year.min(), times.dt.year.max() + 1)
    es = {str(d) for d in holidays.Spain(years=years)}
    return times.dt.date.astype(str).isin(es).astype(int)


def load_raw(data_dir="data", target="total load actual",
             weather_noise_std=0.0, seed=0):
    """Merge energy + city-averaged weather, add calendar/weather features.
    Nothing here depends on the target's values, so nothing here can leak."""
    energy = pd.read_csv(f"{data_dir}/energy_dataset.csv")
    weather = pd.read_csv(f"{data_dir}/weather_features.csv")

    energy["time"] = pd.to_datetime(energy["time"], utc=True)
    weather["city_name"] = weather["city_name"].str.strip()
    weather["dt_iso"] = pd.to_datetime(weather["dt_iso"], utc=True)
    weather = weather.drop_duplicates(subset=["dt_iso", "city_name"])
    weather_avg = (weather.groupby("dt_iso")[WEATHER_COLS].mean().reset_index())

    df = energy.merge(weather_avg, left_on="time", right_on="dt_iso", how="left")
    df = df.drop(columns="dt_iso")
    df["temp"] = df["temp"] - 273.15
    df = df.drop(columns=[c for c in _DROP_COLS if c in df.columns])
    df = df.dropna(subset=[target]).sort_values("time").reset_index(drop=True)

    if weather_noise_std > 0:   # simulate an imperfect day-ahead temperature forecast
        rng = np.random.default_rng(seed)
        df["temp"] = df["temp"] + rng.normal(0, weather_noise_std, len(df))

    df["hour"] = df["time"].dt.hour
    df["dayofweek"] = df["time"].dt.dayofweek
    df["month"] = df["time"].dt.month
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
    df["heating_degree"] = (18 - df["temp"]).clip(lower=0)
    df["cooling_degree"] = (df["temp"] - 18).clip(lower=0)
    df["is_holiday"] = _spain_holiday_flags(df["time"])
    doy = df["time"].dt.dayofyear
    df["doy_sin"], df["doy_cos"] = np.sin(2*np.pi*doy/365.25), np.cos(2*np.pi*doy/365.25)
    day = df["time"].dt.floor("D")
    g = df.groupby(day)["temp"]
    df["temp_day_mean"], df["temp_day_max"], df["temp_day_min"] = g.transform("mean"), g.transform("max"), g.transform("min")
    hol = df.groupby(day)["is_holiday"].max()
    df["pre_holiday"]  = day.map(hol.shift(-1, freq="D")).fillna(0).astype(int)
    df["post_holiday"] = day.map(hol.shift(1,  freq="D")).fillna(0).astype(int)
    return df


# ---------------------------------------------------------------------------
# 2. Target-derived features, parameterised by min_lag (the leakage fix)
# ---------------------------------------------------------------------------

def target_feature_names(min_lag):
    lags = sorted({min_lag} | {l for l in (1, 2, 3, 24, 48, 168) if l >= min_lag})
    s = f"_s{min_lag}"
    names = [f"target_lag{l}" for l in lags]
    names += [f"target_rollmean24{s}", f"target_rollmean168{s}",
              f"target_rollstd6{s}", f"target_diff1{s}", f"target_diff24{s}"]
    return names


def add_target_features(df, target, min_lag):
    """Every column here uses only y_{t-min_lag} and older. Shift is applied
    AFTER the rolling/diff op, which is the step v1 forgot for roll_mean."""
    y = df[target]
    s = f"_s{min_lag}"
    new = {}
    for l in sorted({min_lag} | {l for l in (1, 2, 3, 24, 48, 168) if l >= min_lag}):
        new[f"target_lag{l}"] = y.shift(l)
    new[f"target_rollmean24{s}"] = y.rolling(24).mean().shift(min_lag)
    new[f"target_rollmean168{s}"] = y.rolling(168).mean().shift(min_lag)
    new[f"target_rollstd6{s}"] = y.rolling(6).std().shift(min_lag)
    new[f"target_diff1{s}"] = y.diff(1).shift(min_lag)
    new[f"target_diff24{s}"] = y.diff(24).shift(min_lag)
    for k, v in new.items():
        df[k] = v
    return df


def add_generation_lags(df, min_lag):
    gen_lag = max(24, min_lag)   # never same-hour generation (v1's rule), never fresher than min_lag
    cols = [c for c in df.columns if c.startswith("generation") and "_lag" not in c]
    names = []
    for c in cols:
        name = f"{c}_lag{gen_lag}"
        df[name] = df[c].shift(gen_lag)
        names.append(name)
    return df, names


def build_dataset(data_dir="data", target="total load actual", horizon="da_strict",
                  weather_noise_std=0.0, include_generation=True):
    """One call per (target, horizon). Returns (df_model, feature_names)."""
    min_lag = HORIZONS[horizon]
    df = load_raw(data_dir, target, weather_noise_std=weather_noise_std)
    df = add_target_features(df, target, min_lag)
    feats = WEATHER_COLS + CALENDAR_COLS + target_feature_names(min_lag)
    if include_generation:
        df, gen = add_generation_lags(df, min_lag)
        feats += gen
    df = df.dropna(subset=feats + [target]).reset_index(drop=True)
    return df, feats


# ---------------------------------------------------------------------------
# 3. Empirical leakage audit — run this once per horizon, keep it in the notebook
# ---------------------------------------------------------------------------

def audit_leakage(df_raw, target, min_lag, features, probe_rows=(5000, 12000, 20000)):
    """
    Perturbs the target at row t0 and rebuilds every target-derived feature.
    Any feature that changes at a row t < t0 + min_lag has access to y
    values that would not be known at forecast time -> leakage.
    Tests the code, not our reading of it. Returns list of offending features.
    """
    offenders = set()
    tfeats = [f for f in features if f.startswith("target_")]
    base = add_target_features(df_raw[[target]].copy(), target, min_lag)
    for t0 in probe_rows:
        if t0 >= len(df_raw):
            continue
        pert = df_raw[[target]].copy()
        pert.iloc[t0, 0] += 1e4
        pert = add_target_features(pert, target, min_lag)
        window = slice(max(0, t0 - 200), t0 + min_lag)   # rows that must NOT change
        for f in tfeats:
            a, b = base[f].iloc[window].to_numpy(), pert[f].iloc[window].to_numpy()
            if not np.allclose(a, b, equal_nan=True):
                offenders.add(f)
    return sorted(offenders)


# ---------------------------------------------------------------------------
# 4. Shared chronological split by TIMESTAMP
# ---------------------------------------------------------------------------

def make_time_splits(times, val_frac=0.16, test_frac=0.20):
    """Returns (val_start, test_start) timestamps. 0.16 val of the whole
    series ~= v1's 20% of the 80% train block, so numbers stay comparable."""
    times = pd.Series(pd.to_datetime(times, utc=True)).sort_values().reset_index(drop=True)
    n = len(times)
    test_start = times.iloc[int(n * (1 - test_frac))]
    val_start = times.iloc[int(n * (1 - test_frac - val_frac))]
    return val_start, test_start


def split_mask(times, val_start, test_start):
    t = pd.to_datetime(pd.Series(times), utc=True)
    return {"train": (t < val_start).to_numpy(),
            "val": ((t >= val_start) & (t < test_start)).to_numpy(),
            "test": (t >= test_start).to_numpy()}


# ---------------------------------------------------------------------------
# 5. Sequence windowing — now includes the target row
# ---------------------------------------------------------------------------

def make_sequences(X, y, seq_len=24, times=None):
    """
    Window k covers rows [k, k+seq_len) and predicts y at row k+seq_len-1,
    i.e. the LAST row of the window is the target hour (v1 predicted the
    row after the window). Returns X_seq, y_seq, and the target timestamps
    if `times` is given, so sequence and tabular models can be aligned.
    """
    X_arr = np.asarray(X, dtype=np.float32)
    y_arr = np.asarray(y, dtype=np.float32)
    n_windows = len(X_arr) - seq_len + 1
    if n_windows <= 0:
        raise ValueError(f"seq_len={seq_len} exceeds number of rows ({len(X_arr)})")
    idx = np.arange(seq_len)[None, :] + np.arange(n_windows)[:, None]
    X_seq = X_arr[idx]                        # (n_windows, seq_len, n_features)
    y_seq = y_arr[seq_len - 1:]
    if times is not None:
        t_seq = np.asarray(times)[seq_len - 1:]
        return X_seq, y_seq, t_seq
    return X_seq, y_seq


class SequenceDataset(Dataset):
    """Unchanged from v1: standardise with TRAIN statistics only."""

    def __init__(self, X_seq, y_seq, feature_mean=None, feature_std=None,
                 target_mean=None, target_std=None):
        if feature_mean is None:
            flat = X_seq.reshape(-1, X_seq.shape[-1])
            feature_mean, feature_std = flat.mean(axis=0), flat.std(axis=0) + 1e-8
        if target_mean is None:
            target_mean, target_std = y_seq.mean(), y_seq.std() + 1e-8
        self.feature_mean = feature_mean
        self.feature_std = np.where(np.asarray(feature_std) < 1e-8, 1.0, feature_std)
        self.target_mean, self.target_std = target_mean, target_std
        self.X = torch.from_numpy(((X_seq - feature_mean) / self.feature_std).astype(np.float32))
        self.y = torch.from_numpy(((y_seq - target_mean) / target_std).astype(np.float32))
        self.y_raw = torch.from_numpy(y_seq.astype(np.float32))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def denormalize(self, y_norm_tensor):
        return y_norm_tensor * self.target_std + self.target_mean


# ---------------------------------------------------------------------------
# 6. Back-compat shims so old notebooks still run (now leakage-free)
# ---------------------------------------------------------------------------

def load_and_engineer_features(data_dir="data", target="total load actual"):
    """v1 signature. Builds nowcast AND h24 features on one frame."""
    df = load_raw(data_dir, target)
    df = add_target_features(df, target, 1)
    df = add_target_features(df, target, 24)
    df, gen = add_generation_lags(df, 24)
    return df, gen


def get_feature_sets(df, generation_lag_cols):
    base = WEATHER_COLS + CALENDAR_COLS
    return (base + target_feature_names(1) + generation_lag_cols,
            base + target_feature_names(24) + generation_lag_cols)


def time_series_train_test_split(X_seq, y_seq, test_frac=0.2):
    """Kept for back-compat only. Prefer make_time_splits + split_mask."""
    split_idx = int(len(X_seq) * (1 - test_frac))
    return X_seq[:split_idx], X_seq[split_idx:], y_seq[:split_idx], y_seq[split_idx:], split_idx
