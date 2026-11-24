#!/usr/bin/env python3
"""Agreement Is Not Correctness: every experiment of the paper in one script, for a terminal or a virtual machine.

  python agreement_is_not_correctness.py                              # the runs of the paper, deep plan (long; safe to interrupt and restart)
  python agreement_is_not_correctness.py --plan smoke                 # every part with tiny settings: checks the whole script
  python agreement_is_not_correctness.py --plan deep --parts 1,5      # only some parts
  python agreement_is_not_correctness.py --plan deep --parts 1 --stages bench,stats --models TFT,XGBoost
  python agreement_is_not_correctness.py --list                       # what each part is

Keep it running after you log out of the machine:

  nohup python agreement_is_not_correctness.py > run.log 2>&1 &
  tail -f run.log                                                     # or: tail -f results/logs/part_1_deep.log

Parts
  1  Main pipeline: search, real data, injected effects, synthetic benchmark, calibrated benchmark, shortcut, statistics, figures
  2  Real-data checks: size check, natural experiments, signatures, benchmark sensitivity
  3  Intrinsic and post-hoc explanations of the same RETAIN model, one change at a time (uses src/)
  4  Renewable sources under a hidden demand driver
  5  Replication on Portugal (downloads its data on the first run)
  6  Combining SHAP over the window and over forecasts, LIME, quality indicators, correlated inputs
  7  Causal reference (needs tigramite)
  8  Shortcut, paired with the grouped Shapley value and the causal estimate

How it works. Each part is a block of plain code further down in this file, between "# ==== PART k" lines. The launcher at the
top starts one Python process per part, one after another, and runs that part's block in a fresh namespace, so the parts cannot
interfere and the memory of one part is released before the next starts. If a part fails, the others still run, and the summary
at the end says which ones failed. Every finished run is saved under results/, so starting the same command again continues
where it stopped. A GPU is used automatically when PyTorch finds one. Figures are saved as files (nothing is shown on screen).

Needs: requirements.txt installed, the Spanish data in data/ (see data/README.md), and the src/ folder next to this file.
"""
import argparse, builtins, json, os, re, subprocess, sys, time
from pathlib import Path

PARTS = {1: "main pipeline", 2: "real-data checks", 3: "RETAIN, intrinsic against post-hoc", 4: "renewable sources", 5: "Portugal",
         6: "combining SHAP, LIME, indicators", 7: "causal reference", 8: "shortcut, paired"}
STAGES = "diag,tune,real,inject,bench,calib,shortcut,stats"


def _settings(a):
    return dict(plan=a.plan, seeds=a.seeds, max_epochs=a.max_epochs, patience=a.patience, n_explain=a.n_explain,
                tune_trials=a.tune_trials, tune_epochs=a.tune_epochs, models=a.models, stages=a.stages.split(","))


def _run_part(k, settings):
    """Run the code block of part k of this file in a fresh namespace (line numbers in error messages match this file)."""
    lines = Path(__file__).read_text(encoding="utf-8").split("\n")
    marks = [i for i, l in enumerate(lines) if re.match(r"# ==== PART \d+", l)]
    start = next(i for i in marks if lines[i].startswith(f"# ==== PART {k}:"))
    end = next((i for i in marks if i > start), len(lines))
    os.environ.setdefault("MPLBACKEND", "Agg")                         # figures are saved, not shown
    if not hasattr(builtins, "display"): builtins.display = lambda *x, **kw: print(*x)
    source = "\n" * start + "\n".join(lines[start:end])
    exec(compile(source, __file__, "exec"), {"__name__": "__main__", "__file__": __file__, "GLOBAL_SETTINGS": settings})


def _main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", default="deep", choices=["smoke", "deep"], help="deep (default): the runs of the paper. smoke: tiny settings, to check the script")
    ap.add_argument("--parts", default="1,2,3,4,5,6,7,8", help="comma-separated part numbers")
    ap.add_argument("--seeds", type=int, default=10, help="deep plan: seeds per configuration")
    ap.add_argument("--max-epochs", type=int, default=150, help="deep plan: longest training of the neural models")
    ap.add_argument("--patience", type=int, default=15, help="deep plan: early-stopping patience")
    ap.add_argument("--n-explain", type=int, default=1000, help="deep plan: explained test hours per run")
    ap.add_argument("--tune-trials", type=int, default=60, help="deep plan: hyperparameter-search trials per model")
    ap.add_argument("--tune-epochs", type=int, default=40, help="deep plan: longest training of one search trial")
    ap.add_argument("--models", default="", help="Part 1 only: subset of Ridge,XGBoost,LSTM,Transformer,TFT,RETAIN")
    ap.add_argument("--stages", default=STAGES, help="Part 1 only: subset of " + STAGES)
    ap.add_argument("--list", action="store_true", help="list the parts and exit")
    ap.add_argument("--only", type=int, default=0, help=argparse.SUPPRESS)   # used by the launcher: run one part in this process
    a = ap.parse_args()
    if a.list:
        for k, t in PARTS.items(): print(f"  {k}  {t}")
        return 0
    root = Path(__file__).resolve().parent; os.chdir(root)
    if str(root / "src") not in sys.path: sys.path.insert(0, str(root / "src"))
    settings = _settings(a)
    if a.only:
        print(f"Part {a.only}: {PARTS[a.only]} | {settings}", flush=True); _run_part(a.only, settings); return 0
    assert (root / "src").is_dir() and (root / "data").is_dir(), "keep this file next to the src/ and data/ folders"
    import importlib.util
    need = {"numpy": "all", "pandas": "all", "scipy": "all", "sklearn": "all", "matplotlib": "all", "torch": "all but 7",
            "xgboost": "1-6", "shap": "1-6, 8", "holidays": "1-6", "tigramite": "7"}
    missing = [m for m in need if importlib.util.find_spec(m) is None]
    if missing:
        print("Missing libraries: " + ", ".join(f"{m} (parts {need[m]})" for m in missing))
        print("Install them first:  python -m pip install --no-cache-dir -r requirements.txt")
        print(f"Check that this is the Python of your environment: {sys.executable}")
        return 1
    if importlib.util.find_spec("optuna") is None: print("optuna not installed: Part 1 uses a random search instead.")
    parts = [int(p) for p in a.parts.split(",") if p.strip()]; assert all(p in PARTS for p in parts), f"parts are {list(PARTS)}"
    (root / "results" / "logs").mkdir(parents=True, exist_ok=True)
    (root / "results" / "settings.json").write_text(json.dumps(settings, indent=2))
    passthrough = [x for x in sys.argv[1:]]; status = {}; t_all = time.time()
    for k in parts:
        log = root / "results" / "logs" / f"part_{k}_{a.plan}.log"; t0 = time.time()
        print(f"\n{'=' * 100}\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Part {k}: {PARTS[k]}   (log: {log.relative_to(root)})\n{'=' * 100}", flush=True)
        with open(log, "a", encoding="utf-8") as f:
            f.write(f"\n===== started {time.strftime('%Y-%m-%d %H:%M:%S')} | {' '.join(sys.argv)} =====\n"); f.flush()
            p = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()), *passthrough, "--only", str(k)], stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1)
            for line in p.stdout:                                     # show on the terminal and keep in the part's log
                sys.stdout.write(line); sys.stdout.flush(); f.write(line); f.flush()
            code = p.wait()
        status[k] = (code, time.time() - t0)
        print(f"[{time.strftime('%H:%M:%S')}] Part {k} {'finished' if code == 0 else f'FAILED (exit code {code})'} after {(time.time() - t0) / 60:.1f} min", flush=True)
    print(f"\n{'=' * 100}\nSummary ({a.plan} plan, {(time.time() - t_all) / 3600:.2f} h)")
    for k, (code, sec) in status.items():
        print(f"  Part {k}  {'ok    ' if code == 0 else 'FAILED'}  {sec / 60:8.1f} min  {PARTS[k]}")
    failed = [k for k, (code, _) in status.items() if code]
    if failed: print(f"Failed parts: {failed}. See results/logs/. Fix the cause and run the same command again: finished runs are skipped.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_main())

# Nothing below this line runs when the file is started: the launcher above has already exited.
# Each block is the code of one part, run by _run_part() in its own process and namespace.



# ======================================================================================================================
# ==== PART 1: Part 1. Main pipeline
# ======================================================================================================================

# %% [markdown]
# # Part 1. Main pipeline
#
# One protocol for everything in this part. A single forecast is issued at 12:00 on day D-1 for the 24 hours of day D, with demand and generation known up to 11:00. The synthetic benchmark is built with the same operational features as the real data, so benchmark and real-data results are directly comparable.
#
# | Stage | What it does | Tables written to `results/deep_pipeline/<plan>/tables/` |
# |---|---|---|
# | `diag` | Sanity check of the demand series: naive forecasts, autocorrelation, operator forecast, leakage audit | `D1` to `D3` (diagnostics) |
# | `tune` | Hyperparameter search on the real validation period (Optuna TPE with pruning, or a random search if Optuna is missing) | `best_params.json` |
# | `real` | Train, forecast and explain every model on real data | `A1` accuracy, `A2` Diebold-Mariano, `G1` signatures |
# | `inject` | Same on real demand with injected effects of known size (event, driver, decoy) | `D1_injected_truth`, `D2`, `D3` |
# | `bench` | Same on the synthetic benchmark, all seven versions, with the quality indicators and the altered copies of TreeSHAP | `B1`, `B2`, `C1` to `C3`, `F1`, `F2` |
# | `calib` | Same on a benchmark whose components are fitted to the real training data | `B1_correctness_calib`, `B2_paired_calib` |
# | `shortcut` | TFT, XGBoost and Ridge with and without demand history, on the benchmark and on the injected real data | `E1_shortcut_*` |
# | `stats` | Means, standard deviations, bootstrap 95% intervals, paired Wilcoxon tests, Diebold-Mariano tests | all of the above, and `RESULTS.md` |
#
# Models: Ridge, XGBoost, LSTM, Transformer, TFT, RETAIN. Explanations: exact linear contributions (Ridge), TreeSHAP path-dependent and interventional (XGBoost), SHAP by expected gradients and Integrated Gradients (neural models), the exact RETAIN contribution, and the TFT selection weights.
#
# Nothing in this part selects seeds, runs or settings by test-set results.
#
# **Settings.** The plan (`smoke` or `deep`), the number of seeds and the training length come from the command line (`--plan`, `--seeds`, `--max-epochs`, ...). Everything else is set in the first lines of this part. Finished runs are saved and skipped when the part is run again.

# %%
# The launcher passes the shared settings as GLOBAL_SETTINGS (plan, seeds, training length, explained hours, search size).
import json
from pathlib import Path
print(GLOBAL_SETTINGS)

# %% [markdown]
# ## 1.1 Imports and constants

# %%
import itertools, json, math, re, sys, time, warnings
from pathlib import Path
from types import SimpleNamespace
import numpy as np, pandas as pd
from scipy import stats
from sklearn.linear_model import RidgeCV
warnings.filterwarnings("ignore")

TZ = "Europe/Madrid"
ISSUE_HOUR, LAST_KNOWN_HOUR = 12, 11          # forecast issued at 12:00 on D-1; last observation 11:00 on D-1
SPLIT = (0.64, 0.16, 0.20)
WMAX = 48                                      # longest input window any model may use (same targets for all models)
ALL_MODELS = ["Ridge", "XGBoost", "LSTM", "Transformer", "TFT", "RETAIN"]
SEQ_MODELS = ["LSTM", "Transformer", "TFT", "RETAIN"]

PLANS = {
    "smoke": dict(seeds=1, n_explain=60, shap_bg=30, shap_n=20, ig_steps=4, tune_trials=2, tune_epochs=2, tune_patience=1,
                  max_epochs=2, patience=1, xgb_trees=150, n_ind=30, n_subsets=10, bench_conds=["default"], calib_seeds=1,
                  windows=[24]),
    "deep":  dict(seeds=10, n_explain=1000, shap_bg=300, shap_n=200, ig_steps=32, tune_trials=60, tune_epochs=40, tune_patience=6,
                  max_epochs=150, patience=15, xgb_trees=6000, n_ind=200, n_subsets=50, bench_conds=None, calib_seeds=5,
                  windows=[24, 48]),
}
INJECT = dict(seed=2026, n_event_days=60, event_mw=-1500, event_hours=(8, 21), driver_mw=900,
              driver_temp_interaction=0.4, decoy_corr=0.8)

# %% [markdown]
# ## 1.2 Synthetic benchmark with known drivers

# %%
B_START, B_END, LAG_L = "2015-01-01 00:00", "2018-12-31 23:00", 36
CONDITIONS = [
    dict(name="default",       change="2016-07-01", latent=1200, interaction=0.0),
    dict(name="extrapolation", change="2017-07-01", latent=1200, interaction=0.0),
    dict(name="stationary",    change=None,         latent=1200, interaction=0.0),
    dict(name="latent_0",      change="2016-07-01", latent=0,    interaction=0.0),
    dict(name="latent_600",    change="2016-07-01", latent=600,  interaction=0.0),
    dict(name="latent_2400",   change="2016-07-01", latent=2400, interaction=0.0),
    dict(name="interaction",   change="2016-07-01", latent=1200, interaction=0.5),
]
GEN_DEFAULT = dict(base=28000, hour_amp=(2600, 900), hour_phase=(9, 3), sat=-1800, sun=-3200, holiday=-2500,
                   heat_thr=16, heat_slope=250, cool_thr=22, cool_coef=150, cool_exp=1.3, cool_factor_after=1.6,
                   latent_phi=0.995, ar_phi=0.95, ar_sd=500, noise_sd=150, operator_sd=200, gas_share=0.55,
                   gas_latent=400, gas_noise_sd=300)
CITIES = {"Madrid": 0.0, "Barcelona": 1.5, "Valencia": 2.5, "Seville": 4.0, "Bilbao": -1.5}


def spanish_holidays(years):
    try:
        import holidays as _h
        return set(_h.Spain(years=years).keys())
    except ImportError:
        fixed = [(1, 1), (1, 6), (5, 1), (8, 15), (10, 12), (11, 1), (12, 6), (12, 8), (12, 25)]
        gf = {2015: (4, 3), 2016: (3, 25), 2017: (4, 14), 2018: (3, 30), 2019: (4, 19)}
        out = {pd.Timestamp(y, m, d).date() for y in years for m, d in fixed}
        return out | {pd.Timestamp(y, *gf[y]).date() for y in years if y in gf}


def _ar(n, phi, sd, rng):
    e = rng.normal(0, sd * np.sqrt(1 - phi ** 2), n); x = np.empty(n); x[0] = rng.normal(0, sd)
    for i in range(1, n): x[i] = phi * x[i - 1] + e[i]
    return x


def generate(cond, seed, g=GEN_DEFAULT):
    """Hourly demand D = base + C_hour + C_week + C_holiday + C_temp (+ C_interaction) + C_latent + C_AR + eps."""
    rng = np.random.default_rng(1000 * seed + 17)
    idx = pd.date_range(pd.Timestamp(B_START, tz=TZ).tz_convert("UTC"), pd.Timestamp(B_END, tz=TZ).tz_convert("UTC"), freq="h")
    loc = idx.tz_convert(TZ); n = len(idx)
    h, dow, doy = loc.hour.values, loc.dayofweek.values, loc.dayofyear.values
    day = loc.tz_localize(None).normalize()
    hols = spanish_holidays(range(loc.year.min(), loc.year.max() + 1))
    is_hol = np.array([d.date() in hols for d in day])
    anom = _ar(n, 0.995, 2.5, rng)
    temps = {c: 15 + off - 9 * np.cos(2 * np.pi * (doy - 20) / 365.25) + (5 + 0.3 * off) * np.sin(2 * np.pi * (h - 9) / 24)
                + anom + _ar(n, 0.9, 0.8, rng) for c, off in CITIES.items()}
    T = np.mean(list(temps.values()), axis=0)
    clouds = 100 / (1 + np.exp(-_ar(n, 0.98, 1.5, rng)))
    humidity = np.clip(60 + 0.3 * (clouds - 50) - 1.2 * (T - 15) + rng.normal(0, 4, n), 10, 100)
    pressure = 1015 + _ar(n, 0.99, 6, rng)
    wind_speed = np.abs(4 + _ar(n, 0.97, 2.5, rng))
    C_hour = g["hour_amp"][0] * np.sin(2 * np.pi * (h - g["hour_phase"][0]) / 24) + \
             g["hour_amp"][1] * np.sin(4 * np.pi * (h - g["hour_phase"][1]) / 24)
    C_week = np.where(dow == 5, g["sat"], np.where(dow == 6, g["sun"], 0)).astype(float)
    C_hol = np.where(is_hol, g["holiday"], 0).astype(float)
    k = np.ones(n)
    if cond["change"]:
        k[loc.tz_localize(None) >= pd.Timestamp(cond["change"])] = g["cool_factor_after"]
    heat = g["heat_slope"] * np.clip(g["heat_thr"] - T, 0, None)
    cool = k * g["cool_coef"] * np.clip(T - g["cool_thr"], 0, None) ** g["cool_exp"]
    C_temp = heat + cool
    C_int = cond["interaction"] * cool * np.sin(2 * np.pi * (h - 10) / 24)
    days_u, day_pos = np.unique(day.values, return_inverse=True)
    z = _ar(len(days_u), g["latent_phi"], 1.0, rng)
    z_h = np.interp(day_pos + h / 24, np.arange(len(days_u)), z)
    C_lat = cond["latent"] * z_h
    C_AR = _ar(n, g["ar_phi"], g["ar_sd"], rng)
    eps = rng.normal(0, g["noise_sd"], n)
    D = g["base"] + C_hour + C_week + C_hol + C_temp + C_int + C_lat + C_AR + eps
    season = 0.6 + 0.4 * np.sin(2 * np.pi * (doy - 80) / 365.25)
    solar = 7000 * np.clip(np.sin(np.pi * (h - 7) / 13), 0, None) * season * (1 - 0.6 * clouds / 100)
    wind = np.minimum(15000, 60 * wind_speed ** 3)
    nuclear = np.full(n, 7000.0)
    for s in rng.choice(np.arange(0, n - 24 * 30), 3, replace=False):
        nuclear[s:s + 24 * 30] -= 1000
    hydro = 3000 + 1500 * np.sin(2 * np.pi * (doy - 60) / 365.25) + rng.normal(0, 150, n)
    resid = D - solar - wind - nuclear - hydro
    gas = g["gas_share"] * resid + g["gas_latent"] * z_h * (cond["latent"] > 0) + _ar(n, 0.9, g.get("gas_noise_sd", 0), rng)
    coal = np.clip(resid - gas, 0, None)
    ar_known = np.r_[np.full(LAG_L, 0.0), C_AR[:-LAG_L]] * g["ar_phi"] ** LAG_L
    operator = D - eps - C_AR + ar_known + rng.normal(0, g["operator_sd"], n)
    energy = pd.DataFrame({
        "generation solar": solar, "generation wind onshore": wind, "generation nuclear": nuclear,
        "generation hydro water reservoir": hydro, "generation fossil gas": gas, "generation fossil hard coal": coal,
        "forecast solar day ahead": solar + rng.normal(0, 200, n), "forecast wind onshore day ahead": wind + rng.normal(0, 800, n),
        "total load forecast": operator, "total load actual": D}, index=idx)
    weather = pd.DataFrame(dict(temp=T, humidity=humidity, pressure=pressure, wind_speed=wind_speed, clouds_all=clouds), index=idx)
    truth = pd.DataFrame(dict(C_hour=C_hour, C_week=C_week, C_holiday=C_hol, C_temp=C_temp + C_int, C_latent=C_lat, C_AR=C_AR), index=idx)
    return energy, weather, truth, hols, temps


def write_kaggle(folder, seed=0):
    """Write one benchmark realization in the Kaggle file format (dry runs without the real data)."""
    energy, weather, _, _, temps = generate(CONDITIONS[0], seed)
    folder.mkdir(parents=True, exist_ok=True)
    t = energy.index.tz_convert(TZ).strftime("%Y-%m-%d %H:%M:%S%z"); t = [s[:-2] + ":" + s[-2:] for s in t]
    e = energy.copy(); e.insert(0, "time", t); e.to_csv(folder / "energy_dataset.csv", index=False)
    rows = [pd.DataFrame({"dt_iso": t, "city_name": c, "temp": v + 273.15, "humidity": weather["humidity"].values,
                          "pressure": weather["pressure"].values, "wind_speed": weather["wind_speed"].values,
                          "clouds_all": weather["clouds_all"].values}) for c, v in temps.items()]
    pd.concat(rows).to_csv(folder / "weather_features.csv", index=False)


def calibrate_generator(world):
    """Fit the benchmark's components to the real TRAINING data (least squares), so that the benchmark's daily
    profile, weekend and holiday drops and temperature response are taken from data and not chosen by hand."""
    loc = world.IDX.tz_convert(TZ); day = loc.tz_localize(None).normalize()
    n_tr = int(SPLIT[0] * len(world.IDX)); sl = slice(0, n_tr)
    y = world.LOAD.values[sl]; T = world.W["temp"].values[sl]; h = loc.hour.values[sl]; dow = loc.dayofweek.values[sl]
    hol = np.array([d.date() in world.HOLS for d in day[sl]], float)
    g = dict(GEN_DEFAULT)
    Z = np.column_stack([np.ones(len(y)), np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24), np.sin(4 * np.pi * h / 24),
                         np.cos(4 * np.pi * h / 24), dow == 5, dow == 6, hol, np.clip(g["heat_thr"] - T, 0, None),
                         np.clip(T - g["cool_thr"], 0, None) ** g["cool_exp"]]).astype(float)
    ok = ~np.isnan(Z).any(1) & ~np.isnan(y)
    c = np.linalg.lstsq(Z[ok], y[ok], rcond=None)[0]
    def amp_phase(a, b, harm):                       # a sin(w h) + b cos(w h) = A sin(w (h - p))
        A = float(np.hypot(a, b)); p = float((-np.arctan2(b, a) / (harm * 2 * np.pi / 24)) % (24 / harm)); return A, p
    A1, p1 = amp_phase(c[1], c[2], 1); A2, p2 = amp_phase(c[3], c[4], 2)
    res = y[ok] - Z[ok] @ c
    g.update(base=float(c[0]), hour_amp=(A1, A2), hour_phase=(p1, p2), sat=float(c[5]), sun=float(c[6]), holiday=float(c[7]),
             heat_slope=float(max(c[8], 0)), cool_coef=float(max(c[9], 0)))
    g["_fit"] = dict(residual_sd=float(np.std(res)), n=int(ok.sum()))
    return g

# %% [markdown]
# ## 1.3 Data and operational features

# %%
GENMAP = {"gas": "generation fossil gas", "coal": "generation fossil hard coal", "nuclear": "generation nuclear",
          "hydro": "generation hydro water reservoir", "solar": "generation solar", "wind": "generation wind onshore"}


def make_world(E, W, hols=None, operator_col="total load forecast"):
    IDX = pd.date_range(E.index.min(), E.index.max(), freq="h", tz="UTC")
    E, W = E.reindex(IDX), W.reindex(IDX).interpolate(limit=6, limit_area="inside")
    LOAD = E["total load actual"].interpolate(limit=6, limit_area="inside")
    hols = hols or spanish_holidays(range(IDX.min().year, IDX.max().year + 1))
    return SimpleNamespace(IDX=IDX, E=E, W=W, LOAD=LOAD, HOLS=hols, GEN={k: v for k, v in GENMAP.items() if v in E},
                           OPERATOR=E[operator_col] if operator_col in E else None, OBSERVED=E["total load actual"].notna().values)


def load_real(data_dir):
    e = pd.read_csv(Path(data_dir) / "energy_dataset.csv"); e["time"] = pd.to_datetime(e["time"], utc=True)
    e = e.drop_duplicates("time").set_index("time").sort_index()
    w = pd.read_csv(Path(data_dir) / "weather_features.csv"); w["city_name"] = w["city_name"].astype(str).str.strip()
    w["time"] = pd.to_datetime(w["dt_iso"], utc=True); w = w.drop_duplicates(["time", "city_name"])
    w["temp"] = w["temp"] - 273.15
    if "pressure" in w: w.loc[(w["pressure"] < 900) | (w["pressure"] > 1100), "pressure"] = np.nan   # known outliers
    if "wind_speed" in w: w.loc[w["wind_speed"] > 50, "wind_speed"] = np.nan
    cols = [c for c in ["temp", "humidity", "pressure", "wind_speed", "clouds_all"] if c in w]
    return make_world(e, w.groupby("time")[cols].mean().sort_index())


def to_daily(s):
    loc = s.index.tz_convert(TZ)
    df = pd.DataFrame({"v": s.values, "day": loc.tz_localize(None).normalize(), "hour": loc.hour})
    M = df.groupby(["day", "hour"])["v"].mean().unstack("hour").reindex(columns=range(24))
    n = df.groupby("day")["hour"].nunique(); short = n.index[n == 23]
    if len(short): M.loc[short] = M.loc[short].interpolate(axis=1, limit=1, limit_area="inside")
    M.index = pd.DatetimeIndex(M.index); return M


def build_features(wd, load=None, extra=None):
    """All features use only what is known at 12:00 on D-1 (demand, generation) or is known in advance
    (calendar, weather as a perfect forecast, the operator's solar and wind forecasts)."""
    load = wd.LOAD if load is None else load
    IDX, W, E, HOLS = wd.IDX, wd.W, wd.E, wd.HOLS
    loc = IDX.tz_convert(TZ); day = loc.tz_localize(None).normalize(); h = loc.hour.values
    L = to_daily(load); days = pd.date_range(L.index.min(), L.index.max(), freq="D"); L = L.reindex(days)
    pos = pd.Series(np.arange(len(days)), index=days)
    di = pos.reindex(day).values; Lv = L.values
    dii = np.where(np.isnan(di), 0, di).astype(int)
    def take(M, k, hour):                                          # M[day - k, hour]
        d = np.where(di - k >= 0, di - k, -1).astype(int); out = np.full(len(d), np.nan); ok = (d >= 0) & ~np.isnan(di)
        out[ok] = M[d[ok], hour[ok] if np.ndim(hour) else hour]; return out
    X = pd.DataFrame(index=IDX)
    X["hour_sin"], X["hour_cos"] = np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24)
    X["dow_sin"], X["dow_cos"] = np.sin(2 * np.pi * loc.dayofweek / 7), np.cos(2 * np.pi * loc.dayofweek / 7)
    X["doy_sin"], X["doy_cos"] = np.sin(2 * np.pi * loc.dayofyear / 365.25), np.cos(2 * np.pi * loc.dayofyear / 365.25)
    X["weekend"] = (loc.dayofweek >= 5).astype(float)
    X["summer_time"] = np.array([bool(t.dst()) for t in loc], float)
    inh = lambda dd: np.array([x.date() in HOLS for x in dd], float)
    X["holiday"] = inh(day); X["pre_holiday"] = inh(day + pd.Timedelta(days=1)); X["post_holiday"] = inh(day - pd.Timedelta(days=1))
    X["bridge"] = (((loc.dayofweek == 0) & (X["pre_holiday"] == 1)) | ((loc.dayofweek == 4) & (X["post_holiday"] == 1))).astype(float)
    X["christmas"] = (((loc.month == 12) & (loc.day >= 24)) | ((loc.month == 1) & (loc.day <= 6))).astype(float)
    T = W["temp"]
    X["temp"] = T.values; X["hdd"], X["cdd"] = np.clip(18 - T, 0, None).values, np.clip(T - 18, 0, None).values
    Td = to_daily(T).reindex(days)
    for f in ("mean", "max", "min"):
        X[f"temp_day_{f}"] = getattr(Td, f)(axis=1).reindex(day).values
    X["temp_ewm24"], X["temp_ewm72"] = T.ewm(halflife=24).mean().values, T.ewm(halflife=72).mean().values
    for c in [c for c in W.columns if c != "temp"]:
        X[c] = W[c].values
    for c, nm in [("forecast solar day ahead", "solar_fc"), ("forecast wind onshore day ahead", "wind_fc")]:
        if c in E: X[nm] = E[c].interpolate(limit=6, limit_area="inside").values
    LK = LAST_KNOWN_HOUR
    X["load_last_obs"] = take(Lv, 1, LK)
    X["load_change_24h"] = take(Lv, 1, LK) - take(Lv, 2, LK)
    win24 = np.full((len(days), 24), np.nan)                       # 24 h before issue: D-2 12..23 + D-1 0..11
    win24[2:, :24 - LK - 1] = Lv[:-2, LK + 1:]; win24[2:, 24 - LK - 1:] = Lv[1:-1, :LK + 1]
    for f in ("mean", "max", "min", "std"):
        X[f"load_24h_{f}"] = getattr(np, "nan" + f)(win24, axis=1)[dii]
    wk = np.full(len(days), np.nan)
    for d in range(8, len(days)):
        wk[d] = np.nanmean(np.r_[Lv[d - 8, LK + 1:], Lv[d - 7:d - 1].ravel(), Lv[d - 1, :LK + 1]])
    X["load_week_mean"] = wk[dii]
    for f in ("mean", "max", "min"):
        X[f"load_D2_{f}"] = take(getattr(np, "nan" + f)(Lv, axis=1)[:, None], 2, 0)
    X["load_same_hour_7d"], X["load_same_hour_14d"] = take(Lv, 7, h), take(Lv, 14, h)
    X["load_same_hour_latest"] = np.where(h <= LK, take(Lv, 1, h), take(Lv, 2, h))
    X["latest_is_D1"] = (h <= LK).astype(float)
    for k, c in wd.GEN.items():
        G = to_daily(E[c].interpolate(limit=6, limit_area="inside")).reindex(days).values
        X[f"gen_{k}_at_issue"] = take(G, 1, LK)
    X.loc[np.isnan(di)] = np.nan
    if extra is not None:
        for c in extra.columns: X[c] = extra[c].values
    return X


GROUPS = {"Hour of day": ["hour_sin", "hour_cos"], "Weekday": ["dow_sin", "dow_cos", "weekend"],
          "Holiday and special days": ["holiday", "pre_holiday", "post_holiday", "bridge", "christmas"],
          "Season": ["doy_sin", "doy_cos", "summer_time"],
          "Temperature": ["temp", "hdd", "cdd", "temp_day_mean", "temp_day_max", "temp_day_min", "temp_ewm24", "temp_ewm72"],
          "Other weather": ["humidity", "pressure", "wind_speed", "clouds_all"], "Renewable forecasts": ["solar_fc", "wind_fc"],
          "Gas and coal (consequences)": ["gen_gas_at_issue", "gen_coal_at_issue"],
          "Other generation": ["gen_nuclear_at_issue", "gen_hydro_at_issue", "gen_solar_at_issue", "gen_wind_at_issue"],
          "Injected event": ["event_flag"], "Injected driver": ["driver_z"], "Injected decoy": ["decoy_d"]}
def is_history(f): return f.startswith("load_") or f == "latest_is_D1"
def group_of(f):
    if is_history(f): return "Demand history"
    return next((g for g, v in GROUPS.items() if f in v), "Other")

# benchmark truth: which features carry each true driver, and the causal role of every feature
B_GROUPS = {"hour": GROUPS["Hour of day"], "week": GROUPS["Weekday"], "holiday": ["holiday"], "temperature": GROUPS["Temperature"]}
B_TRUTH = {"hour": "C_hour", "week": "C_week", "holiday": "C_holiday", "temperature": "C_temp"}
def role(f):
    if any(f in v for v in B_GROUPS.values()): return "direct cause"
    if is_history(f): return "demand history"
    if f in GROUPS["Gas and coal (consequences)"]: return "consequence"
    return "irrelevant"


def leakage_audit(wd, builder, n=10, load=None):
    """Raise one demand value by 100,000 MW, rebuild all features, and fail if any forecast issued before that
    value existed has changed."""
    load = wd.LOAD if load is None else load
    issue_time = wd.IDX.tz_convert(TZ).tz_localize(None).normalize() - pd.Timedelta(days=1) + pd.Timedelta(hours=ISSUE_HOUR)
    base = builder(load); rng = np.random.default_rng(0); viol, nch = [], 0
    okpos = np.flatnonzero(load.notna().values)
    for j in rng.choice(okpos[len(okpos) // 10:-24 * 30], n, replace=False):
        l2 = load.copy(); l2.iloc[j] += 1e5; X2 = builder(l2)
        ch = ~((X2.values == base.values) | (np.isnan(X2.values) & np.isnan(base.values))).all(axis=1); nch += ch.any()
        t_obs = wd.IDX[j].tz_convert(TZ).tz_localize(None); bad = ch & (issue_time <= t_obs)
        if bad.any(): viol.append(f"demand at {t_obs} changes rows issued at {issue_time[bad][0]}")
    assert nch > 0, "audit is not sensitive"
    if viol: raise RuntimeError("LEAKAGE:\n" + "\n".join(viol))
    return int(nch)


def make_injection(wd, cfg=INJECT):
    """Real demand + an event of exactly event_mw on random weekdays + a nonlinear driver + a decoy with no effect."""
    rng = np.random.default_rng(cfg["seed"]); IDX = wd.IDX
    loc = IDX.tz_convert(TZ); day = loc.tz_localize(None).normalize()
    days = pd.DatetimeIndex(sorted(set(day)))
    cand = [d for d in days[30:-2] if d.dayofweek < 5 and d.date() not in wd.HOLS]
    ev_days = set(pd.DatetimeIndex(rng.choice(cand, cfg["n_event_days"], replace=False)))
    h0, h1 = cfg["event_hours"]
    flag = np.array([(d in ev_days) and (h0 <= hh <= h1) for d, hh in zip(day, loc.hour)], float)
    zd = np.zeros(len(days)); e = rng.normal(0, np.sqrt(1 - 0.9 ** 2), len(days))
    for i in range(1, len(days)): zd[i] = 0.9 * zd[i - 1] + e[i]
    di = pd.Series(np.arange(len(days)), index=days).reindex(day).values
    z = np.interp(di + loc.hour / 24, np.arange(len(days)), zd) + rng.normal(0, 0.05, len(IDX))
    decoy = cfg["decoy_corr"] * z + np.sqrt(1 - cfg["decoy_corr"] ** 2) * rng.normal(0, 1, len(IDX))
    cdd = np.clip(wd.W["temp"].values - 18, 0, None)
    C_event = cfg["event_mw"] * flag
    C_z = cfg["driver_mw"] * np.tanh(1.2 * z) * (1 + cfg["driver_temp_interaction"] * np.nan_to_num(cdd) / 5)
    truth = pd.DataFrame({"C_event": C_event, "C_z": C_z}, index=IDX)
    extra = pd.DataFrame({"event_flag": flag, "driver_z": z, "decoy_d": decoy}, index=IDX)
    return wd.LOAD + C_event + C_z, extra, truth


def prepare(wd, X, Y, extra_mask, n_explain, truth=None, seed=99):
    """Chronological split by whole local days, standardisation on the training part, explained sample."""
    loc_day = wd.IDX.tz_convert(TZ).tz_localize(None).normalize()
    valid = X.notna().all(axis=1).values & Y.notna().values
    dv = pd.DatetimeIndex(sorted(set(loc_day[valid]))); n = len(dv)
    val_start, test_start = dv[int(SPLIT[0] * n)], dv[int((SPLIT[0] + SPLIT[1]) * n)]
    part = np.where(loc_day < val_start, 0, np.where(loc_day < test_start, 1, 2))
    vr = pd.Series(valid.astype(int)).rolling(WMAX).sum().fillna(0).values == WMAX
    I = [np.flatnonzero(vr & (part == k)) for k in range(3)]
    mu, sd = np.nanmean(X.values[I[0]], 0), np.nanstd(X.values[I[0]], 0) + 1e-9
    ym, ysd = float(np.nanmean(Y.values[I[0]])), float(np.nanstd(Y.values[I[0]]))
    XS = np.nan_to_num(((X.values - mu) / sd).astype(np.float32)); YS = ((Y.values - ym) / ysd).astype(np.float32)
    rand = np.random.default_rng(seed).choice(I[2], min(n_explain, len(I[2])), replace=False)
    extra = np.flatnonzero(extra_mask & vr & (part == 2)) if extra_mask is not None else np.array([], int)
    sample = np.unique(np.r_[rand, extra]).astype(int)
    feats = list(X.columns)
    return dict(wd=wd, Y=Y, XS=XS, YS=YS, ym=ym, ysd=ysd, I_TR=I[0], I_VA=I[1], I_TE=I[2], SAMPLE=sample, feats=feats,
                truth=None if truth is None else truth.iloc[sample].reset_index(drop=True),
                test_start=str(test_start.date()), val_start=str(val_start.date()))


def windows(XS, idx, T):
    return np.stack([XS[i - T + 1:i + 1] for i in idx]).astype(np.float32)

# %% [markdown]
# ## 1.4 Scoring against the truth, and indicators that need no truth

# %%
def slope_r(a, c):
    c = c - c.mean()
    if np.std(c) < 1e-9 or np.std(a) < 1e-12: return np.nan, (0.0 if np.std(c) > 1e-9 else np.nan)
    return float(np.corrcoef(a, c)[0, 1]), float(np.cov(a, c)[0, 1] / np.var(c, ddof=1))


def score_bench(A, feats, truth):
    """Pattern r and magnitude slope b per driver, magnitude correctness = 1 - mean|1 - b|, shares by causal role."""
    A = np.asarray(A, float); out = {}; bs, rs = [], []
    for g, cols in B_GROUPS.items():
        j = [feats.index(c) for c in cols if c in feats]
        r, b = slope_r(A[:, j].sum(1), truth[B_TRUTH[g]].values)
        out[f"r_{g}"], out[f"b_{g}"] = r, b; bs.append(b); rs.append(r)
    out["magnitude_correctness"] = float(1 - np.nanmean(np.abs(1 - np.array(bs, float))))
    out["pattern_correctness"] = float(np.nanmean(rs)); out["mean_slope"] = float(np.nanmean(bs))
    tot = np.abs(A).sum(); roles = np.array([role(f) for f in feats])
    for rl in ("direct cause", "demand history", "consequence", "irrelevant"):
        out[f"share_{rl}"] = float(np.abs(A[:, roles == rl]).sum() / max(tot, 1e-12))
    return out


def score_injected(A, feats, truth):
    out = {}
    for comp, feat, col in [("event", "event_flag", "C_event"), ("driver", "driver_z", "C_z")]:
        out[f"r_{comp}"], out[f"b_{comp}"] = slope_r(A[:, feats.index(feat)], truth[col].values)
    dz, dd = np.abs(A[:, feats.index("driver_z")]).mean(), np.abs(A[:, feats.index("decoy_d")]).mean()
    out["decoy_credit"] = float(dd / max(dz, 1e-9))
    out["magnitude_correctness"] = float(1 - np.mean([abs(1 - out["b_event"]), abs(1 - out["b_driver"])]))
    tot = np.abs(A).sum()
    out["share_demand history"] = float(np.abs(A[:, [is_history(f) for f in feats]]).sum() / max(tot, 1e-12))
    return out


def indicators(predict, Xex, A, seq, ysd, n_subsets, rng, K=10):
    """Truth-free indicators. Features are removed by setting them to the training mean (0 after standardisation),
    for sequence models at every lag. A is (n, F) in MW; forecasts are converted to MW with ysd.
    deletion / insertion: most-relevant-first, k = 1..K, scaled by the sd of the forecasts (higher = more faithful).
    faithfulness_correlation: per forecast, corr over random 20% subsets S of (sum of attributions in S, change in forecast).
    relative_infidelity: E[(sum_S A - change)^2] / E[change^2]   (0 perfect, 1 = an explanation of zeros)."""
    n, F = A.shape; f0 = predict(Xex) * ysd; scale = max(np.std(f0), 1e-9)
    def masked(keep_cols=None, drop_cols=None):
        M = Xex.copy() if drop_cols is not None else np.zeros_like(Xex)
        for i in range(n):
            if drop_cols is not None:
                if seq: M[i, :, drop_cols[i]] = 0.0
                else:   M[i, drop_cols[i]] = 0.0
            else:
                if seq: M[i, :, keep_cols[i]] = Xex[i, :, keep_cols[i]]
                else:   M[i, keep_cols[i]] = Xex[i, keep_cols[i]]
        return predict(M) * ysd
    order = np.argsort(-np.abs(A), axis=1); dele, ins = [], []
    for m in range(1, min(K, F) + 1):
        dele.append(np.mean(np.abs(f0 - masked(drop_cols=order[:, :m]))))
        ins.append(np.mean(np.abs(masked(keep_cols=order[:, :m]) - f0)))
    size = max(1, int(round(0.2 * F))); sa, df = np.zeros((n_subsets, n)), np.zeros((n_subsets, n))
    for s in range(n_subsets):
        S = rng.choice(F, size, replace=False); M = Xex.copy()
        if seq: M[:, :, S] = 0.0
        else:   M[:, S] = 0.0
        df[s] = f0 - predict(M) * ysd; sa[s] = A[:, S].sum(1)
    fc = [np.corrcoef(sa[:, i], df[:, i])[0, 1] for i in range(n) if np.std(sa[:, i]) > 0 and np.std(df[:, i]) > 0]
    return dict(deletion=float(np.mean(dele) / scale), insertion=float(1 - np.mean(ins) / scale),
                faithfulness_correlation=float(np.mean(fc)) if fc else np.nan,
                relative_infidelity=float(np.mean((sa - df) ** 2) / max(np.mean(df ** 2), 1e-12)))


def make_controls(A, feats, rng):
    """Altered copies of one explanation whose errors are known (Table 4 of the paper)."""
    s = A.std(); out = {"halved": 0.5 * A, "times 1.5": 1.5 * A}
    cal = [k for k, f in enumerate(feats) if any(f in v for v in B_GROUPS.values())]
    hist = [k for k, f in enumerate(feats) if is_history(f)]
    if cal and hist:
        a = A.copy(); moved = 0.5 * a[:, cal]; a[:, cal] -= moved; a[:, hist[0]] += moved.sum(1); out["half of driver credit moved to history"] = a
    out["noise 1 sd"] = A + rng.normal(0, s, A.shape); out["noise 2 sd"] = A + rng.normal(0, 2 * s, A.shape)
    out["permuted labels"] = A[:, rng.permutation(A.shape[1])]; out["random"] = rng.normal(0, s, A.shape)
    return out


def pair_agreement(a, b, k=5):
    out = []
    for x, y in zip(a, b):
        ta, tb = np.argsort(-np.abs(x))[:k], np.argsort(-np.abs(y))[:k]
        out.append((stats.spearmanr(np.abs(x), np.abs(y)).statistic, stats.weightedtau(np.abs(x), np.abs(y)).statistic,
                    len(set(ta) & set(tb)) / len(set(ta) | set(tb)), np.mean(np.sign(x[tb]) == np.sign(y[tb]))))
    return np.nanmean(np.array(out), axis=0)


def diebold_mariano(e1, e2, h=24):
    """Two-sided DM test on absolute errors with the Harvey-Leybourne-Newbold correction."""
    d = np.abs(e1) - np.abs(e2); n, m = len(d), d.mean()
    g = [np.mean((d[k:] - m) * (d[:n - k] - m)) for k in range(h)]
    var = (g[0] + 2 * sum(g[1:])) / n
    if var <= 0: return np.nan, np.nan
    s = m / math.sqrt(var) * math.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    return float(s), float(2 * stats.t.sf(abs(s), n - 1))


def boot_ci(v, n_boot=5000, seed=0):
    v = np.asarray(v, float); v = v[~np.isnan(v)]
    if len(v) < 2: return (np.nan, np.nan)
    rng = np.random.default_rng(seed); m = v[rng.integers(0, len(v), (n_boot, len(v)))].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def paired(a, b):
    """Paired comparison over seeds: mean difference, bootstrap interval, wins, exact Wilcoxon signed-rank p."""
    d = np.asarray(a, float) - np.asarray(b, float); d = d[~np.isnan(d)]
    if len(d) < 2: return dict(n=len(d))
    lo, hi = boot_ci(d)
    try: p = float(stats.wilcoxon(d).pvalue) if np.any(d != 0) else np.nan
    except ValueError: p = np.nan
    return dict(n=len(d), mean_difference=float(d.mean()), ci_low=lo, ci_high=hi, a_better=int((d > 0).sum()), wilcoxon_p=p)

# %% [markdown]
# ## 1.5 Models and explainers

# %%
class RidgeModel:
    """Transparent reference: ridge regression on the target hour's features, explained exactly (linear SHAP)."""
    def fit(self, Xtr, ytr, Xva=None, yva=None):
        self.m = RidgeCV(alphas=np.logspace(-2, 4, 25)).fit(Xtr, ytr); self.mu = Xtr.mean(0); self.epochs = 0; return self
    def predict(self, X): return self.m.predict(X).astype(np.float32)
    def explain(self, X): return (X - self.mu) * self.m.coef_


_NN = None
def nn_lib():
    """Neural models (imported lazily so that the rest of the pipeline runs without PyTorch)."""
    global _NN
    if _NN is not None: return _NN
    import torch, torch.nn as nn
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class LSTMNet(nn.Module):
        def __init__(self, F, hidden, layers, dropout):
            super().__init__()
            self.rnn = nn.LSTM(F, hidden, layers, batch_first=True, dropout=dropout if layers > 1 else 0.0)
            self.head = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, 1))
        def forward(self, x):
            h, _ = self.rnn(x); return self.head(h[:, -1]).squeeze(-1)

    class RETAINNet(nn.Module):
        def __init__(self, F, hidden, dropout):
            super().__init__()
            self.emb = nn.Linear(F, hidden)
            self.gru_a = nn.GRU(hidden, hidden, batch_first=True); self.gru_b = nn.GRU(hidden, hidden, batch_first=True)
            self.att_a = nn.Linear(hidden, 1); self.att_b = nn.Linear(hidden, hidden)
            self.drop = nn.Dropout(dropout); self.out = nn.Linear(hidden, 1)
        def attention(self, x):
            v = self.emb(x); rv = torch.flip(self.drop(v), [1])
            g, _ = self.gru_a(rv); h, _ = self.gru_b(rv); g, h = torch.flip(g, [1]), torch.flip(h, [1])
            return v, torch.softmax(self.att_a(g).squeeze(-1), dim=1), torch.tanh(self.att_b(h))
        def forward(self, x):
            v, a, b = self.attention(x); return self.out((a.unsqueeze(-1) * b * v).sum(1)).squeeze(-1)
        def contributions(self, x):
            """Exact contribution (n, T, F), Equation 1 of the paper, and the feature-independent bias (n,)."""
            v, a, b = self.attention(x); w, W = self.out.weight.squeeze(0), self.emb.weight
            coef = torch.einsum("nt,nth,h,hf->ntf", a, b, w, W)
            bias = ((a.unsqueeze(-1) * b * self.emb.bias).sum(1) * w).sum(-1) + self.out.bias
            return x * coef, bias

    class PosEnc(nn.Module):
        def __init__(self, d, T=512):
            super().__init__()
            pe = torch.zeros(T, d); pos = torch.arange(T).unsqueeze(1)
            div = torch.exp(torch.arange(0, d, 2) * (-np.log(10000.0) / d))
            pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
            self.register_buffer("pe", pe)
        def forward(self, x): return x + self.pe[: x.shape[1]]

    class EncLayer(nn.Module):
        def __init__(self, d, heads, dropout):
            super().__init__()
            self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
            self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.ReLU(), nn.Linear(4 * d, d))
            self.n1, self.n2, self.drop = nn.LayerNorm(d), nn.LayerNorm(d), nn.Dropout(dropout)
        def forward(self, x):
            a, _ = self.attn(x, x, x, need_weights=False)
            x = self.n1(x + self.drop(a)); return self.n2(x + self.drop(self.ff(x)))

    class TransformerNet(nn.Module):
        def __init__(self, F, d_model, heads, layers, dropout):
            super().__init__()
            self.inp, self.pos = nn.Linear(F, d_model), PosEnc(d_model)
            self.layers = nn.ModuleList([EncLayer(d_model, heads, dropout) for _ in range(layers)])
            self.head = nn.Linear(d_model, 1)
        def forward(self, x):
            z = self.pos(self.inp(x))
            for L in self.layers: z = L(z)
            return self.head(z[:, -1]).squeeze(-1)

    class GLU(nn.Module):
        def __init__(self, d_in, d_out):
            super().__init__(); self.l = nn.Linear(d_in, 2 * d_out)
        def forward(self, x):
            a, b = self.l(x).chunk(2, dim=-1); return a * torch.sigmoid(b)

    class GRN(nn.Module):
        def __init__(self, d_in, d_hid, d_out, dropout):
            super().__init__()
            self.skip = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()
            self.l1, self.l2 = nn.Linear(d_in, d_hid), nn.Linear(d_hid, d_hid)
            self.drop, self.glu, self.norm = nn.Dropout(dropout), GLU(d_hid, d_out), nn.LayerNorm(d_out)
        def forward(self, a):
            h = self.drop(self.l2(nn.functional.elu(self.l1(a)))); return self.norm(self.skip(a) + self.glu(h))

    class InterpretableMHA(nn.Module):
        def __init__(self, d, heads, dropout):
            super().__init__()
            self.dk = d // heads
            self.q = nn.ModuleList([nn.Linear(d, self.dk) for _ in range(heads)])
            self.k = nn.ModuleList([nn.Linear(d, self.dk) for _ in range(heads)])
            self.v, self.o, self.drop = nn.Linear(d, self.dk), nn.Linear(self.dk, d), nn.Dropout(dropout)
        def forward(self, q, k):
            v, outs = self.v(k), []
            for qi, ki in zip(self.q, self.k):
                a = torch.softmax(qi(q) @ ki(k).transpose(1, 2) / self.dk ** 0.5, dim=-1); outs.append(self.drop(a) @ v)
            return self.o(torch.stack(outs).mean(0))

    class TFTNet(nn.Module):
        """TFT-style network for one target hour (after Lim et al., 2021): variable selection -> LSTM encoder with
        gated skip -> enrichment GRN -> interpretable attention from the last step -> gated GRN -> output.
        Simplifications: no static covariates; one GRN shared by all variables."""
        def __init__(self, F, d_model, heads, dropout):
            super().__init__()
            d = d_model
            self.emb_w, self.emb_b = nn.Parameter(torch.randn(F, d) * 0.1), nn.Parameter(torch.zeros(F, d))
            self.vsn, self.var_grn = GRN(F * d, d, F, dropout), GRN(d, d, d, dropout)
            self.lstm = nn.LSTM(d, d, batch_first=True)
            self.gate1, self.norm1 = GLU(d, d), nn.LayerNorm(d)
            self.enrich = GRN(d, d, d, dropout); self.attn = InterpretableMHA(d, heads, dropout)
            self.gate2, self.norm2 = GLU(d, d), nn.LayerNorm(d)
            self.ff, self.gate3, self.norm3 = GRN(d, d, d, dropout), GLU(d, d), nn.LayerNorm(d)
            self.out = nn.Linear(d, 1); self.last_w = None
        def forward(self, x, keep=False):
            e = x.unsqueeze(-1) * self.emb_w + self.emb_b
            n, T, F_, d = e.shape
            w = torch.softmax(self.vsn(e.reshape(n, T, F_ * d)), dim=-1)
            z = (w.unsqueeze(-1) * self.var_grn(e)).sum(2)
            h, _ = self.lstm(z); h = self.enrich(self.norm1(z + self.gate1(h)))
            last = h[:, -1:]; y = self.norm2(last + self.gate2(self.attn(last, h)))
            y = self.norm3(y + self.gate3(self.ff(y)))
            if keep: self.last_w = w
            return self.out(y[:, 0]).squeeze(-1)
        def selection_weights(self, x):
            self.forward(x, keep=True); return self.last_w.mean(1)

    class Out2D(nn.Module):
        def __init__(self, net): super().__init__(); self.net = net
        def forward(self, x): return self.net(x).unsqueeze(-1)

    def build_net(kind, F, a):
        if kind == "LSTM": return LSTMNet(F, a["hidden"], a["layers"], a["dropout"])
        if kind == "RETAIN": return RETAINNet(F, a["hidden"], a["dropout"])
        if kind == "TFT": return TFTNet(F, a["d_model"], a["heads"], a["dropout"])
        return TransformerNet(F, a["d_model"], a["heads"], a["layers"], a["dropout"])

    class SeqModel:
        """Train / predict / explain one network on standardised windows (n, T, F)."""
        def __init__(self, kind, F, seed, p):
            torch.manual_seed(seed); np.random.seed(seed)
            if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
            self.kind, self.p, self.net = kind, p, build_net(kind, F, p).to(DEVICE)
        def _t(self, a): return torch.as_tensor(a, dtype=torch.float32, device=DEVICE)
        def fit(self, Xtr, ytr, Xva, yva, max_epochs, patience, trial=None):
            p = self.p
            opt = torch.optim.AdamW(self.net.parameters(), lr=p.get("lr", 1e-3), weight_decay=p.get("weight_decay", 1e-4))
            sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=max(2, patience // 4))
            loss_fn = nn.HuberLoss(); best, best_state, bad = np.inf, None, 0
            Xtr_t, ytr_t = self._t(Xtr), self._t(ytr); n = len(Xtr); bs = int(p.get("batch", 256)); self.epochs = 0
            for ep in range(max_epochs):
                self.net.train(); perm = torch.randperm(n, device=DEVICE)
                for i in range(0, n, bs):
                    idx = perm[i:i + bs]
                    opt.zero_grad(); loss = loss_fn(self.net(Xtr_t[idx]), ytr_t[idx]); loss.backward()
                    nn.utils.clip_grad_norm_(self.net.parameters(), 1.0); opt.step()
                val = float(np.mean(np.abs(self.predict(Xva) - yva)))
                if not np.isfinite(val): break
                sched.step(val); self.epochs = ep + 1
                if val < best - 1e-5:
                    best, bad = val, 0; best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
                else:
                    bad += 1
                if trial is not None:
                    import optuna
                    trial.report(val, ep)
                    if trial.should_prune(): raise optuna.TrialPruned()
                if bad >= patience: break
            if best_state is not None: self.net.load_state_dict(best_state)
            self.net.eval(); self.best_val = best; return self
        @torch.no_grad()
        def predict(self, Xw, bs=2048):
            self.net.eval()
            return np.concatenate([self.net(self._t(Xw[i:i + bs])).cpu().numpy() for i in range(0, len(Xw), bs)])
        def shap(self, Xw, background, nsamples, seed=0):
            """Expected gradients (shap.GradientExplainer), one value per lag and feature."""
            import shap
            self.net.eval(); wrapped = Out2D(self.net).eval()
            with torch.backends.cudnn.flags(enabled=False):
                ex = shap.GradientExplainer(wrapped, self._t(background))
                sv = ex.shap_values(self._t(Xw), nsamples=nsamples, rseed=seed)
            sv = np.asarray(sv[0] if isinstance(sv, list) else sv)
            return sv.reshape(len(Xw), Xw.shape[1], Xw.shape[2])
        def ig(self, Xw, steps, bs=64):
            """Integrated Gradients from the training mean (0 after standardisation)."""
            self.net.eval(); out = []
            with torch.backends.cudnn.flags(enabled=False):
                for i in range(0, len(Xw), bs):
                    x = self._t(Xw[i:i + bs]); total = torch.zeros_like(x)
                    for a in torch.arange(1, steps + 1, device=DEVICE, dtype=torch.float32) / steps:
                        xi = (a * x).detach().requires_grad_(True)
                        g, = torch.autograd.grad(self.net(xi).sum(), xi); total += g
                    out.append((total / steps * x).detach().cpu().numpy())
            return np.concatenate(out)
        @torch.no_grad()
        def intrinsic(self, Xw):
            self.net.eval(); x = self._t(Xw)
            if self.kind == "RETAIN":
                c, bias = self.net.contributions(x)
                err = (c.sum((1, 2)) + bias - self.net(x)).abs().max().item()
                assert err < 1e-3, f"RETAIN identity violated ({err:.2e})"
                return c.cpu().numpy(), err
            if self.kind == "TFT": return self.net.selection_weights(x).cpu().numpy(), None
            return None, None

    _NN = SimpleNamespace(SeqModel=SeqModel, DEVICE=DEVICE, torch=torch)
    return _NN


class XGBModel:
    def __init__(self, seed, p, n_trees):
        import xgboost as xgb
        kw = dict(n_estimators=n_trees, max_depth=int(p.get("max_depth", 6)), learning_rate=p.get("lr", 0.03),
                  min_child_weight=p.get("min_child_weight", 5), subsample=p.get("subsample", 0.8),
                  colsample_bytree=p.get("colsample", 0.8), reg_lambda=p.get("reg_lambda", 1.0),
                  early_stopping_rounds=200, n_jobs=-1, random_state=seed, verbosity=0, tree_method="hist")
        try:
            import torch
            if torch.cuda.is_available() and int(xgb.__version__.split(".")[0]) >= 2: kw["device"] = "cuda"
        except ImportError: pass
        self.m = xgb.XGBRegressor(**kw)
    def fit(self, Xtr, ytr, Xva, yva):
        self.m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False); self.epochs = int(self.m.best_iteration); return self
    def predict(self, X): return self.m.predict(X)
    def explain(self, X):
        """TreeSHAP, path-dependent (no background): the variant used in the paper."""
        import shap
        return np.asarray(shap.TreeExplainer(self.m).shap_values(X))
    def explain_interventional(self, X, bg):
        import shap
        return np.asarray(shap.TreeExplainer(self.m, data=bg, feature_perturbation="interventional").shap_values(X, check_additivity=False))

# %% [markdown]
# ## 1.6 Run bookkeeping, hyperparameter search and one run

# %%
class Run:
    def __init__(self, args):
        self.a = args; self.cfg = dict(PLANS[args.plan])
        if args.seeds: self.cfg["seeds"] = args.seeds
        self.seeds = list(range(self.cfg["seeds"]))
        self.dir = Path(args.out) / args.plan
        for d in ("attributions", "forecasts", "tables"): (self.dir / d).mkdir(parents=True, exist_ok=True)
        self.logf = self.dir / "log.jsonl"; self._cache = {}; self._real = None
        self.models = [m for m in (args.models.split(",") if args.models else ALL_MODELS)]
        self.best = json.loads((self.dir / "best_params.json").read_text()) if (self.dir / "best_params.json").exists() else {}
    def rows(self):
        return [json.loads(l) for l in self.logf.read_text().splitlines() if l.strip()] if self.logf.exists() else []
    def log(self, row):
        with open(self.logf, "a") as f: f.write(json.dumps(row, default=float) + "\n")
    def done(self, **key):
        return any(all(r.get(k) == v for k, v in key.items()) for r in self.rows())
    def say(self, msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    def real(self):
        if self._real is None: self._real = load_real(self.a.data)
        return self._real

    def dataset(self, name):
        """name: real | injected | bench:<variant>:<seed> | calib:<seed>, optionally followed by |nolags."""
        if name in self._cache: return self._cache[name]
        base, _, variant = name.partition("|"); n_ex = self.cfg["n_explain"]; truth = None; extra = None; mask = None
        if base == "real":
            wd = self.real(); load = wd.LOAD
            mask = np.array([d.date() in wd.HOLS for d in wd.IDX.tz_convert(TZ).tz_localize(None).normalize()]) & (wd.IDX.tz_convert(TZ).dayofweek < 5)
        elif base == "injected":
            wd = self.real(); load, extra, truth = make_injection(wd); mask = extra["event_flag"].values == 1
        else:
            kind, *rest = base.split(":")
            if kind == "bench":
                cond = next(c for c in CONDITIONS if c["name"] == rest[0]); g = GEN_DEFAULT; seed = int(rest[1])
            else:
                cond = CONDITIONS[0]; seed = int(rest[0]); g = self.calibrated()
            energy, weather, truth, hols, _ = generate(cond, seed, {k: v for k, v in g.items() if not k.startswith("_")})
            wd = make_world(energy, weather, hols); load = wd.LOAD
        X = build_features(wd, load, extra)
        if variant == "nolags":                                   # no demand history, no lagged generation
            X = X[[c for c in X.columns if not is_history(c) and not c.startswith("gen_")]]
        P = prepare(wd, X, load, mask, n_ex, truth); P["name"] = name
        if len(self._cache) > 6: self._cache.pop(next(iter(self._cache)))
        self._cache[name] = P; return P

    def calibrated(self):
        f = self.dir / "calibrated_generator.json"
        if not f.exists(): f.write_text(json.dumps(calibrate_generator(self.real()), indent=2))
        g = json.loads(f.read_text()); g["hour_amp"], g["hour_phase"] = tuple(g["hour_amp"]), tuple(g["hour_phase"]); return g


def tag(dataset, kind, seed): return re.sub(r"[^A-Za-z0-9_]+", "-", f"{dataset}__{kind}__seed{seed}")

DEFAULTS = {"LSTM": dict(hidden=128, layers=2, dropout=0.1, lr=1e-3, window=24),
            "Transformer": dict(d_model=64, heads=4, layers=2, dropout=0.1, lr=1e-3, window=24),
            "TFT": dict(d_model=32, heads=4, dropout=0.1, lr=1e-3, window=24),
            "RETAIN": dict(hidden=128, dropout=0.2, lr=1e-3, window=24), "XGBoost": dict(max_depth=6, lr=0.03)}


def suggest(kind, trial, cfg):
    if kind == "XGBoost":
        return dict(max_depth=trial.suggest_int("max_depth", 3, 10), lr=trial.suggest_float("lr", 0.01, 0.1, log=True),
                    min_child_weight=trial.suggest_float("min_child_weight", 1, 20, log=True),
                    subsample=trial.suggest_float("subsample", 0.6, 1.0), colsample=trial.suggest_float("colsample", 0.4, 1.0),
                    reg_lambda=trial.suggest_float("reg_lambda", 1e-2, 10, log=True))
    p = dict(lr=trial.suggest_float("lr", 1e-4, 5e-3, log=True), weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
             batch=trial.suggest_categorical("batch", [128, 256, 512]), dropout=trial.suggest_float("dropout", 0.0, 0.4),
             window=trial.suggest_categorical("window", cfg["windows"]))
    if kind == "LSTM": p.update(hidden=trial.suggest_categorical("hidden", [64, 128, 256, 384]), layers=trial.suggest_int("layers", 1, 3))
    elif kind == "RETAIN": p.update(hidden=trial.suggest_categorical("hidden", [64, 128, 256]))
    elif kind == "Transformer": p.update(d_model=trial.suggest_categorical("d_model", [32, 64, 128]), heads=trial.suggest_categorical("heads", [2, 4]),
                                         layers=trial.suggest_int("layers", 1, 4))
    elif kind == "TFT": p.update(d_model=trial.suggest_categorical("d_model", [16, 32, 64]), heads=trial.suggest_categorical("heads", [2, 4]))
    return p


class _RandomTrial:
    """Stand-in for an Optuna trial when Optuna is not installed (plain random search, no pruning)."""
    def __init__(self, rng): self.rng, self.params = rng, {}
    def suggest_int(self, n, lo, hi): self.params[n] = int(self.rng.integers(lo, hi + 1)); return self.params[n]
    def suggest_float(self, n, lo, hi, log=False):
        self.params[n] = float(np.exp(self.rng.uniform(np.log(lo), np.log(hi)))) if log else float(self.rng.uniform(lo, hi)); return self.params[n]
    def suggest_categorical(self, n, choices): self.params[n] = choices[int(self.rng.integers(len(choices)))]; return self.params[n]
    def report(self, *a): pass
    def should_prune(self): return False


def stage_tune(R):
    """Search on the REAL validation period only. The test period is never touched here."""
    P = R.dataset("real"); cfg = R.cfg
    for kind in [m for m in R.models if m != "Ridge"]:
        if kind in R.best: continue
        def objective(trial):
            p = suggest(kind, trial, cfg)
            if kind == "XGBoost":
                m = XGBModel(0, p, cfg["xgb_trees"]).fit(P["XS"][P["I_TR"]], P["YS"][P["I_TR"]], P["XS"][P["I_VA"]], P["YS"][P["I_VA"]])
                pred = m.predict(P["XS"][P["I_VA"]])
            else:
                T = p["window"]; Wtr, Wva = windows(P["XS"], P["I_TR"], T), windows(P["XS"], P["I_VA"], T)
                m = nn_lib().SeqModel(kind, len(P["feats"]), 0, p).fit(Wtr, P["YS"][P["I_TR"]], Wva, P["YS"][P["I_VA"]],
                                                                         cfg["tune_epochs"], cfg["tune_patience"], trial=trial)
                pred = m.predict(Wva)
            return float(np.nanmean(np.abs(pred - P["YS"][P["I_VA"]])) * P["ysd"])
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            study = optuna.create_study(study_name=kind, direction="minimize", storage=f"sqlite:///{R.dir / 'optuna.db'}",
                                        load_if_exists=True, sampler=optuna.samplers.TPESampler(seed=0),
                                        pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=5))
            left = cfg["tune_trials"] - len([t for t in study.trials if t.state.name in ("COMPLETE", "PRUNED")])
            if left > 0:
                study.enqueue_trial({k: v for k, v in DEFAULTS.get(kind, {}).items()}, skip_if_exists=True)
                def _log_trial(s, t, kind=kind):
                    # A pruned trial keeps the last value it reported while training: the validation MAE in standardised
                    # units, not MW. Optuna's best trial is chosen among completed trials only, so pruned ones never win.
                    done = [x.value for x in s.trials if x.state.name == "COMPLETE"]
                    best_so_far = f" | best so far {min(done):.0f} MW" if done else ""
                    if t.state.name == "PRUNED":
                        at = f" (val MAE {t.value * P['ysd']:.0f} MW at that epoch)" if t.value is not None else ""
                        R.say(f"tune {kind:11s} trial {t.number}: pruned after epoch {(t.last_step or 0) + 1}{at}{best_so_far}")
                    elif t.state.name == "COMPLETE":
                        R.say(f"tune {kind:11s} trial {t.number}: val MAE {t.value:.0f} MW{best_so_far}")
                    else:
                        R.say(f"tune {kind:11s} trial {t.number}: {t.state.name.lower()}{best_so_far}")
                study.optimize(objective, n_trials=left, catch=(RuntimeError, ValueError), gc_after_trial=True, callbacks=[_log_trial])
            best, val = dict(study.best_params), study.best_value
        except ImportError:
            rng = np.random.default_rng(1); trials = []
            for t in range(cfg["tune_trials"]):
                tr = _RandomTrial(rng); v = objective(tr); trials.append((v, dict(tr.params)))
                R.say(f"tune {kind:11s} trial {t}: {v:.0f} MW (random search; install optuna for TPE)")
            val, best = min(trials, key=lambda r: r[0])
        R.best[kind] = dict(DEFAULTS.get(kind, {}), **best, val_MAE=val)
        (R.dir / "best_params.json").write_text(json.dumps(R.best, indent=2, default=float))
        R.say(f"BEST {kind}: val MAE {val:.0f} MW | {best}")


def run_one(R, dataset, kind, seed, with_ig=False, with_indicators=False, with_controls=False):
    """Train one model on one dataset with one seed, forecast the test period, explain the sample, save everything."""
    if R.done(stage="run", dataset=dataset, model=kind, seed=seed): return
    P = R.dataset(dataset); cfg = R.cfg; t0 = time.time(); F = len(P["feats"])
    XS, YS, ysd, ym = P["XS"], P["YS"], P["ysd"], P["ym"]; I_TR, I_VA, I_TE, S = P["I_TR"], P["I_VA"], P["I_TE"], P["SAMPLE"]
    p = dict(DEFAULTS.get(kind, {}), **R.best.get(kind, {})); A, extra_row = {}, {}
    rng = np.random.default_rng(seed + 7)
    if kind in ("Ridge", "XGBoost"):
        seq = False; Xex = XS[S]
        m = (RidgeModel() if kind == "Ridge" else XGBModel(seed, p, cfg["xgb_trees"])).fit(XS[I_TR], YS[I_TR], XS[I_VA], YS[I_VA])
        pred = m.predict(XS[I_TE]) * ysd + ym; predict = m.predict
        base = "Ridge, exact" if kind == "Ridge" else "XGBoost, TreeSHAP"
        A[base] = m.explain(Xex) * ysd
        if kind == "XGBoost":
            try:
                bg = XS[I_TR][rng.choice(len(I_TR), min(cfg["shap_bg"], len(I_TR)), replace=False)]
                A["XGBoost, SHAP interventional"] = m.explain_interventional(Xex, bg) * ysd
            except Exception as e:
                extra_row["interventional_error"] = str(e)[:200]
    else:
        seq = True; T = int(p.get("window", 24)); nn = nn_lib()
        Wtr, Wva, Wte, Xex = windows(XS, I_TR, T), windows(XS, I_VA, T), windows(XS, I_TE, T), windows(XS, S, T)
        bg = Wtr[np.random.default_rng(seed).choice(len(Wtr), min(cfg["shap_bg"], len(Wtr)), replace=False)]
        m = nn.SeqModel(kind, F, seed, p).fit(Wtr, YS[I_TR], Wva, YS[I_VA], cfg["max_epochs"], cfg["patience"])
        pred = m.predict(Wte) * ysd + ym; predict = m.predict; base = f"{kind}, SHAP"
        A[base] = m.shap(Xex, bg, cfg["shap_n"], seed).sum(1) * ysd           # summed over the lags (Section 4.3)
        if with_ig: A[f"{kind}, IG"] = m.ig(Xex, cfg["ig_steps"]).sum(1) * ysd
        if kind == "RETAIN":
            c, err = m.intrinsic(Xex); A["RETAIN, exact intrinsic"] = c.sum(1) * ysd; extra_row["retain_identity_err"] = err
            ag = pair_agreement(A["RETAIN, exact intrinsic"], A[base])
            extra_row.update(agree_spearman=ag[0], agree_wtau=ag[1], agree_J5=ag[2], agree_S5=ag[3])
            ch = pair_agreement(rng.normal(size=A[base].shape), A[base]); extra_row["agree_spearman_chance"] = ch[0]
        if kind == "TFT": A["TFT, variable selection (unsigned)"] = m.intrinsic(Xex)[0]
    ind = {}
    if with_indicators:
        ni = min(cfg["n_ind"], len(S)); sub = np.arange(ni); todo = {k: v for k, v in A.items() if "unsigned" not in k}
        if with_controls: todo.update({f"control: {k}": v for k, v in make_controls(A[base], P["feats"], rng).items()})
        if "ridge_top10" not in P:
            rm = RidgeModel().fit(XS[I_TR], YS[I_TR]); P["ridge_top10"] = set(np.argsort(-np.abs(rm.m.coef_))[:10])
        for k, v in todo.items():
            ind[k] = indicators(predict, Xex[sub], v[sub], seq, ysd, cfg["n_subsets"], np.random.default_rng(seed))
            ind[k]["recall_at_10"] = len(set(np.argsort(-np.abs(v).mean(0))[:10]) & P["ridge_top10"]) / 10
            if k.startswith("control: "): A[k] = v
    t = tag(dataset, kind, seed)
    pd.DataFrame({"row": I_TE, "forecast": pred}).to_csv(R.dir / "forecasts" / f"{t}.csv", index=False)
    tr = {} if P["truth"] is None else {f"T::{c}": P["truth"][c].values.astype(np.float32) for c in P["truth"].columns}
    np.savez_compressed(R.dir / "attributions" / f"{t}.npz", sample=S, feats=np.array(P["feats"]), dataset=dataset, model=kind,
                        seed=seed, indicators=json.dumps(ind), **{f"A::{k}": v.astype(np.float32) for k, v in A.items()}, **tr)
    mae = float(np.nanmean(np.abs(pred - P["Y"].values[I_TE])))
    R.log(dict(stage="run", dataset=dataset, model=kind, seed=seed, MAE=mae, epochs=getattr(m, "epochs", 0),
               seconds=round(time.time() - t0), params=p, **extra_row))
    R.say(f"{dataset:28s} seed {seed} {kind:11s}: MAE {mae:7.0f} MW  ({time.time() - t0:.0f}s)")

# %% [markdown]
# ## 1.7 Stages

# %%
def stage_diag(R):
    """Is the demand series as hard to forecast as the model errors suggest? Compares the naive forecasts, the
    autocorrelation and the operator forecast, on the whole series and on the test period."""
    wd = R.real(); y = wd.LOAD; P = R.dataset("real"); te = P["I_TE"]; yv = y.values
    def err(pred, idx):
        e = pred[idx] - yv[idx]; ok = ~np.isnan(e); return dict(MAE=float(np.abs(e[ok]).mean()), MAPE=float(100 * np.abs(e[ok] / yv[idx][ok]).mean()))
    rows = {}
    for nm, lag in [("naive, 1 hour earlier", 1), ("naive, 24 hours earlier", 24), ("naive, 48 hours earlier", 48), ("naive, one week earlier", 168)]:
        rows[nm] = err(y.shift(lag).values, te)
    if wd.OPERATOR is not None: rows["operator day-ahead forecast"] = err(wd.OPERATOR.values, te)
    tab = pd.DataFrame(rows).T
    acf = {f"autocorrelation at lag {l} h": float(y.autocorr(l)) for l in (1, 24, 48, 168)}
    loc = wd.IDX.tz_convert(TZ)
    wk = (y - y.shift(168)).abs(); bym = wk.groupby(loc.month).mean().round(0)
    daily = y.groupby(loc.tz_localize(None).normalize()).mean()
    info = dict(hours=int(len(y)), missing_hours=int(y.isna().sum()), mean_MW=float(y.mean()), sd_MW=float(y.std()),
                sd_of_daily_mean_MW=float(daily.std()), sd_of_week_on_week_change_of_daily_mean_MW=float((daily - daily.shift(7)).std()),
                test_start=P["test_start"], **acf)
    tab.round(2).to_csv(R.dir / "tables" / "D1_naive_and_operator_errors_test.csv")
    bym.to_csv(R.dir / "tables" / "D2_weekly_naive_MAE_by_month.csv")
    (R.dir / "tables" / "D3_series_diagnostics.json").write_text(json.dumps(info, indent=2))
    R.say("leakage audit (real): %d perturbations changed later rows only" % leakage_audit(wd, lambda l: build_features(wd, l), 4 if R.a.plan == "smoke" else 20))
    R.say("diagnostics:\n" + tab.round(2).to_string() + "\n" + json.dumps(acf, indent=1))
    R.say("READ THIS: if the weekly naive MAPE is far above ~5% while the 1-hour naive is small, the series itself has large "
          "week-to-week level changes; compare a few days with the operator's published values before blaming the models.")


def stage_runs(R, which):
    cfg = R.cfg; conds = cfg["bench_conds"] or [c["name"] for c in CONDITIONS]
    if which == "real":
        for s in R.seeds:
            for k in R.models: run_one(R, "real", k, s, with_ig=True)
    elif which == "inject":
        lb = lambda l: build_features(R.real(), l, make_injection(R.real())[1])
        if not R.done(stage="audit_injected"):
            leakage_audit(R.real(), lb, 3 if R.a.plan == "smoke" else 10, load=make_injection(R.real())[0]); R.log(dict(stage="audit_injected"))
        for s in R.seeds:
            for k in R.models: run_one(R, "injected", k, s, with_ig=True)
    elif which == "bench":
        for c in conds:
            for s in R.seeds:
                for k in R.models:
                    run_one(R, f"bench:{c}:{s}", k, s, with_indicators=(c == "default"), with_controls=(c == "default" and k == "XGBoost"))
    elif which == "calib":
        for s in R.seeds[:cfg["calib_seeds"]]:
            for k in R.models: run_one(R, f"calib:{s}", k, s)
    elif which == "shortcut":
        for s in R.seeds:
            for k in [m for m in ("TFT", "XGBoost", "Ridge") if m in R.models]:
                for ds in (f"bench:default:{s}", "injected"):
                    run_one(R, ds, k, s, with_indicators=ds.startswith("bench"), with_controls=(ds.startswith("bench") and k == "XGBoost"),
                            with_ig=(ds == "injected"))
                    run_one(R, ds + "|nolags", k, s)


def stage_stats(R):
    """Every table is written to <out>/<plan>/tables and listed in RESULTS.md with the paper table it feeds."""
    TB = R.dir / "tables"; rows = [r for r in R.rows() if r.get("stage") == "run"]
    if not rows: R.say("no finished runs"); return
    LOGDF = pd.DataFrame(rows); notes = []
    def save(df, name, title):
        df.to_csv(TB / name); notes.append(f"### {title}\n`tables/{name}`\n\n{df.round(3).to_string()}\n")
    def fmt(g, cols):
        out = g[cols].agg(["mean", "std"]).round(3); out.columns = [f"{a}_{b}" for a, b in out.columns]; return out
    # ---- explanation scores from the saved attributions
    S, IND = [], []
    for f in sorted((R.dir / "attributions").glob("*.npz")):
        z = np.load(f, allow_pickle=True); ds, kind, seed = str(z["dataset"]), str(z["model"]), int(z["seed"]); feats = list(z["feats"])
        truth = pd.DataFrame({k[3:]: z[k] for k in z.files if k.startswith("T::")})
        base, _, variant = ds.partition("|"); fam = base.split(":")[0]; cond = base.split(":")[1] if fam == "bench" else fam
        for k in [k for k in z.files if k.startswith("A::")]:
            e = k[3:]; A = z[k]; row = dict(dataset=ds, family=fam, condition=cond, variant=variant or "with lags", model=kind, seed=seed, explainer=e)
            if "unsigned" in e:
                if fam in ("bench", "calib"):
                    roles = np.array([role(x) for x in feats]); row["share_irrelevant"] = float(A[:, roles == "irrelevant"].sum() / A.sum())
            elif fam in ("bench", "calib"): row.update(score_bench(A, feats, truth))
            elif fam == "injected": row.update(score_injected(A, feats, truth))
            else: row["share_demand history"] = float(np.abs(A[:, [is_history(x) for x in feats]]).sum() / np.abs(A).sum())
            S.append(row)
        for e, d in json.loads(str(z["indicators"])).items():
            IND.append(dict(dataset=ds, condition=cond, model=kind, seed=seed, explainer=e, **d))
    SC = pd.DataFrame(S); SC.to_csv(TB / "all_scores.csv", index=False)
    INDD = pd.DataFrame(IND)
    if len(INDD): INDD = INDD.merge(SC[["dataset", "seed", "explainer", "magnitude_correctness", "pattern_correctness"]], on=["dataset", "seed", "explainer"], how="left") if "magnitude_correctness" in SC else INDD
    real_expl = ~SC["explainer"].str.startswith("control") & ~SC["explainer"].str.contains("unsigned")

    # ---- A. accuracy on real data: single models, seed ensembles, Diebold-Mariano
    if (LOGDF.dataset == "real").any():
        P = R.dataset("real"); wd = P["wd"]; te = P["I_TE"]; y = P["Y"].values[te]; obs = wd.OBSERVED[te] & ~np.isnan(y)
        ens, tab = {}, {}
        for kind, g in LOGDF[LOGDF.dataset == "real"].groupby("model"):
            F_ = np.stack([pd.read_csv(R.dir / "forecasts" / f"{tag('real', kind, s)}.csv")["forecast"].values for s in sorted(g.seed)])
            maes = np.abs(F_[:, obs] - y[obs]).mean(1); ens[kind] = F_.mean(0); lo, hi = boot_ci(maes)
            tab[kind] = dict(seeds=len(F_), MAE=maes.mean(), MAE_sd=maes.std(ddof=1) if len(F_) > 1 else 0, MAE_ci_low=lo, MAE_ci_high=hi,
                             ensemble_MAE=np.abs(ens[kind][obs] - y[obs]).mean(), RMSE=np.sqrt(((F_[:, obs] - y[obs]) ** 2).mean()),
                             MAPE=100 * np.abs((F_[:, obs] - y[obs]) / y[obs]).mean())
        refs = {"Naive, one week earlier": P["Y"].shift(168).values[te]}
        if wd.OPERATOR is not None: refs["Operator forecast"] = wd.OPERATOR.values[te]
        for k, v in refs.items():
            ok = obs & ~np.isnan(v); tab[k] = dict(MAE=np.abs(v[ok] - y[ok]).mean(), RMSE=np.sqrt(((v[ok] - y[ok]) ** 2).mean()), MAPE=100 * np.abs((v[ok] - y[ok]) / y[ok]).mean())
        T = pd.DataFrame(tab).T; nv = T.loc["Naive, one week earlier", "MAE"]
        T["gain_vs_weekly_naive_pct"] = 100 * (1 - T["MAE"] / nv)
        if "Operator forecast" in T.index: T["ratio_to_operator"] = T["MAE"] / T.loc["Operator forecast", "MAE"]
        save(T.sort_values("MAE"), "A1_accuracy_real.csv", "A1. Accuracy on the real test period (paper Table 3)")
        allp = {**{f"{k} (ensemble)": v for k, v in ens.items()}, **refs}; dm = []
        for a, b in itertools.combinations(allp, 2):
            ok = obs & ~np.isnan(allp[a]) & ~np.isnan(allp[b]); s_, p_ = diebold_mariano(allp[a][ok] - y[ok], allp[b][ok] - y[ok])
            dm.append(dict(model_1=a, model_2=b, DM_stat=s_, p_value=p_))
        save(pd.DataFrame(dm).set_index(["model_1", "model_2"]), "A2_diebold_mariano.csv", "A2. Diebold-Mariano tests (h = 24, HLN correction)")

    # ---- B. correctness on the benchmark, per variant, with intervals over seeds
    for fam, label in [("bench", "synthetic benchmark"), ("calib", "benchmark fitted to real data")]:
        B = SC[(SC.family == fam) & (SC.variant == "with lags") & real_expl]
        if not len(B): continue
        cols = [c for c in B.columns if c.startswith(("r_", "b_"))] + ["magnitude_correctness", "pattern_correctness", "mean_slope",
                "share_direct cause", "share_demand history", "share_consequence", "share_irrelevant"]
        for cond, g in B.groupby("condition"):
            t = fmt(g.groupby("explainer"), cols); ci = g.groupby("explainer")["magnitude_correctness"].apply(lambda v: boot_ci(v.values))
            t["mc_ci_low"], t["mc_ci_high"] = [c[0] for c in ci], [c[1] for c in ci]; t["seeds"] = g.groupby("explainer")["seed"].nunique()
            mae = LOGDF[LOGDF.dataset.str.startswith(f"{fam}:{cond}:" if fam == "bench" else "calib:") & ~LOGDF.dataset.str.contains("nolags")].groupby("model")["MAE"].mean()
            t["model_MAE"] = [mae.get(e.split(",")[0], np.nan) for e in t.index]
            save(t.sort_values("magnitude_correctness_mean", ascending=False), f"B1_correctness_{fam}_{cond}.csv",
                 f"B1. Correctness on the {label}, variant '{cond}' (paper Table 5 uses 'default')")
        D = B[B.condition == ("default" if fam == "bench" else "calib")]
        pv = D.pivot_table(index="seed", columns="explainer", values="magnitude_correctness"); pr = []
        for a, b in itertools.combinations(pv.columns, 2):
            pr.append(dict(a=a, b=b, **paired(pv[a], pv[b])))
        if pr: save(pd.DataFrame(pr).set_index(["a", "b"]), f"B2_paired_{fam}.csv",
                    f"B2. Paired comparisons of magnitude correctness over seeds, {label} (independent seeds; exact Wilcoxon)")
    # ---- C. do truth-free metrics see explanation errors?  (paper Table 4 and the 'faithful is not correct' result)
    if len(INDD):
        mcols = ["magnitude_correctness", "pattern_correctness", "recall_at_10", "deletion", "insertion", "faithfulness_correlation", "relative_infidelity"]
        mcols = [c for c in mcols if c in INDD]
        X_ = INDD[(INDD.model == "XGBoost") & (INDD.condition == "default") & ~INDD.dataset.str.contains("nolags")]
        if len(X_): save(fmt(X_.groupby("explainer"), mcols), "C1_metrics_vs_truth_controls.csv",
                         "C1. TreeSHAP on XGBoost and altered copies with known errors (paper Table 4)")
        M_ = INDD[~INDD.explainer.str.startswith("control") & (INDD.condition == "default") & ~INDD.dataset.str.contains("nolags")]
        if len(M_):
            save(fmt(M_.groupby("explainer"), mcols), "C2_faithful_vs_correct.csv", "C2. Faithfulness indicators next to correctness, by explainer")
            rk = M_.groupby("explainer")[mcols].mean(); sp = {}
            for c in mcols[2:]:
                sgn = -1 if c == "relative_infidelity" else 1
                sp[c] = dict(spearman_with_magnitude_correctness=stats.spearmanr(sgn * rk[c], rk["magnitude_correctness"]).statistic, explainers=len(rk))
            save(pd.DataFrame(sp).T, "C3_metric_ranking_agreement.csv",
                 "C3. Does each indicator rank the explainers as the truth does? (descriptive: few explainers, no test)")
    # ---- D. injected truth on real data
    I_ = SC[(SC.family == "injected") & (SC.variant == "with lags") & real_expl]
    if len(I_):
        cols = ["r_event", "b_event", "r_driver", "b_driver", "decoy_credit", "magnitude_correctness", "share_demand history"]
        t = fmt(I_.groupby("explainer"), cols); ci = I_.groupby("explainer")["magnitude_correctness"].apply(lambda v: boot_ci(v.values))
        t["mc_ci_low"], t["mc_ci_high"] = [c[0] for c in ci], [c[1] for c in ci]
        save(t.sort_values("magnitude_correctness_mean", ascending=False), "D1_injected_truth.csv", "D1. Injected effects on real data (paper Table 6)")
        pv = I_.pivot_table(index="seed", columns="explainer", values="magnitude_correctness")
        pr = [dict(a=a, b=b, **paired(pv[a], pv[b])) for a, b in itertools.combinations(pv.columns, 2)]
        if pr: save(pd.DataFrame(pr).set_index(["a", "b"]), "D2_paired_injected.csv", "D2. Paired comparisons on injected real data")
        bm = SC[(SC.family == "bench") & (SC.condition == "default") & (SC.variant == "with lags") & real_expl].groupby("explainer")["magnitude_correctness"].mean()
        both = pd.DataFrame({"benchmark": bm, "injected_real": I_.groupby("explainer")["magnitude_correctness"].mean()}).dropna()
        if len(both): both["difference"] = both["injected_real"] - both["benchmark"]; save(both.sort_values("benchmark", ascending=False),
                 "D3_benchmark_vs_injected.csv", "D3. Benchmark and injected real data side by side, same protocol and features")
    # ---- E. shortcut: with and without demand history, paired by seed
    for fam, key, cols in [("bench", "default", ["magnitude_correctness", "b_hour", "b_temperature", "b_week", "b_holiday", "share_demand history"]),
                           ("injected", "injected", ["magnitude_correctness", "b_event", "b_driver", "decoy_credit", "share_demand history"])]:
        G = SC[(SC.family == fam) & (SC.condition == key) & real_expl & ~SC.explainer.str.contains("IG|interventional")]
        if not (G.variant == "nolags").any(): continue
        out = []
        for e, g in G.groupby("explainer"):
            w, n = g[g.variant == "with lags"].set_index("seed"), g[g.variant == "nolags"].set_index("seed"); common = w.index.intersection(n.index)
            if not len(common): continue
            base = f"{fam}:{key}:" if fam == "bench" else "injected"; mdl = e.split(",")[0]
            L = LOGDF[(LOGDF.model == mdl) & LOGDF.seed.isin(common)]
            mw = L[L.dataset.str.startswith(base) & ~L.dataset.str.contains("nolags")]["MAE"].mean(); mn = L[L.dataset.str.startswith(base) & L.dataset.str.contains("nolags")]["MAE"].mean()
            row = dict(explainer=e, seeds=len(common), MAE_with=mw, MAE_without=mn)
            for c in cols:
                if c in w: row[f"{c}_with"], row[f"{c}_without"] = w.loc[common, c].mean(), n.loc[common, c].mean()
            row.update({f"mc_{k}": v for k, v in paired(n.loc[common, "magnitude_correctness"], w.loc[common, "magnitude_correctness"]).items()})
            out.append(row)
        if out: save(pd.DataFrame(out).set_index("explainer"), f"E1_shortcut_{fam}.csv",
                     f"E1. Removing demand history, {'benchmark' if fam == 'bench' else 'injected real data'} (paper shortcut table). "
                     "mc_mean_difference = correctness without history minus with history")
    # ---- F. RETAIN: exact contribution against SHAP on the same model, every seed
    RT = LOGDF[(LOGDF.model == "RETAIN") & LOGDF.get("agree_spearman", pd.Series(dtype=float)).notna()] if "agree_spearman" in LOGDF else pd.DataFrame()
    if len(RT):
        RT = RT.assign(family=RT.dataset.str.split(":").str[0])
        save(fmt(RT.groupby("family"), ["agree_spearman", "agree_wtau", "agree_J5", "agree_S5", "agree_spearman_chance", "retain_identity_err"]),
             "F1_retain_agreement.csv", "F1. RETAIN exact contribution vs SHAP on the same model (paper agreement figure), mean and sd over seeds")
        Rb = SC[(SC.model == "RETAIN") & (SC.family == "bench") & real_expl]
        if len(Rb):
            pv = Rb.pivot_table(index=["condition", "seed"], columns="explainer", values="magnitude_correctness")
            if {"RETAIN, SHAP", "RETAIN, exact intrinsic"} <= set(pv.columns):
                save(pd.DataFrame([paired(pv["RETAIN, SHAP"], pv["RETAIN, exact intrinsic"])], index=["SHAP minus exact (all variants pooled; variants are not independent)"]),
                     "F2_retain_correctness.csv", "F2. Which RETAIN explanation is more correct on the benchmark?")
    # ---- G. signatures on real data
    RS = SC[(SC.family == "real") & real_expl]
    if len(RS): save(fmt(RS.groupby("explainer"), ["share_demand history"]), "G1_history_share_real.csv", "G1. Share of credit to demand history on real data")
    head = (f"# Results ({R.a.plan} plan)\n\nRuns finished: {len(LOGDF)}; seeds: {sorted(LOGDF.seed.unique().tolist())}; "
            f"total compute: {LOGDF.seconds.sum() / 3600:.1f} h.\n\nEvery number below is a mean over seeds unless stated. "
            "Intervals are 95% bootstrap intervals over seeds. Nothing was selected using test-set results.\n\n")
    (R.dir / "RESULTS.md").write_text(head + "\n".join(notes))
    R.say(f"tables written to {TB}; summary in {R.dir / 'RESULTS.md'}")

# %% [markdown]
# ## 1.8 Run
#
# To run a subset, change `models` (for example `"Ridge,XGBoost"`) or skip stage cells. Each stage skips the runs that are already finished.

# %%
ARGS = SimpleNamespace(data="data", out="results/deep_pipeline", plan=GLOBAL_SETTINGS["plan"], models=GLOBAL_SETTINGS["models"], seeds=0)
PLANS["deep"].update({k: GLOBAL_SETTINGS[k] for k in ("seeds", "max_epochs", "patience", "n_explain", "tune_trials", "tune_epochs")})
R = Run(ARGS); R.say(f"plan {ARGS.plan} | models {R.models} | seeds {R.seeds} | output {R.dir}")
try:
    import torch; R.say("neural models run on " + (f"GPU ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else "CPU (no GPU found: expect a very long run)"))
except ImportError: pass
(R.dir / "config.json").write_text(json.dumps(dict(plan=ARGS.plan, cfg=R.cfg, models=R.models, inject=INJECT, generator=GEN_DEFAULT,
                                                   split=SPLIT, issue_hour=ISSUE_HOUR), indent=2))

# %% [markdown]
# ### Stage `diag`: Diagnostics of the demand series and leakage audit

# %%
if "diag" in GLOBAL_SETTINGS["stages"]: stage_diag(R)

# %% [markdown]
# ### Stage `tune`: Hyperparameter search on the real validation period

# %%
if "tune" in GLOBAL_SETTINGS["stages"]: stage_tune(R)

# %% [markdown]
# ### Stage `real`: Real data (Section 6.1)

# %%
if "real" in GLOBAL_SETTINGS["stages"]: stage_runs(R, "real")

# %% [markdown]
# ### Stage `inject`: Real data with injected effects (Section 6.5)

# %%
if "inject" in GLOBAL_SETTINGS["stages"]: stage_runs(R, "inject")

# %% [markdown]
# ### Stage `bench`: Synthetic benchmark, all versions (Sections 6.2 to 6.4, 6.6)

# %%
if "bench" in GLOBAL_SETTINGS["stages"]: stage_runs(R, "bench")

# %% [markdown]
# ### Stage `calib`: Benchmark fitted to the real training data

# %%
if "calib" in GLOBAL_SETTINGS["stages"]: stage_runs(R, "calib")

# %% [markdown]
# ### Stage `shortcut`: With and without demand history (Section 6.10)

# %%
if "shortcut" in GLOBAL_SETTINGS["stages"]: stage_runs(R, "shortcut")

# %% [markdown]
# ### Stage `stats`: tables

# %%
if "stats" in GLOBAL_SETTINGS["stages"]:
    stage_stats(R)
    try: show = display
    except NameError: show = print
    for f in sorted((R.dir / "tables").glob("*.csv")):
        if f.name != "all_scores.csv": print("\n" + f.name); show(pd.read_csv(f).round(3))

# %% [markdown]
# ### Figures
#
# Drawn from `tables/all_scores.csv` and the run log, and saved to `results/deep_pipeline/<plan>/figures/`: where each explanation puts its credit, forecast accuracy against explanation correctness, and the effect of the strength of the hidden driver.

# %%
import matplotlib.pyplot as plt
FIG = R.dir / "figures"; FIG.mkdir(exist_ok=True)
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 300, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#8a8a85", "axes.labelcolor": "#3d3d3a", "xtick.color": "#3d3d3a", "ytick.color": "#3d3d3a",
                     "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6, "axes.axisbelow": True})
SC = pd.read_csv(R.dir / "tables" / "all_scores.csv")
LOGDF = pd.DataFrame([r for r in R.rows() if r.get("stage") == "run"])
# one fixed colour per explanation, whatever subset has been run
EXPLAINERS = ["TFT, SHAP", "Transformer, SHAP", "LSTM, SHAP", "RETAIN, SHAP", "RETAIN, exact intrinsic", "XGBoost, TreeSHAP", "Ridge, exact"]
COLOR = dict(zip(EXPLAINERS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]))
MARK = dict(zip(EXPLAINERS, ["o", "s", "^", "D", "v", "P", "X"]))
BD = SC[(SC.family == "bench") & (SC.variant == "with lags") & SC.explainer.isin(EXPLAINERS)]
DEF = BD[BD.condition == "default"]; present = [e for e in EXPLAINERS if e in set(DEF.explainer)]

# (1) where each explanation puts its credit, default benchmark
ROLES = [("share_direct cause", "True causes (calendar, temperature)", "#2a78d6"), ("share_demand history", "Demand history", "#eb6834"),
         ("share_consequence", "Consequences (gas, coal)", "#1baf7a"), ("share_irrelevant", "Irrelevant inputs", "#eda100")]
ROLES = [r for r in ROLES if r[0] in DEF]
if len(DEF) and ROLES:
    sh = DEF.groupby("explainer")[[r[0] for r in ROLES]].mean().reindex(present[::-1])
    fig, ax = plt.subplots(figsize=(7.2, 0.42 * len(sh) + 1.3)); left = np.zeros(len(sh))
    for col, lab, c in ROLES:
        ax.barh(sh.index, sh[col], left=left, color=c, edgecolor="white", linewidth=2, height=0.62, label=lab)
        for y, (l, v) in enumerate(zip(left, sh[col])):
            if v >= 0.07: ax.text(l + v / 2, y, f"{100 * v:.0f}%", ha="center", va="center", color="white", fontsize=8)
        left += sh[col].values
    ax.set_xlim(0, 1); ax.set_xlabel("share of absolute attribution"); ax.grid(False)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{100 * v:.0f}%"))
    ax.legend(ncol=2, frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "fig_credit_shares.png"); plt.show()

# (2) forecast accuracy against explanation correctness, default benchmark
if len(DEF) and "mean_slope" in DEF:
    L = LOGDF[LOGDF.dataset.str.startswith("bench:default:") & ~LOGDF.dataset.str.contains("nolags")]
    mae = L.groupby("model")["MAE"].agg(["mean", "std"]).fillna(0); sl = DEF.groupby("explainer")["mean_slope"].agg(["mean", "std"]).fillna(0)
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    for e in present:
        m = e.split(",")[0]
        ax.errorbar(sl.loc[e, "mean"], mae.loc[m, "mean"], xerr=sl.loc[e, "std"], yerr=mae.loc[m, "std"], fmt=MARK[e], ms=8, color=COLOR[e],
                    ecolor="#8a8a85", elinewidth=0.8, capsize=2, markeredgecolor="white", markeredgewidth=1.2, label=e)
    if {"RETAIN, SHAP", "RETAIN, exact intrinsic"} <= set(present):
        ax.annotate("", xy=(sl.loc["RETAIN, SHAP", "mean"], mae.loc["RETAIN", "mean"]), xytext=(sl.loc["RETAIN, exact intrinsic", "mean"], mae.loc["RETAIN", "mean"]),
                    arrowprops=dict(arrowstyle="->", color="#8a8a85", lw=1))
    ax.axvline(1, color="#8a8a85", lw=0.8, ls="--"); ax.set_xlabel("mean magnitude slope b over the four drivers (1 = full effect credited)")
    ax.set_ylabel("test MAE on the benchmark (MW)"); ax.legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout(); fig.savefig(FIG / "fig_accuracy_vs_correctness.png", bbox_inches="tight"); plt.show()

# (3) strength of the hidden driver
LAT = {"latent_0": 0, "latent_600": 600, "default": 1200, "latent_2400": 2400}
SW = BD[BD.condition.isin(LAT)].assign(hidden=lambda d: d.condition.map(LAT))
PAN = [(c, t) for c, t in [("magnitude_correctness", "(a) Magnitude correctness"), ("share_demand history", "(b) Share to demand history"),
                           ("share_consequence", "(c) Share to gas and coal")] if c in SW]
if SW.hidden.nunique() > 1 and PAN:
    fig, axs = plt.subplots(1, len(PAN), figsize=(3.7 * len(PAN), 3.4), squeeze=False)
    for ax, (col, title) in zip(axs[0], PAN):
        for e in present:
            g = SW[SW.explainer == e].groupby("hidden")[col].agg(["mean", "std"]).fillna(0)
            ax.plot(g.index, g["mean"], marker=MARK[e], ms=5, lw=2, color=COLOR[e], markeredgecolor="white", markeredgewidth=0.8, label=e)
            ax.fill_between(g.index, g["mean"] - g["std"], g["mean"] + g["std"], color=COLOR[e], alpha=0.12, lw=0)
        ax.set_title(title, loc="left", fontsize=9, color="#3d3d3a"); ax.set_xlabel("hidden driver (MW per unit)"); ax.set_xticks(sorted(LAT.values()))
    fig.tight_layout()
    fig.legend(*axs[0][0].get_legend_handles_labels(), frameon=False, fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 0.02))
    fig.savefig(FIG / "fig_hidden_driver.png", bbox_inches="tight"); plt.show()
print("figures saved to", FIG)


# ======================================================================================================================
# ==== PART 2: Part 2. Real-data checks: size check, natural experiments, signatures, benchmark sensitivity
# ======================================================================================================================

# %% [markdown]
# # Part 2. Real-data checks: size check, natural experiments, signatures, benchmark sensitivity
#
# Tests LSTM, Transformer and TFT (explained with SHAP and Integrated Gradients) and XGBoost (TreeSHAP) on the real Spanish data under the operational day-ahead protocol: one forecast issued at 12:00 on D-1 for the 24 hours of day D. Real data have no ground truth, so correctness is tested in several complementary ways.
#
# **In the paper:** Section 6.5 (size check, natural experiments, signatures and benchmark sensitivity). This part trains its own models, with its own random search, because the size check has to be computed while each model is in memory. Its accuracy and injected-effect tables (A and C) repeat Part 1 with these models and are a consistency check. The tables of Part 1 are the ones reported.
#
# | # | Evidence | What it shows | Ground truth? |
# |---|---|---|---|
# | A | **Accuracy** on real data (5 seeds, REE, naive, Diebold-Mariano) | Which model forecasts best | Yes (observed demand) |
# | B | **Size check**: each explanation's group attribution against the change in the forecast when the group is replaced by background values | Whether the explainer reports the size of what the model does | Yes, for the model's behaviour |
# | C | **Injected truth**: an event of known size, a driver with a known nonlinear effect and a correlated decoy with no effect are added to the real demand series | Whether explanations recover known effects in real conditions, and whether the benchmark ranking holds | Yes, for the injected effects |
# | D | **Natural experiments**: holiday effects, the average Sunday drop and the temperature sensitivity, estimated directly from the data | Whether explanations of real drivers have plausible sizes | Approximate (independent estimate) |
# | E | **Signatures**: shares to demand history and to gas and coal, TFT selection weights against SHAP, agreement between seeds, models and explainers | Whether the mechanisms found on the benchmark appear in real data | No |
# | F | **Benchmark sensitivity**: the synthetic benchmark regenerated with two alternative parameter settings | Whether the benchmark ranking depends on generator choices | Yes (synthetic) |
#
# **Hyperparameters.** A random search on the validation period selects the architecture and learning rate of each model once. The selected configuration is then used for every seed and for the injected data. To skip the search, enter the values in `BEST_OVERRIDE`.
#
# Results are written to `results/final_validation/`. "Chapter 8" in code comments and column names means the synthetic benchmark (Section 6.3 of the paper).
#
# **Settings.** The plan (`smoke` or `deep`), the number of seeds and the training length come from the command line (`--plan`, `--seeds`, `--max-epochs`, ...). Everything else is set in the first lines of this part. Finished runs are saved and skipped when the part is run again.

# %%
# The launcher passes the shared settings as GLOBAL_SETTINGS (plan, seeds, training length, explained hours, search size).
import json
from pathlib import Path
print(GLOBAL_SETTINGS)

# %%
# ============================== CONFIGURATION ==============================
from pathlib import Path
DATA_DIR = Path("data")
ENERGY_CSV, WEATHER_CSV = DATA_DIR / "energy_dataset.csv", DATA_DIR / "weather_features.csv"
OUT_DIR = Path("results/final_validation")
TZ = "Europe/Madrid"
ISSUE_HOUR, LAST_KNOWN_HOUR = 12, 11
SPLIT = (0.64, 0.16, 0.20)

RUN_PLAN = "smoke" if GLOBAL_SETTINGS["plan"] == "smoke" else "full"
MODELS = ["LSTM", "Transformer", "TFT", "XGBoost"]
SEEDS = list(range(GLOBAL_SETTINGS["seeds"]))
WINDOW = 24
N_EXPLAIN = GLOBAL_SETTINGS["n_explain"]             # random test hours explained per run (plus holiday or event hours)
SHAP_BACKGROUND, SHAP_NSAMPLES = 300, 200
IG_STEPS = 32                          # Integrated Gradients (second post-hoc explainer)
ABLATION_K = 16                        # background draws for the model-level size check

# hyperparameter search (section 3)
TUNE_TRIALS, TUNE_EPOCHS = GLOBAL_SETTINGS["tune_trials"], GLOBAL_SETTINGS["tune_epochs"]
BEST_OVERRIDE = {}                     # e.g. {"LSTM": dict(hidden=128, layers=2, dropout=0.1, lr=1e-3), ...}
TRAIN = dict(lr=1e-3, weight_decay=1e-4, batch=256, max_epochs=GLOBAL_SETTINGS["max_epochs"], patience=GLOBAL_SETTINGS["patience"], clip=1.0)

# injected ground truth (section 6)
INJECT = dict(seed=2026, n_event_days=60, event_mw=-1500, event_hours=(8, 21),
              driver_mw=900, driver_temp_interaction=0.4, decoy_corr=0.8)

# stages
RUN_REAL, RUN_INJECTED, RUN_SENSITIVITY = True, True, True
SENS_SEEDS = list(range(min(5, GLOBAL_SETTINGS["seeds"])))

ARCH = dict(LSTM=dict(hidden=128, layers=2, dropout=0.1), Transformer=dict(d_model=64, heads=4, layers=2, dropout=0.1),
            TFT=dict(d_model=32, heads=4, dropout=0.1), RETAIN=dict(hidden=128, dropout=0.2))
BENCH_ARCH = {k: dict(v) for k, v in ARCH.items()}      # architectures used in Chapter 8 (for the sensitivity runs)
if RUN_PLAN == "smoke":
    SEEDS, SENS_SEEDS, N_EXPLAIN, SHAP_NSAMPLES, IG_STEPS, ABLATION_K = [0], [0], 80, 30, 8, 4
    TUNE_TRIALS, TUNE_EPOCHS = 2, 2
    TRAIN.update(max_epochs=2, patience=1)
RUN_DIR = OUT_DIR / RUN_PLAN

# %%
import itertools, json, time, warnings, zipfile, copy
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import Ridge
warnings.filterwarnings("ignore")
for d in ("attributions", "forecasts"): (RUN_DIR / d).mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 200, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
try:
    show = display
except NameError:
    show = print
HAVE_XGB = HAVE_SHAP = False
try:
    import xgboost as xgb; HAVE_XGB = True
except ImportError: pass
try:
    import shap; HAVE_SHAP = True
except ImportError: pass
print(f"xgboost {HAVE_XGB} | shap {HAVE_SHAP}")

LOG = RUN_DIR / "log.jsonl"            # one JSON line per finished step (robust to changing columns)
def logged():
    if not LOG.exists(): return []
    return [json.loads(l) for l in LOG.read_text().splitlines() if l.strip()]
def log(row):
    with open(LOG, "a") as f: f.write(json.dumps(row, default=float) + "\n")
def is_done(**key):
    return any(all(r.get(k) == v for k, v in key.items()) for r in logged())

# %% [markdown]
# ## 1. Data and features (operational protocol), as in the real-data notebook

# %%
def load_energy(path):
    df = pd.read_csv(path); df["time"] = pd.to_datetime(df["time"], utc=True)
    return df.drop_duplicates("time").set_index("time").sort_index()

def load_weather(path):
    w = pd.read_csv(path); w["city_name"] = w["city_name"].astype(str).str.strip()
    w["time"] = pd.to_datetime(w["dt_iso"], utc=True); w = w.drop_duplicates(["time", "city_name"])
    w["temp"] = w["temp"] - 273.15
    if "pressure" in w: w.loc[(w["pressure"] < 900) | (w["pressure"] > 1100), "pressure"] = np.nan      # known outliers in this dataset
    if "wind_speed" in w: w.loc[w["wind_speed"] > 50, "wind_speed"] = np.nan
    cols = [c for c in ["temp", "humidity", "pressure", "wind_speed", "clouds_all"] if c in w]
    return w.groupby("time")[cols].mean().sort_index()

def spanish_holidays(years):
    try:
        import holidays as _h
        return set(_h.Spain(years=years).keys())
    except ImportError:
        fixed = [(1, 1), (1, 6), (5, 1), (8, 15), (10, 12), (11, 1), (12, 6), (12, 8), (12, 25)]
        gf = {2015: (4, 3), 2016: (3, 25), 2017: (4, 14), 2018: (3, 30), 2019: (4, 19)}
        out = {pd.Timestamp(y, m, d).date() for y in years for m, d in fixed}
        return out | {pd.Timestamp(y, *gf[y]).date() for y in years if y in gf}

def to_daily(s):
    loc = s.index.tz_convert(TZ)
    df = pd.DataFrame({"v": s.values, "day": loc.tz_localize(None).normalize(), "hour": loc.hour})
    M = df.groupby(["day", "hour"])["v"].mean().unstack("hour").reindex(columns=range(24))
    n = df.groupby("day")["hour"].nunique(); short = n.index[n == 23]
    if len(short): M.loc[short] = M.loc[short].interpolate(axis=1, limit=1, limit_area="inside")
    M.index = pd.DatetimeIndex(M.index); return M

E = load_energy(ENERGY_CSV); W = load_weather(WEATHER_CSV)
IDX = pd.date_range(E.index.min(), E.index.max(), freq="h", tz="UTC")
E, W = E.reindex(IDX), W.reindex(IDX).interpolate(limit=6, limit_area="inside")
GEN = {"gas": "generation fossil gas", "coal": "generation fossil hard coal", "nuclear": "generation nuclear",
       "hydro": "generation hydro water reservoir", "solar": "generation solar", "wind": "generation wind onshore"}
GEN = {k: v for k, v in GEN.items() if v in E}
LOAD = E["total load actual"].interpolate(limit=6, limit_area="inside")
HOLS = spanish_holidays(range(IDX.min().year, IDX.max().year + 1))
print(len(IDX), "hours;", list(W.columns), "| generation:", list(GEN))

# %%
def build_features(load):
    loc = IDX.tz_convert(TZ); day = loc.tz_localize(None).normalize(); h = loc.hour.values
    L = to_daily(load); days = pd.date_range(L.index.min(), L.index.max(), freq="D"); L = L.reindex(days)
    pos = pd.Series(np.arange(len(days)), index=days)
    di = pos.reindex(day).values                                   # day index of each hour
    Lv = L.values
    def day_shift(k): return np.where(di - k >= 0, di - k, -1).astype(int)
    def take(M, k, hour):                                          # M[day - k, hour]
        d = day_shift(k); out = np.full(len(d), np.nan); ok = (d >= 0) & ~np.isnan(di)
        out[ok] = M[d[ok], hour[ok] if np.ndim(hour) else hour]; return out
    X = pd.DataFrame(index=IDX)
    X["hour_sin"], X["hour_cos"] = np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24)
    X["dow_sin"], X["dow_cos"] = np.sin(2 * np.pi * loc.dayofweek / 7), np.cos(2 * np.pi * loc.dayofweek / 7)
    X["doy_sin"], X["doy_cos"] = np.sin(2 * np.pi * loc.dayofyear / 365.25), np.cos(2 * np.pi * loc.dayofyear / 365.25)
    X["weekend"] = (loc.dayofweek >= 5).astype(float)
    X["summer_time"] = np.array([bool(t.dst()) for t in loc], float)
    inh = lambda dd: np.array([x.date() in HOLS for x in dd], float)
    X["holiday"] = inh(day); X["pre_holiday"] = inh(day + pd.Timedelta(days=1)); X["post_holiday"] = inh(day - pd.Timedelta(days=1))
    X["bridge"] = (((loc.dayofweek == 0) & (X["pre_holiday"] == 1)) | ((loc.dayofweek == 4) & (X["post_holiday"] == 1))).astype(float)
    X["christmas"] = (((loc.month == 12) & (loc.day >= 24)) | ((loc.month == 1) & (loc.day <= 6))).astype(float)
    T = W["temp"]
    X["temp"] = T.values; X["hdd"], X["cdd"] = np.clip(18 - T, 0, None).values, np.clip(T - 18, 0, None).values
    Td = to_daily(T).reindex(days)
    for f in ("mean", "max", "min"):
        X[f"temp_day_{f}"] = getattr(Td, f)(axis=1).reindex(day).values
    X["temp_ewm24"], X["temp_ewm72"] = T.ewm(halflife=24).mean().values, T.ewm(halflife=72).mean().values
    for c in [c for c in W.columns if c != "temp"]:
        X[c] = W[c].values
    for c, nm in [("forecast solar day ahead", "solar_fc"), ("forecast wind onshore day ahead", "wind_fc")]:
        if c in E: X[nm] = E[c].interpolate(limit=6, limit_area="inside").values
    # demand history relative to the issue time (12:00 on D-1; last observation 11:00 on D-1)
    LK = LAST_KNOWN_HOUR
    X["load_last_obs"] = take(Lv, 1, LK)
    X["load_change_24h"] = take(Lv, 1, LK) - take(Lv, 2, LK)
    win24 = np.full((len(days), 24), np.nan)                       # 24 h before issue of day d: D-2 12..23 + D-1 0..11
    win24[2:, :24 - LK - 1] = Lv[:-2, LK + 1:]; win24[2:, 24 - LK - 1:] = Lv[1:-1, :LK + 1]
    for f in ("mean", "max", "min", "std"):
        X[f"load_24h_{f}"] = getattr(np, "nan" + f)(win24, axis=1)[np.where(np.isnan(di), 0, di).astype(int)]
    wk = np.full(len(days), np.nan)
    for d in range(8, len(days)):
        wk[d] = np.nanmean(np.r_[Lv[d - 8, LK + 1:], Lv[d - 7:d - 1].ravel(), Lv[d - 1, :LK + 1]])
    X["load_week_mean"] = wk[np.where(np.isnan(di), 0, di).astype(int)]
    for f in ("mean", "max", "min"):
        X[f"load_D2_{f}"] = take(getattr(np, "nan" + f)(Lv, axis=1)[:, None], 2, 0)
    X["load_same_hour_7d"], X["load_same_hour_14d"] = take(Lv, 7, h), take(Lv, 14, h)
    X["load_same_hour_latest"] = np.where(h <= LK, take(Lv, 1, h), take(Lv, 2, h))
    X["latest_is_D1"] = (h <= LK).astype(float)
    for k, c in GEN.items():
        G = to_daily(E[c].interpolate(limit=6, limit_area="inside")).reindex(days).values
        X[f"gen_{k}_at_issue"] = take(G, 1, LK)
    X.loc[np.isnan(di)] = np.nan
    return X

GROUPS = {"Hour of day": ["hour_sin", "hour_cos"], "Weekday": ["dow_sin", "dow_cos", "weekend"],
          "Holiday and special days": ["holiday", "pre_holiday", "post_holiday", "bridge", "christmas"],
          "Season": ["doy_sin", "doy_cos", "summer_time"],
          "Temperature": ["temp", "hdd", "cdd", "temp_day_mean", "temp_day_max", "temp_day_min", "temp_ewm24", "temp_ewm72"],
          "Other weather": ["humidity", "pressure", "wind_speed", "clouds_all"],
          "Renewable forecasts": ["solar_fc", "wind_fc"],
          "Gas and coal (consequences)": ["gen_gas_at_issue", "gen_coal_at_issue"],
          "Other generation": ["gen_nuclear_at_issue", "gen_hydro_at_issue", "gen_solar_at_issue", "gen_wind_at_issue"]}
def group_of(f):
    if f.startswith("load_") or f == "latest_is_D1": return "Demand history"
    return next((g for g, v in GROUPS.items() if f in v), "Other")

X_ALL = build_features(LOAD); Y_ALL = LOAD.copy()
FEATS = list(X_ALL.columns); FGROUP = pd.Series([group_of(f) for f in FEATS], index=FEATS)
print(len(FEATS), "features:", FGROUP.value_counts().to_dict())

# %%
# Leakage audit (original data)
issue_time = IDX.tz_convert(TZ).tz_localize(None).normalize() - pd.Timedelta(days=1) + pd.Timedelta(hours=ISSUE_HOUR)
def leakage_audit(load, builder, n):
    base = builder(load); rng = np.random.default_rng(0); viol, nch = [], 0
    okpos = np.flatnonzero(load.notna().values)
    for j in rng.choice(okpos[len(okpos) // 10:-24 * 30], n, replace=False):
        l2 = load.copy(); l2.iloc[j] += 1e5; X2 = builder(l2)
        ch = ~((X2.values == base.values) | (np.isnan(X2.values) & np.isnan(base.values))).all(axis=1); nch += ch.any()
        t_obs = IDX[j].tz_convert(TZ).tz_localize(None); bad = ch & (issue_time <= t_obs)
        if bad.any(): viol.append(f"demand at {t_obs} changes rows issued at {issue_time[bad][0]}")
    assert nch > 0
    if viol: raise RuntimeError("LEAKAGE:\n" + "\n".join(viol))
    return nch
print("leakage audit passed:", leakage_audit(LOAD, build_features, 6 if RUN_PLAN == "smoke" else 20), "perturbations")

# %% [markdown]
# ## 2. Injected ground truth (semi-synthetic real data)
#
# Three components are added to the **real** data; everything else stays real.
#
# * **Event**: on `n_event_days` randomly chosen non-holiday weekdays, demand is lowered by exactly `event_mw` between `event_hours`. The model sees an `event_flag` known in advance, like a holiday. True contribution: *C*_event.
# * **Driver**: an exogenous index *z* (smooth daily process, known at issue time like a forecast) with a nonlinear effect that grows with heat: *C*_z = `driver_mw` · tanh(1.2 *z*) · (1 + `driver_temp_interaction` · cooling degrees / 5). True contribution: *C*_z, credited to the driver.
# * **Decoy**: a variable correlated with *z* (correlation `decoy_corr`) with **no effect**. Any attribution to it is an error.
#
# The same injected series is used for every model and seed.

# %%
def make_injection(cfg):
    rng = np.random.default_rng(cfg["seed"])
    loc = IDX.tz_convert(TZ); day = loc.tz_localize(None).normalize()
    days = pd.DatetimeIndex(sorted(set(day)))
    cand = [d for d in days[30:-2] if d.dayofweek < 5 and d.date() not in HOLS]
    ev_days = set(pd.DatetimeIndex(rng.choice(cand, cfg["n_event_days"], replace=False)))
    h0, h1 = cfg["event_hours"]
    flag = np.array([(d in ev_days) and (h0 <= hh <= h1) for d, hh in zip(day, loc.hour)], float)
    zd = np.zeros(len(days)); e = rng.normal(0, np.sqrt(1 - 0.9 ** 2), len(days))
    for i in range(1, len(days)): zd[i] = 0.9 * zd[i - 1] + e[i]
    di = pd.Series(np.arange(len(days)), index=days).reindex(day).values
    z = np.interp(di + loc.hour / 24, np.arange(len(days)), zd) + rng.normal(0, 0.05, len(IDX))
    decoy = cfg["decoy_corr"] * z + np.sqrt(1 - cfg["decoy_corr"] ** 2) * rng.normal(0, 1, len(IDX))
    cdd = np.clip(W["temp"].values - 18, 0, None)
    C_event = cfg["event_mw"] * flag
    C_z = cfg["driver_mw"] * np.tanh(1.2 * z) * (1 + cfg["driver_temp_interaction"] * np.nan_to_num(cdd) / 5)
    truth = pd.DataFrame({"C_event": C_event, "C_z": C_z}, index=IDX)
    extra = pd.DataFrame({"event_flag": flag, "driver_z": z, "decoy_d": decoy}, index=IDX)
    return LOAD + C_event + C_z, extra, truth, ev_days

LOAD_INJ, EXTRA_INJ, TRUTH_INJ, EVENT_DAYS = make_injection(INJECT)
def build_features_injected(load):
    X = build_features(load)
    for c in EXTRA_INJ.columns: X[c] = EXTRA_INJ[c].values
    return X
GROUPS.update({"Injected event": ["event_flag"], "Injected driver": ["driver_z"], "Injected decoy": ["decoy_d"]})
print(f"{len(EVENT_DAYS)} event days; driver effect sd {TRUTH_INJ['C_z'].std():.0f} MW; "
      f"corr(driver, decoy) = {np.corrcoef(EXTRA_INJ['driver_z'], EXTRA_INJ['decoy_d'])[0, 1]:.2f}")
print("leakage audit (injected):", leakage_audit(LOAD_INJ, build_features_injected, 4 if RUN_PLAN == "smoke" else 10), "perturbations")

# %% [markdown]
# ## 3. Datasets, split, windows

# %%
loc_day = IDX.tz_convert(TZ).tz_localize(None).normalize()
_X0 = build_features(LOAD)
_valid0 = _X0.notna().all(axis=1).values & LOAD.notna().values
_dv = pd.DatetimeIndex(sorted(set(loc_day[_valid0]))); _n = len(_dv)
VAL_START, TEST_START = _dv[int(SPLIT[0] * _n)], _dv[int((SPLIT[0] + SPLIT[1]) * _n)]
PART = np.where(loc_day < VAL_START, 0, np.where(loc_day < TEST_START, 1, 2))

def prepare(X, Y, extra_mask, seed=99):
    valid = X.notna().all(axis=1).values & Y.notna().values
    vr = pd.Series(valid.astype(int)).rolling(WINDOW).sum().fillna(0).values == WINDOW
    I = [np.flatnonzero(vr & (PART == k)) for k in range(3)]
    mu, sd = np.nanmean(X.values[I[0]], 0), np.nanstd(X.values[I[0]], 0) + 1e-9
    ym, ysd = np.nanmean(Y.values[I[0]]), np.nanstd(Y.values[I[0]])
    XS = np.nan_to_num(((X.values - mu) / sd).astype(np.float32)); YS = ((Y.values - ym) / ysd).astype(np.float32)
    rand = np.random.default_rng(seed).choice(I[2], min(N_EXPLAIN, len(I[2])), replace=False)
    extra = np.flatnonzero(extra_mask & vr & (PART == 2))
    sample = np.unique(np.r_[rand, extra])
    feats = list(X.columns)
    return dict(X=X, Y=Y, XS=XS, YS=YS, ym=ym, ysd=ysd, I_TR=I[0], I_VA=I[1], I_TE=I[2], SAMPLE=sample,
                feats=feats, group=np.array([group_of(f) for f in feats]), n_rand=len(rand))

loc_all = IDX.tz_convert(TZ)
hol_wd = (_X0["holiday"].values == 1) & (loc_all.dayofweek < 5)
DATA = {"real": prepare(_X0, LOAD, hol_wd)}
_XI = build_features_injected(LOAD_INJ)
DATA["injected"] = prepare(_XI, LOAD_INJ, EXTRA_INJ["event_flag"].values == 1)
for k, P in DATA.items():
    print(f"{k:9s}: {len(P['feats'])} features, windows {len(P['I_TR'])}/{len(P['I_VA'])}/{len(P['I_TE'])}, explained {len(P['SAMPLE'])}")
print(f"split: val from {VAL_START.date()}, test from {TEST_START.date()}")

# %% [markdown]
# ## 4. Models and explainers

# %%
import torch, torch.nn as nn
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", DEVICE, "| torch", torch.__version__, "| shap", shap.__version__ if HAVE_SHAP else "missing")
assert HAVE_SHAP, "pip install shap"

class LSTMNet(nn.Module):
    def __init__(self, F, hidden, layers, dropout):
        super().__init__()
        self.rnn = nn.LSTM(F, hidden, layers, batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(self, x):
        h, _ = self.rnn(x); return self.head(h[:, -1]).squeeze(-1)

class RETAINNet(nn.Module):
    def __init__(self, F, hidden, dropout):
        super().__init__()
        self.emb = nn.Linear(F, hidden)
        self.gru_a = nn.GRU(hidden, hidden, batch_first=True); self.gru_b = nn.GRU(hidden, hidden, batch_first=True)
        self.att_a = nn.Linear(hidden, 1); self.att_b = nn.Linear(hidden, hidden)
        self.drop = nn.Dropout(dropout); self.out = nn.Linear(hidden, 1)
    def attention(self, x):
        v = self.emb(x)
        rv = torch.flip(self.drop(v), [1])                       # reverse time
        g, _ = self.gru_a(rv); h, _ = self.gru_b(rv)
        g, h = torch.flip(g, [1]), torch.flip(h, [1])
        alpha = torch.softmax(self.att_a(g).squeeze(-1), dim=1)   # (n, T)
        beta = torch.tanh(self.att_b(h))                         # (n, T, H)
        return v, alpha, beta
    def forward(self, x):
        v, a, b = self.attention(x)
        return self.out((a.unsqueeze(-1) * b * v).sum(1)).squeeze(-1)
    def contributions(self, x):
        """Exact contribution (n, T, F) and the feature-independent bias (n,)."""
        v, a, b = self.attention(x)
        w, W = self.out.weight.squeeze(0), self.emb.weight          # (H,), (H, F)
        coef = torch.einsum("nt,nth,h,hf->ntf", a, b, w, W)
        bias = ((a.unsqueeze(-1) * b * self.emb.bias).sum(1) * w).sum(-1) + self.out.bias
        return x * coef, bias

class PosEnc(nn.Module):
    def __init__(self, d, T=512):
        super().__init__()
        pe = torch.zeros(T, d); pos = torch.arange(T).unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2) * (-np.log(10000.0) / d))
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer("pe", pe)
    def forward(self, x): return x + self.pe[: x.shape[1]]

class EncLayer(nn.Module):
    def __init__(self, d, heads, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.ReLU(), nn.Linear(4 * d, d))
        self.n1, self.n2, self.drop = nn.LayerNorm(d), nn.LayerNorm(d), nn.Dropout(dropout)
        self.last_w = None
    def forward(self, x, keep=False):
        a, w = self.attn(x, x, x, need_weights=keep, average_attn_weights=True)
        if keep: self.last_w = w                                  # (n, T, T)
        x = self.n1(x + self.drop(a)); return self.n2(x + self.drop(self.ff(x)))

class TransformerNet(nn.Module):
    def __init__(self, F, d_model, heads, layers, dropout):
        super().__init__()
        self.inp, self.pos = nn.Linear(F, d_model), PosEnc(d_model)
        self.layers = nn.ModuleList([EncLayer(d_model, heads, dropout) for _ in range(layers)])
        self.head = nn.Linear(d_model, 1)
    def forward(self, x, keep=False):
        z = self.pos(self.inp(x))
        for L in self.layers: z = L(z, keep)
        return self.head(z[:, -1]).squeeze(-1)
    def attention_score(self, x):
        """Section 4.2.2: attention of the last position on each step x |input|, summed over time (unsigned)."""
        self.forward(x, keep=True)
        w = self.layers[-1].last_w[:, -1, :]                      # (n, T)
        return (w.unsqueeze(-1) * x.abs()).sum(1)

class GLU(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__(); self.l = nn.Linear(d_in, 2 * d_out)
    def forward(self, x):
        a, b = self.l(x).chunk(2, dim=-1); return a * torch.sigmoid(b)

class GRN(nn.Module):
    """Gated residual network of the TFT."""
    def __init__(self, d_in, d_hid, d_out, dropout):
        super().__init__()
        self.skip = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()
        self.l1, self.l2 = nn.Linear(d_in, d_hid), nn.Linear(d_hid, d_hid)
        self.drop, self.glu, self.norm = nn.Dropout(dropout), GLU(d_hid, d_out), nn.LayerNorm(d_out)
    def forward(self, a):
        h = self.drop(self.l2(nn.functional.elu(self.l1(a))))
        return self.norm(self.skip(a) + self.glu(h))

class InterpretableMHA(nn.Module):
    """TFT interpretable multi-head attention: per-head queries/keys, shared values, heads averaged."""
    def __init__(self, d, heads, dropout):
        super().__init__()
        self.dk = d // heads
        self.q = nn.ModuleList([nn.Linear(d, self.dk) for _ in range(heads)])
        self.k = nn.ModuleList([nn.Linear(d, self.dk) for _ in range(heads)])
        self.v, self.o, self.drop = nn.Linear(d, self.dk), nn.Linear(self.dk, d), nn.Dropout(dropout)
    def forward(self, q, k):
        v, outs, atts = self.v(k), [], []
        for qi, ki in zip(self.q, self.k):
            a = torch.softmax(qi(q) @ ki(k).transpose(1, 2) / self.dk ** 0.5, dim=-1)
            atts.append(a); outs.append(self.drop(a) @ v)
        return self.o(torch.stack(outs).mean(0)), torch.stack(atts).mean(0)

class TFTNet(nn.Module):
    """Temporal Fusion Transformer (Lim et al., 2021), single-horizon version for one target hour:
    variable selection network -> LSTM encoder with gated skip -> enrichment GRN ->
    interpretable attention from the last step -> gated position-wise GRN -> output.
    Simplifications: no static covariates; one GRN shared by all variables (the original uses one per variable)."""
    def __init__(self, F, d_model, heads, dropout):
        super().__init__()
        d = d_model
        self.emb_w, self.emb_b = nn.Parameter(torch.randn(F, d) * 0.1), nn.Parameter(torch.zeros(F, d))
        self.vsn, self.var_grn = GRN(F * d, d, F, dropout), GRN(d, d, d, dropout)
        self.lstm = nn.LSTM(d, d, batch_first=True)
        self.gate1, self.norm1 = GLU(d, d), nn.LayerNorm(d)
        self.enrich = GRN(d, d, d, dropout)
        self.attn = InterpretableMHA(d, heads, dropout)
        self.gate2, self.norm2 = GLU(d, d), nn.LayerNorm(d)
        self.ff, self.gate3, self.norm3 = GRN(d, d, d, dropout), GLU(d, d), nn.LayerNorm(d)
        self.out = nn.Linear(d, 1)
        self.last_w = None
    def select(self, x):
        e = x.unsqueeze(-1) * self.emb_w + self.emb_b                         # (n, T, F, d)
        n, T, F_, d = e.shape
        w = torch.softmax(self.vsn(e.reshape(n, T, F_ * d)), dim=-1)           # (n, T, F) selection weights
        return (w.unsqueeze(-1) * self.var_grn(e)).sum(2), w
    def forward(self, x, keep=False):
        z, w = self.select(x)
        h, _ = self.lstm(z)
        h = self.enrich(self.norm1(z + self.gate1(h)))
        last = h[:, -1:]
        a, _ = self.attn(last, h)
        y = self.norm2(last + self.gate2(a))
        y = self.norm3(y + self.gate3(self.ff(y)))
        if keep: self.last_w = w
        return self.out(y[:, 0]).squeeze(-1)
    def selection_weights(self, x):
        """TFT's built-in importance: variable-selection weights averaged over the window (unsigned, sum to 1)."""
        self.forward(x, keep=True); return self.last_w.mean(1)

def build_net(kind, F):
    a = ARCH[kind]
    if kind == "LSTM": return LSTMNet(F, a["hidden"], a["layers"], a["dropout"])
    if kind == "RETAIN": return RETAINNet(F, a["hidden"], a["dropout"])
    if kind == "TFT": return TFTNet(F, a["d_model"], a["heads"], a["dropout"])
    return TransformerNet(F, a["d_model"], a["heads"], a["layers"], a["dropout"])

class Out2D(nn.Module):
    def __init__(self, net): super().__init__(); self.net = net
    def forward(self, x): return self.net(x).unsqueeze(-1)

class SeqModel:
    """Train / predict / explain one network on standardized windows (n, T, F)."""
    def __init__(self, kind, F, seed):
        torch.manual_seed(seed); np.random.seed(seed)
        self.kind, self.net = kind, build_net(kind, F).to(DEVICE)
    def _t(self, a): return torch.as_tensor(a, dtype=torch.float32, device=DEVICE)
    def fit(self, Xtr, ytr, Xva, yva):
        opt = torch.optim.AdamW(self.net.parameters(), lr=TRAIN["lr"], weight_decay=TRAIN["weight_decay"])
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)
        loss_fn = nn.HuberLoss(); best, best_state, bad = np.inf, None, 0
        Xtr_t, ytr_t = self._t(Xtr), self._t(ytr); n = len(Xtr)
        self.epochs = 0
        for ep in range(TRAIN["max_epochs"]):
            self.net.train(); perm = torch.randperm(n, device=DEVICE)
            for i in range(0, n, TRAIN["batch"]):
                idx = perm[i:i + TRAIN["batch"]]
                opt.zero_grad(); loss = loss_fn(self.net(Xtr_t[idx]), ytr_t[idx]); loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), TRAIN["clip"]); opt.step()
            val = float(np.mean(np.abs(self.predict(Xva) - yva)))
            sched.step(val); self.epochs = ep + 1
            if val < best - 1e-5:
                best, bad = val, 0; best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
            else:
                bad += 1
                if bad >= TRAIN["patience"]: break
        self.net.load_state_dict(best_state); self.net.eval(); return self
    @torch.no_grad()
    def predict(self, Xw, bs=2048):
        self.net.eval()
        return np.concatenate([self.net(self._t(Xw[i:i + bs])).cpu().numpy() for i in range(0, len(Xw), bs)])
    def shap(self, Xw, background, seed=0):
        self.net.eval()
        wrapped = Out2D(self.net).eval()                          # shap expects (n, 1) outputs
        with torch.backends.cudnn.flags(enabled=False):          # RNN backward in eval mode on GPU
            ex = shap.GradientExplainer(wrapped, self._t(background))
            sv = ex.shap_values(self._t(Xw), nsamples=SHAP_NSAMPLES, rseed=seed)
        sv = np.asarray(sv[0] if isinstance(sv, list) else sv)
        return sv.reshape(len(Xw), Xw.shape[1], Xw.shape[2])     # (n, T, F)
    @torch.no_grad()
    def intrinsic(self, Xw):
        self.net.eval(); x = self._t(Xw)
        if self.kind == "RETAIN":
            c, bias = self.net.contributions(x)
            err = (c.sum((1, 2)) + bias - self.net(x)).abs().max().item()
            assert err < 1e-3, f"RETAIN identity violated ({err:.2e})"
            return c.cpu().numpy(), err
        if self.kind == "Transformer":
            return self.net.attention_score(x).cpu().numpy(), None
        if self.kind == "TFT":
            return self.net.selection_weights(x).cpu().numpy(), None
        return None, None

# %%
def windows(Xs, idx, T=WINDOW):
    return np.stack([Xs[i - T + 1:i + 1] for i in idx]).astype(np.float32)

def seq_deletion_insertion(model, Xw, A, K=10):
    n, T, F = Xw.shape; order = np.argsort(-np.abs(A), axis=1)
    f0 = model.predict(Xw); scale = max(np.std(f0), 1e-9); dele, ins = [], []
    for m in range(1, min(K, F) + 1):
        M = Xw.copy()
        for i in range(n): M[i, :, order[i, :m]] = 0.0
        dele.append(np.mean(np.abs(f0 - model.predict(M))))
        M = np.zeros_like(Xw)
        for i in range(n): M[i, :, order[i, :m]] = Xw[i, :, order[i, :m]]
        ins.append(np.mean(np.abs(model.predict(M) - f0)))
    return float(np.mean(dele) / scale), float(1 - np.mean(ins) / scale)

def pair_agreement(a, b, k=5):
    """Agreement of two explanations of the same forecasts (Section 4.3)."""
    out = []
    for x, y in zip(a, b):
        ta, tb = np.argsort(-np.abs(x))[:k], np.argsort(-np.abs(y))[:k]
        out.append((stats.spearmanr(np.abs(x), np.abs(y)).statistic, stats.weightedtau(np.abs(x), np.abs(y)).statistic,
                    len(set(ta) & set(tb)) / len(set(ta) | set(tb)), np.mean(np.sign(x[tb]) == np.sign(y[tb]))))
    return np.nanmean(np.array(out), axis=0)

# %%
def _ig(self, Xw, steps=IG_STEPS, bs=64):
    """Integrated Gradients from the training mean (0 after standardization), per time step and feature."""
    self.net.eval(); out = []
    with torch.backends.cudnn.flags(enabled=False):
        for i in range(0, len(Xw), bs):
            x = self._t(Xw[i:i + bs]); total = torch.zeros_like(x)
            for a in torch.arange(1, steps + 1, device=DEVICE, dtype=torch.float32) / steps:
                xi = (a * x).detach().requires_grad_(True)
                g, = torch.autograd.grad(self.net(xi).sum(), xi); total += g
            out.append((total / steps * x).detach().cpu().numpy())
    return np.concatenate(out)
SeqModel.ig = _ig

class XGBRef:
    def __init__(self, seed, p):
        self.m = xgb.XGBRegressor(n_estimators=3000 if RUN_PLAN != "smoke" else 200, max_depth=p.get("max_depth", 6),
                                  learning_rate=p.get("lr", 0.03), min_child_weight=p.get("min_child_weight", 5),
                                  subsample=0.8, colsample_bytree=p.get("colsample", 0.8), early_stopping_rounds=100,
                                  n_jobs=-1, random_state=seed, verbosity=0)
    def fit(self, Xtr, ytr, Xva, yva):
        self.m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False); self.ex = shap.TreeExplainer(self.m); return self
    def predict(self, X): return self.m.predict(X)
    def shap(self, X): return np.asarray(self.ex.shap_values(X))

ABL_GROUPS = ["Demand history", "Hour of day", "Weekday", "Holiday and special days", "Season", "Temperature",
              "Other weather", "Renewable forecasts", "Gas and coal (consequences)", "Other generation"]
def group_ablation(predict, Xex, Xbg, group, seq):
    """Model-level effect of each feature group: f(x) minus the mean forecast when the group's inputs are replaced
    by values from random background (training) inputs. Returns (n, groups) in standardized target units."""
    f0 = predict(Xex); rng = np.random.default_rng(0); out = {}
    for g in ABL_GROUPS:
        cols = np.flatnonzero(group == g)
        if not len(cols): continue
        acc = np.zeros(len(Xex))
        for b in rng.choice(len(Xbg), ABLATION_K, replace=False):
            M = Xex.copy()
            if seq: M[:, :, cols] = Xbg[b][:, cols]
            else:   M[:, cols] = Xbg[b][cols]
            acc += predict(M)
        out[g] = f0 - acc / ABLATION_K
    return out

# %% [markdown]
# ## 5. Hyperparameter search (once, on the real validation period)
#
# Random search with `TUNE_TRIALS` candidates per model, each trained with seed 0 for at most `TUNE_EPOCHS` epochs, scored by validation MAE. The best configuration is saved to `best_params.json` and used for every later run (all seeds, real and injected data). `BEST_OVERRIDE` replaces the search for any model.

# %%
SPACE = {
    "LSTM": dict(hidden=[64, 128, 256], layers=[1, 2], dropout=[0.1, 0.2, 0.3], lr=[3e-4, 1e-3, 3e-3]),
    "Transformer": dict(d_model=[32, 64, 128], heads=[4], layers=[1, 2, 3], dropout=[0.1, 0.2], lr=[3e-4, 1e-3, 3e-3]),
    "TFT": dict(d_model=[16, 32, 64], heads=[2, 4], dropout=[0.1, 0.2], lr=[3e-4, 1e-3, 3e-3]),
    "XGBoost": dict(max_depth=[4, 6, 8], lr=[0.03, 0.05], min_child_weight=[1, 5], colsample=[0.6, 0.8]),
}
BEST_FILE = RUN_DIR / "best_params.json"
BEST = json.loads(BEST_FILE.read_text()) if BEST_FILE.exists() else {}
P = DATA["real"]
def val_mae(pred_s, idx): return float(np.nanmean(np.abs(pred_s * P["ysd"] + P["ym"] - P["Y"].values[idx])))
for kind in MODELS:
    if kind in BEST_OVERRIDE: BEST[kind] = BEST_OVERRIDE[kind]; continue
    if kind in BEST: continue
    if kind == "XGBoost" and not (HAVE_XGB and HAVE_SHAP): continue
    rng = np.random.default_rng(1)
    TRIAL_FILE = RUN_DIR / f"tuning_{kind}_trials.jsonl"     # one line per finished trial: an interrupted search resumes here
    done = [json.loads(l) for l in TRIAL_FILE.read_text().splitlines() if l.strip()] if TRIAL_FILE.exists() else []
    trials = [r["row"] for r in done]
    if done: print(f"tune {kind}: {len(done)} finished trials loaded from {TRIAL_FILE}")
    for t in range(TUNE_TRIALS):
        cand = {k: v[rng.integers(len(v))] for k, v in SPACE[kind].items()}   # drawn in the same order on every run
        if t < len(done): continue
        t0 = time.time()
        if kind == "XGBoost":
            m = XGBRef(0, cand).fit(P["XS"][P["I_TR"]], P["YS"][P["I_TR"]], P["XS"][P["I_VA"]], P["YS"][P["I_VA"]])
            score = val_mae(m.predict(P["XS"][P["I_VA"]]), P["I_VA"])
        else:
            ARCH[kind] = {k: v for k, v in cand.items() if k != "lr"}
            saved = dict(TRAIN); TRAIN.update(lr=cand["lr"], max_epochs=TUNE_EPOCHS, patience=3)
            Wtr, Wva = windows(P["XS"], P["I_TR"]), windows(P["XS"], P["I_VA"])
            m = SeqModel(kind, len(P["feats"]), 0).fit(Wtr, P["YS"][P["I_TR"]], Wva, P["YS"][P["I_VA"]])
            score = val_mae(m.predict(Wva), P["I_VA"]); TRAIN.clear(); TRAIN.update(saved)
        trials.append(dict(**cand, val_MAE=score)); print(f"tune {kind:11s} {cand} -> {score:.0f} MW ({time.time() - t0:.0f}s)")
        with open(TRIAL_FILE, "a") as f: f.write(json.dumps(dict(trial=t, row=trials[-1]), default=float) + "\n")
    pd.DataFrame(trials).to_csv(RUN_DIR / f"tuning_{kind}.csv", index=False)
    BEST[kind] = {k: v for k, v in min(trials, key=lambda r: r["val_MAE"]).items() if k != "val_MAE"}
    BEST_FILE.write_text(json.dumps(BEST, indent=2, default=float))
print("selected configurations:"); show(pd.DataFrame(BEST).T)

# %% [markdown]
# ## 6. Train, forecast and explain: real data and injected data (resumable)

# %%
def run_one(dataset, kind, seed, ablation):
    P = DATA[dataset]; F = len(P["feats"]); t0 = time.time()
    XS, YS, ysd, ym = P["XS"], P["YS"], P["ysd"], P["ym"]
    I_TR, I_VA, I_TE, S = P["I_TR"], P["I_VA"], P["I_TE"], P["SAMPLE"]
    A, abl = {}, {}
    if kind == "XGBoost":
        m = XGBRef(seed, BEST.get("XGBoost", {})).fit(XS[I_TR], YS[I_TR], XS[I_VA], YS[I_VA]); epochs = int(m.m.best_iteration)
        pred = m.predict(XS[I_TE]) * ysd + ym
        A["XGBoost, TreeSHAP"] = m.shap(XS[S]) * ysd
        if ablation:
            bg = XS[I_TR][np.random.default_rng(seed).choice(len(I_TR), 2000, replace=False)]
            abl = group_ablation(m.predict, XS[S], bg, P["group"], seq=False)
    else:
        ARCH[kind] = {k: v for k, v in BEST[kind].items() if k != "lr"}
        TRAIN["lr"] = BEST[kind].get("lr", 1e-3)
        Wtr, Wva, Wte, Wex = windows(XS, I_TR), windows(XS, I_VA), windows(XS, I_TE), windows(XS, S)
        bg = Wtr[np.random.default_rng(seed).choice(len(Wtr), SHAP_BACKGROUND, replace=False)]
        m = SeqModel(kind, F, seed).fit(Wtr, YS[I_TR], Wva, YS[I_VA]); epochs = m.epochs
        pred = m.predict(Wte) * ysd + ym
        A[f"{kind}, SHAP"] = m.shap(Wex, bg, seed).sum(1) * ysd
        A[f"{kind}, IG"] = m.ig(Wex).sum(1) * ysd
        if kind == "TFT": A["TFT, variable selection (unsigned)"] = m.intrinsic(Wex)[0]
        if ablation: abl = group_ablation(m.predict, Wex, bg, P["group"], seq=True)
    tag = f"{dataset}_{kind}_seed{seed}"
    pd.DataFrame({"row": I_TE, "forecast": pred}).to_csv(RUN_DIR / "forecasts" / f"{tag}.csv", index=False)
    np.savez_compressed(RUN_DIR / "attributions" / f"{tag}.npz", sample=S, feats=np.array(P["feats"]),
                        **{f"A::{k}": v.astype(np.float32) for k, v in A.items()},
                        **{f"ABL::{k}": (v * ysd).astype(np.float32) for k, v in abl.items()})
    mae = float(np.nanmean(np.abs(pred - P["Y"].values[I_TE])))
    log(dict(stage="main", dataset=dataset, model=kind, seed=seed, MAE=mae, epochs=epochs, seconds=round(time.time() - t0)))
    print(f"{dataset:9s} seed {seed} {kind:11s}: MAE {mae:6.0f} MW, {time.time() - t0:.0f}s")

t_all = time.time()
for dataset, flag in [("real", RUN_REAL), ("injected", RUN_INJECTED)]:
    if not flag: continue
    for seed in SEEDS:
        for kind in MODELS:
            if kind == "XGBoost" and not (HAVE_XGB and HAVE_SHAP): continue
            if is_done(stage="main", dataset=dataset, model=kind, seed=seed): continue
            run_one(dataset, kind, seed, ablation=(dataset == "real"))
print(f"main stage finished in {time.time() - t_all:.0f}s")

# %% [markdown]
# ## 7. Benchmark sensitivity (synthetic data, alternative generator settings)
#
# The Chapter 8 generator is re-run with two alternative settings (a different daily profile; twice the shock, noise and dispatch noise). The same four models with the Chapter 8 architectures are trained and explained with SHAP; magnitude correctness is scored against the known truth. If the ranking of Chapter 8 (TFT > Transformer ≈ LSTM > XGBoost) holds, it does not depend on the generator choices.

# %%
BENCH_CODE = r'''
import itertools, json, re, time, warnings
import numpy as np, pandas as pd
from scipy import stats
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.preprocessing import StandardScaler
try:
    import xgboost as xgb
except ImportError:
    xgb = None
try:
    import shap
except ImportError:
    shap = None

from pathlib import Path

OUT_DIR = Path("results/benchmark_experiments")
TZ = "Europe/Madrid"
START, END = "2015-01-01 00:00", "2018-12-31 23:00"
SPLIT = (0.64, 0.16, 0.20)
LAG_L = 36                      # strict day-ahead (fixed-lag protocol, as in Chapter 8)
SEEDS = [0, 1, 2, 3, 4]
N_EXPLAIN = 1000                # explained test windows per run (as in the report)
SEED_SAMPLE = 0

# ---- conditions: variants (Section 5.3), hidden-driver sweep, interaction variant ----
CONDITIONS = [
    dict(name="default",       change="2016-07-01", latent=1200, interaction=0.0),
    dict(name="extrapolation", change="2017-07-01", latent=1200, interaction=0.0),
    dict(name="stationary",    change=None,         latent=1200, interaction=0.0),
    dict(name="latent_0",      change="2016-07-01", latent=0,    interaction=0.0),
    dict(name="latent_600",    change="2016-07-01", latent=600,  interaction=0.0),
    dict(name="latent_2400",   change="2016-07-01", latent=2400, interaction=0.0),
    dict(name="interaction",   change="2016-07-01", latent=1200, interaction=0.5),
]

# ---- generator parameters (Section 5.2). ASSUMED = not stated in the draft ----
GEN = dict(
    base=28000, hour_amp=(2600, 900), hour_phase=(9, 3),     # phases ASSUMED
    sat=-1800, sun=-3200, holiday=-2500,
    heat_thr=16, heat_slope=250, cool_thr=22, cool_coef=150, cool_exp=1.3, cool_factor_after=1.6,
    latent_phi=0.995, ar_phi=0.95, ar_sd=500,                 # ar_sd (stationary sd, MW) ASSUMED
    noise_sd=150, operator_sd=200, gas_share=0.55, gas_latent=400, gas_noise_sd=300,   # gas_latent, gas_noise_sd ASSUMED
)
USE_EXTERNAL_GENERATOR = None   # e.g. "data/synthetic/{name}_seed{seed}" with energy_dataset.csv, weather_features.csv, truth_components.csv
WRITE_FILES = False             # write each realization in Kaggle format for the main pipeline / LEAR notebook
FILES_DIR = Path("data/synthetic")

# ---- models and hooks ----
USE_XGB = True
EXTERNAL_ATTRIBUTIONS = {}      # {"SHAP on RETAIN": "ext/{name}_seed{seed}_retain_shap.csv"}  columns: time + feature columns
EXTERNAL_FEATURE_GROUPS = {}    # regex -> group for external feature names, e.g. {r"^hour_": "hour", r"^dow_|weekend": "week", ...}
EXTERNAL_MODELS = {}            # {"RETAIN": wrapper_instance} with .fit(X, y), .predict(X), .explain(X) -> (n, F) in MW

# ---- runtime ----
FAST_MODE = False               # smoke test: 2 conditions, 2 seeds, 1 year, no XGBoost tuning
RUN_CAUSAL = True
CAUSAL_SEEDS = 3                # PCMCI+ takes minutes per run; seeds used for causal discovery
CAUSAL_HOURS = 24 * 365         # hours of the training period used for causal discovery
def spanish_holidays(years):
    try:
        import holidays as _h
        return set(_h.Spain(years=years).keys())
    except ImportError:
        fixed = [(1, 1), (1, 6), (5, 1), (8, 15), (10, 12), (11, 1), (12, 6), (12, 8), (12, 25)]
        gf = {2015: (4, 3), 2016: (3, 25), 2017: (4, 14), 2018: (3, 30), 2019: (4, 19)}
        out = {pd.Timestamp(y, m, d).date() for y in years for m, d in fixed}
        return out | {pd.Timestamp(y, *gf[y]).date() for y in years if y in gf}

def _ar(n, phi, sd, rng):
    """AR(1) with stationary standard deviation sd."""
    e = rng.normal(0, sd * np.sqrt(1 - phi ** 2), n); x = np.empty(n); x[0] = rng.normal(0, sd)
    for i in range(1, n): x[i] = phi * x[i - 1] + e[i]
    return x

CITIES = {"Madrid": 0.0, " Barcelona": 1.5, "Valencia": 2.5, "Seville": 4.0, "Bilbao": -1.5}

def generate(cond, seed, g=GEN):
    rng = np.random.default_rng(1000 * seed + 17)
    idx = pd.date_range(pd.Timestamp(START, tz=TZ).tz_convert("UTC"),
                        pd.Timestamp(END, tz=TZ).tz_convert("UTC"), freq="h")
    loc = idx.tz_convert(TZ); n = len(idx)
    h, dow, doy = loc.hour.values, loc.dayofweek.values, loc.dayofyear.values
    day = loc.tz_localize(None).normalize()
    hols = spanish_holidays(range(loc.year.min(), loc.year.max() + 1))
    is_hol = np.array([d.date() in hols for d in day])

    # weather: five cities, seasonal + daily cycle + persistent national anomaly + local noise
    anom = _ar(n, 0.995, 2.5, rng)
    temps = {c: 15 + off - 9 * np.cos(2 * np.pi * (doy - 20) / 365.25) + (5 + 0.3 * off) * np.sin(2 * np.pi * (h - 9) / 24)
                + anom + _ar(n, 0.9, 0.8, rng) for c, off in CITIES.items()}
    T = np.mean(list(temps.values()), axis=0)
    clouds = 100 / (1 + np.exp(-_ar(n, 0.98, 1.5, rng)))
    humidity = np.clip(60 + 0.3 * (clouds - 50) - 1.2 * (T - 15) + rng.normal(0, 4, n), 10, 100)
    pressure = 1015 + _ar(n, 0.99, 6, rng)
    wind_speed = np.abs(4 + _ar(n, 0.97, 2.5, rng))

    # demand components
    C_hour = g["hour_amp"][0] * np.sin(2 * np.pi * (h - g["hour_phase"][0]) / 24) + \
             g["hour_amp"][1] * np.sin(4 * np.pi * (h - g["hour_phase"][1]) / 24)
    C_week = np.where(dow == 5, g["sat"], np.where(dow == 6, g["sun"], 0)).astype(float)
    C_hol = np.where(is_hol, g["holiday"], 0).astype(float)
    k = np.ones(n)
    if cond["change"]:
        k[loc.tz_localize(None) >= pd.Timestamp(cond["change"])] = g["cool_factor_after"]
    heat = g["heat_slope"] * np.clip(g["heat_thr"] - T, 0, None)
    cool = k * g["cool_coef"] * np.clip(T - g["cool_thr"], 0, None) ** g["cool_exp"]
    C_temp = heat + cool
    C_int = cond["interaction"] * cool * np.sin(2 * np.pi * (h - 10) / 24)
    days_u, day_pos = np.unique(day.values, return_inverse=True)
    z = _ar(len(days_u), g["latent_phi"], 1.0, rng)
    z_h = np.interp(day_pos + h / 24, np.arange(len(days_u)), z)
    C_lat = cond["latent"] * z_h
    C_AR = _ar(n, g["ar_phi"], g["ar_sd"], rng)
    eps = rng.normal(0, g["noise_sd"], n)
    D = g["base"] + C_hour + C_week + C_hol + C_temp + C_int + C_lat + C_AR + eps

    # generation: consequences of demand (gas, coal) and independent sources
    season = 0.6 + 0.4 * np.sin(2 * np.pi * (doy - 80) / 365.25)
    solar = 7000 * np.clip(np.sin(np.pi * (h - 7) / 13), 0, None) * season * (1 - 0.6 * clouds / 100)
    wind = np.minimum(15000, 60 * wind_speed ** 3)
    nuclear = np.full(n, 7000.0)
    for s in rng.choice(np.arange(0, n - 24 * 30), 3, replace=False):
        nuclear[s:s + 24 * 30] -= 1000
    hydro = 3000 + 1500 * np.sin(2 * np.pi * (doy - 60) / 365.25) + rng.normal(0, 150, n)
    resid = D - solar - wind - nuclear - hydro
    gas = g["gas_share"] * resid + g["gas_latent"] * z_h * (cond["latent"] > 0) + _ar(n, 0.9, g.get("gas_noise_sd", 0), rng)
    coal = np.clip(resid - gas, 0, None)
    ar_known = np.r_[np.full(LAG_L, 0.0), C_AR[:-LAG_L]] * g["ar_phi"] ** LAG_L
    operator = D - eps - C_AR + ar_known + rng.normal(0, g["operator_sd"], n)

    energy = pd.DataFrame({
        "generation solar": solar, "generation wind onshore": wind, "generation nuclear": nuclear,
        "generation hydro water reservoir": hydro, "generation fossil gas": gas, "generation fossil hard coal": coal,
        "forecast solar day ahead": solar + rng.normal(0, 200, n), "forecast wind onshore day ahead": wind + rng.normal(0, 800, n),
        "total load forecast": operator, "total load actual": D,
        "price day ahead": 30 + 0.002 * gas + rng.normal(0, 3, n)}, index=idx)
    weather = dict(temp=T, humidity=humidity, pressure=pressure, wind_speed=wind_speed, clouds_all=clouds, cities=temps)
    truth = pd.DataFrame(dict(C_hour=C_hour, C_week=C_week, C_holiday=C_hol, C_temp=C_temp, C_interaction=C_int,
                              C_latent=C_lat, C_AR=C_AR, eps=eps, ar_known=ar_known), index=idx)
    return energy, weather, truth, hols

def write_kaggle(energy, weather, truth, folder):
    folder.mkdir(parents=True, exist_ok=True)
    t = energy.index.tz_convert(TZ).strftime("%Y-%m-%d %H:%M:%S%z")
    t = [s[:-2] + ":" + s[-2:] for s in t]
    e = energy.copy(); e.insert(0, "time", t); e["price actual"] = e["price day ahead"]
    e.to_csv(folder / "energy_dataset.csv", index=False)
    rows = [pd.DataFrame({"dt_iso": t, "city_name": c, "temp": v + 273.15, "humidity": weather["humidity"],
                          "pressure": weather["pressure"], "wind_speed": weather["wind_speed"],
                          "clouds_all": weather["clouds_all"]}) for c, v in weather["cities"].items()]
    pd.concat(rows).to_csv(folder / "weather_features.csv", index=False)
    tr = truth.copy(); tr.insert(0, "time", t); tr.to_csv(folder / "truth_components.csv", index=False)


GROUPS = {"hour": ["hour_sin", "hour_cos"], "week": ["dow_sin", "dow_cos", "weekend"], "holiday": ["holiday"],
          "temperature": ["temp", "hdd", "cdd", "temp_day_mean", "temp_day_max", "temp_day_min"]}
TRUTH_COL = {"hour": "C_hour", "week": "C_week", "holiday": "C_holiday", "temperature": "C_temp"}

def build_features(energy, weather, hols, L=LAG_L):
    loc = energy.index.tz_convert(TZ); day = loc.tz_localize(None).normalize()
    X = pd.DataFrame(index=energy.index)
    X["hour_sin"], X["hour_cos"] = np.sin(2 * np.pi * loc.hour / 24), np.cos(2 * np.pi * loc.hour / 24)
    X["dow_sin"], X["dow_cos"] = np.sin(2 * np.pi * loc.dayofweek / 7), np.cos(2 * np.pi * loc.dayofweek / 7)
    X["doy_sin"], X["doy_cos"] = np.sin(2 * np.pi * loc.dayofyear / 365.25), np.cos(2 * np.pi * loc.dayofyear / 365.25)
    X["weekend"] = (loc.dayofweek >= 5).astype(float)
    inh = lambda dd: np.array([x.date() in hols for x in dd], float)
    X["holiday"] = inh(day)
    X["pre_holiday"], X["post_holiday"] = inh(day + pd.Timedelta(days=1)), inh(day - pd.Timedelta(days=1))
    X["bridge"] = (((loc.dayofweek == 0) & (X["pre_holiday"] == 1)) | ((loc.dayofweek == 4) & (X["post_holiday"] == 1))).astype(float)
    T = pd.Series(weather["temp"], index=energy.index)
    X["temp"] = T
    for c in ("humidity", "pressure", "wind_speed", "clouds_all"):
        X[c] = weather[c]
    X["hdd"], X["cdd"] = np.clip(18 - T, 0, None), np.clip(T - 18, 0, None)
    gb = T.groupby(day.values)
    X["temp_day_mean"], X["temp_day_max"], X["temp_day_min"] = (gb.transform(f).values for f in ("mean", "max", "min"))
    y = energy["total load actual"]
    for lag in sorted({L, 48, 168} | {l for l in (24, 48, 168) if l >= L}):
        X[f"load_lag{lag}"] = y.shift(lag)
    X["load_rollmean24"] = y.rolling(24).mean().shift(L)
    X["load_rollmean168"] = y.rolling(168).mean().shift(L)
    X["load_rollstd6"] = y.rolling(6).std().shift(L)
    X["load_diff1"] = y.diff(1).shift(L); X["load_diff24"] = y.diff(24).shift(L)
    gl = max(24, L)
    for c, nm in [("generation solar", "solar"), ("generation wind onshore", "wind_gen"), ("generation nuclear", "nuclear"),
                  ("generation hydro water reservoir", "hydro"), ("generation fossil gas", "gas"), ("generation fossil hard coal", "coal")]:
        X[f"{nm}_lag{gl}"] = energy[c].shift(gl)
    ok = X.notna().all(axis=1)
    return X[ok], y[ok]

def role(f):
    if any(f in v for v in GROUPS.values()): return "direct cause"
    if f.startswith("load_"): return "demand history"
    if f.startswith(("gas_", "coal_")): return "consequence"
    return "irrelevant"

def concept(f):
    for g, v in GROUPS.items():
        if f in v: return {"week": "day of week"}.get(g, g)
    if f.startswith("load_"): return "demand history"
    return re.sub(r"_lag\d+$", "", f)

TRUE_PARENTS = {"hour", "day of week", "holiday", "temperature", "demand history"}

class RidgeSHAP:
    name = "Ridge, linear SHAP"
    def fit(self, X, y, Xv=None, yv=None):
        self.sc = StandardScaler().fit(X); self.m = RidgeCV(alphas=np.logspace(-2, 4, 25)).fit(self.sc.transform(X), y)
        self.mu = self.sc.transform(X).mean(0); return self
    def predict(self, X): return self.m.predict(self.sc.transform(X))
    def explain(self, X): return (self.sc.transform(X) - self.mu) * self.m.coef_

class XGBTreeSHAP:
    name = "XGBoost, TreeSHAP"
    def fit(self, X, y, Xv=None, yv=None):
        params = dict(n_estimators=2000, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                      min_child_weight=5, early_stopping_rounds=50, n_jobs=-1, verbosity=0)
        if FAST_MODE: params.update(n_estimators=300, early_stopping_rounds=20)
        self.m = xgb.XGBRegressor(**params).fit(X, y, eval_set=[(Xv, yv)], verbose=False)
        self.ex = shap.TreeExplainer(self.m); return self
    def predict(self, X): return self.m.predict(X)
    def explain(self, X): return np.asarray(self.ex.shap_values(X))

def controls(base_attr, rng, feats=None):
    s = base_attr.std()
    out = {"control: random": rng.normal(0, s, base_attr.shape)}
    for k in (0.5, 1.0, 2.0):
        out[f"control: noise x{k}"] = base_attr + rng.normal(0, k * s, base_attr.shape)
    perm = rng.permutation(base_attr.shape[1])
    out["control: permuted labels"] = base_attr[:, perm]
    # magnitude controls: same ranking or same pattern, wrong size
    out["control: shrunk x0.5"] = 0.5 * base_attr
    out["control: inflated x1.5"] = 1.5 * base_attr
    if feats is not None:
        cal = [k for k, f in enumerate(feats) if f in sum(GROUPS.values(), [])]
        hist = [k for k, f in enumerate(feats) if f.startswith("load_")]
        if cal and hist:
            a = base_attr.copy(); moved = 0.5 * a[:, cal]
            a[:, cal] -= moved; a[:, hist[0]] += moved.sum(1)
            out["control: half of drivers moved to history"] = a
    return out

def split_idx(n):
    a, b = int(SPLIT[0] * n), int((SPLIT[0] + SPLIT[1]) * n)
    return slice(0, a), slice(a, b), slice(b, n)
def gt_scores(A, feats, truth_s, interaction=False):
    A = pd.DataFrame(A, columns=feats); out = {}
    tr = {g: truth_s[c].values.copy() for g, c in TRUTH_COL.items()}
    conv = {"": tr}
    if interaction:
        ci = truth_s["C_interaction"].values
        conv = {"": {**tr, "temperature": tr["temperature"] + ci},
                "_split": {**tr, "temperature": tr["temperature"] + ci / 2, "hour": tr["hour"] + ci / 2}}
    for suffix, tt in conv.items():
        bs = []
        for g, cols in GROUPS.items():
            a = A[[c for c in cols if c in A]].sum(1).values; c = tt[g] - tt[g].mean()
            if np.std(c) < 1e-9 or np.std(a) < 1e-12:
                out[f"r_{g}{suffix}"], out[f"b_{g}{suffix}"] = np.nan, 0.0 if np.std(c) > 1e-9 else np.nan
            else:
                out[f"r_{g}{suffix}"] = np.corrcoef(a, c)[0, 1]; out[f"b_{g}{suffix}"] = np.cov(a, c)[0, 1] / np.var(c, ddof=1)
            bs.append(out[f"b_{g}{suffix}"])
        out[f"magnitude_correctness{suffix}"] = 1 - np.nanmean(np.abs(1 - np.array(bs, float)))
        out[f"pattern_correctness{suffix}"] = np.nanmean([out[f"r_{g}{suffix}"] for g in GROUPS])
    tot = A.abs().sum().sum()
    for rl in ("direct cause", "demand history", "consequence", "irrelevant"):
        out[f"share_{rl}"] = A[[f for f in feats if role(f) == rl]].abs().sum().sum() / tot
    imp = A.abs().mean().groupby(lambda f: concept(f)).sum().sort_values(ascending=False)
    named = set(imp.index[:5]); tp = len(named & TRUE_PARENTS)
    out["parents_precision"], out["parents_recall"] = tp / 5, tp / len(TRUE_PARENTS)
    out["parents_F1"] = 0 if tp == 0 else 2 * tp / (5 + len(TRUE_PARENTS))
    out["false_parents_consequence"] = len(named & {"gas", "coal"})
    return out

def deletion_insertion(model, Xs, y, A, mu, K=10):
    """Most-relevant-first perturbation on the model output (per-forecast rankings by |A|).
    Deletion: replace the top-k features by their training mean and record |f(x) - f(x_-k)|;
    a faithful ranking moves the prediction the most for small k.
    Insertion: start from all features at the mean, restore the top-k, record |f(x_+k) - f(x)|;
    a faithful ranking recovers the prediction fastest. Both use k = 1..K and are scaled by the
    standard deviation of the predictions, so higher = more faithful for both."""
    n, F = Xs.shape; order = np.argsort(-np.abs(A), axis=1); rows = np.arange(n)[:, None]
    f = lambda M: model.predict(pd.DataFrame(M, columns=FEATS_CUR))
    f0 = f(Xs); scale = max(np.std(f0), 1e-9); dele, ins = [], []
    for m in range(1, min(K, F) + 1):
        M = Xs.copy(); M[rows, order[:, :m]] = mu[order[:, :m]]; dele.append(np.mean(np.abs(f0 - f(M))))
        M = np.tile(mu, (n, 1)); M[rows, order[:, :m]] = Xs[rows, order[:, :m]]; ins.append(np.mean(np.abs(f(M) - f0)))
    return float(np.mean(dele) / scale), float(1 - np.mean(ins) / scale)

def regime_stability(A, err):
    hi = np.abs(err) >= np.quantile(np.abs(err), 0.9)
    v1, v0 = A[hi].var(0), A[~hi].var(0); ok = v0 > 1e-12
    return float(np.median(v1[ok] / v0[ok]))

def eval_metrics(model, Xte, yte, A, mu, ridge_top10, feats):
    out = {}
    top10 = set(np.array(feats)[np.argsort(-np.abs(A).mean(0))[:10]])
    out["metric_recall@10"] = len(top10 & ridge_top10) / 10
    out["metric_deletion"], out["metric_insertion"] = deletion_insertion(model, Xte.values, yte.values, A, mu)
    out["metric_regime_stability"] = regime_stability(A, model.predict(Xte) - yte.values)
    return out
'''

# %%
B = {}; exec(BENCH_CODE, B)
SENS = {"alt_daily_profile": dict(hour_amp=(3200, 600), hour_phase=(7, 5)),
        "high_noise": dict(ar_sd=1000, noise_sd=300, gas_noise_sd=600)}
def run_sens(cname, seed, kind):
    g = dict(B["GEN"]); g.update(SENS[cname])
    cond = dict(name=cname, change="2016-07-01", latent=1200, interaction=0.0)
    energy, weather, truth, hols = B["generate"](cond, seed, g=g)
    X, y = B["build_features"](energy, weather, hols); feats = list(X.columns)
    tr, va, te = B["split_idx"](len(X)); t0 = time.time()
    rng = np.random.default_rng(seed + 99)
    sample = np.sort(rng.choice(np.arange(te.start, te.stop), min(N_EXPLAIN, te.stop - te.start), replace=False))
    truth_s = truth.reindex(X.index[sample])
    mu, sd = X.iloc[tr].mean().values, X.iloc[tr].std().values + 1e-9
    ym, ysd = y.iloc[tr].mean(), y.iloc[tr].std()
    Xs = ((X.values - mu) / sd).astype(np.float32); ys = ((y.values - ym) / ysd).astype(np.float32)
    if kind == "XGBoost":
        m = XGBRef(seed, {}).fit(Xs[tr], ys[tr], Xs[va], ys[va])
        A = m.shap(Xs[sample]) * ysd; mae = float(np.mean(np.abs(m.predict(Xs[te]) * ysd + ym - y.values[te])))
    else:
        ARCH[kind] = dict(BENCH_ARCH[kind]); TRAIN["lr"] = 1e-3
        i_tr = np.arange(max(tr.start, WINDOW - 1), tr.stop); i_va = np.arange(va.start, va.stop); i_te = np.arange(te.start, te.stop)
        Wtr, Wva, Wte, Wex = windows(Xs, i_tr), windows(Xs, i_va), windows(Xs, i_te), windows(Xs, sample)
        bg = Wtr[np.random.default_rng(seed).choice(len(Wtr), SHAP_BACKGROUND, replace=False)]
        m = SeqModel(kind, len(feats), seed).fit(Wtr, ys[i_tr], Wva, ys[i_va])
        A = m.shap(Wex, bg, seed).sum(1) * ysd; mae = float(np.mean(np.abs(m.predict(Wte) * ysd + ym - y.values[i_te])))
    sc = B["gt_scores"](A, feats, truth_s)
    log(dict(stage="sensitivity", condition=cname, model=kind, seed=seed, MAE=mae, seconds=round(time.time() - t0),
             **{k: v for k, v in sc.items() if isinstance(v, (int, float, np.floating))}))
    print(f"sensitivity {cname:17s} seed {seed} {kind:11s}: correctness {sc['magnitude_correctness']:.2f}, MAE {mae:.0f}")

if RUN_SENSITIVITY:
    if RUN_PLAN == "smoke": B["START"] = "2017-06-01 00:00"
    for cname in SENS:
        for seed in SENS_SEEDS:
            for kind in MODELS:
                if kind == "XGBoost" and not (HAVE_XGB and HAVE_SHAP): continue
                if is_done(stage="sensitivity", condition=cname, model=kind, seed=seed): continue
                run_sens(cname, seed, kind)

# %% [markdown]
# ## 8. Analyses and results pack

# %%
PACK = OUT_DIR / f"final_results_pack_{RUN_PLAN}"; PACK.mkdir(parents=True, exist_ok=True)
MDL = [f"# Final validation results ({pd.Timestamp.now():%Y-%m-%d %H:%M}), plan '{RUN_PLAN}'", ""]
def add(df, title, fname, note=None):
    df.to_csv(PACK / fname)
    try: txt = df.to_markdown(floatfmt=".3f")
    except Exception: txt = df.to_string()
    MDL.extend([f"## {title}", f"(file: {fname})", ""] + ([note, ""] if note else []) + [txt, ""]); show(df)
LOGDF = pd.DataFrame(logged())
def load_run(dataset, kind, seed):
    p = RUN_DIR / "attributions" / f"{dataset}_{kind}_seed{seed}.npz"
    if not p.exists(): return None
    z = np.load(p, allow_pickle=True)
    return dict(sample=z["sample"], feats=list(z["feats"]),
                A={k[3:]: z[k] for k in z.files if k.startswith("A::")}, ABL={k[5:]: z[k] for k in z.files if k.startswith("ABL::")})
RUNS = {(d, k, s): r for d in DATA for k in MODELS for s in SEEDS if (r := load_run(d, k, s)) is not None}
print(len(RUNS), "runs with explanations")
SIGNED = ["XGBoost, TreeSHAP", "LSTM, SHAP", "Transformer, SHAP", "TFT, SHAP", "LSTM, IG", "Transformer, IG", "TFT, IG"]

# %%
# ---------------- A. Accuracy on real data ----------------
P = DATA["real"]; y_te = P["Y"].values[P["I_TE"]]
def metr(f):
    e = f - y_te; ok = ~np.isnan(e); return dict(MAE=np.mean(np.abs(e[ok])), RMSE=np.sqrt(np.mean(e[ok] ** 2)), MAPE=100 * np.mean(np.abs(e[ok]) / y_te[ok]))
FC = {}
for k in MODELS:
    for s in SEEDS:
        p = RUN_DIR / "forecasts" / f"real_{k}_seed{s}.csv"
        if p.exists(): FC.setdefault(k, {})[s] = pd.read_csv(p)["forecast"].values
ENS = {k: np.mean(list(v.values()), axis=0) for k, v in FC.items()}
REF = {"REE operator forecast": E["total load forecast"].values[P["I_TE"]] if "total load forecast" in E else None,
       "Naive, one week earlier": LOAD.shift(168).values[P["I_TE"]], "Naive, two days earlier": LOAD.shift(48).values[P["I_TE"]]}
REF = {k: v for k, v in REF.items() if v is not None}
rows = []
for k, v in FC.items():
    ms = pd.DataFrame([metr(f) for f in v.values()])
    rows.append(dict(model=k, seeds=len(v), MAE=ms.MAE.mean(), MAE_sd=ms.MAE.std(), RMSE=ms.RMSE.mean(), MAPE=ms.MAPE.mean(), ensemble_MAE=metr(ENS[k])["MAE"]))
for k, v in REF.items():
    m_ = metr(v); rows.append(dict(model=k, MAE=m_["MAE"], RMSE=m_["RMSE"], MAPE=m_["MAPE"]))
ACC = pd.DataFrame(rows).set_index("model")
if "REE operator forecast" in REF: ACC["ratio_to_REE"] = ACC["MAE"] / metr(REF["REE operator forecast"])["MAE"]
ACC["reduction_vs_weekly_naive_pct"] = 100 * (1 - ACC["MAE"] / metr(REF["Naive, one week earlier"])["MAE"])
add(ACC.round(2), "A1. Accuracy on the real test period (MW)", "A1_accuracy.csv")
add(pd.DataFrame(BEST).T, "A2. Selected hyperparameters (validation search)", "A2_best_params.csv")
def dm(f1, f2, h=24):
    d = np.abs(f1 - y_te) - np.abs(f2 - y_te); d = d[~np.isnan(d)]; n_ = len(d); db = d.mean()
    g = [np.sum((d[k:] - db) * (d[:n_ - k] - db)) / n_ for k in range(h)]; var = (g[0] + 2 * sum(g[1:])) / n_
    var = var if var > 0 else g[0] / n_; s = db / np.sqrt(var) * np.sqrt((n_ + 1 - 2 * h + h * (h - 1) / n_) / n_)
    return s, 2 * stats.t.sf(abs(s), n_ - 1)
ALLF = {**{f"{k} (ensemble)": v for k, v in ENS.items()}, **REF}
rows = [dict(model_1=a, model_2=b, DM_stat=(r := dm(ALLF[a], ALLF[b]))[0], p_value=r[1],
             better=(a if r[0] < 0 else b) if r[1] < 0.05 else "no significant difference") for a, b in itertools.combinations(ALLF, 2)]
add(pd.DataFrame(rows).set_index(["model_1", "model_2"]).round(4), "A3. Diebold–Mariano tests (seed ensembles)", "A3_dm_tests.csv")
te_t = IDX[P["I_TE"]].tz_convert(TZ)
err = pd.DataFrame({k: np.abs(v - y_te) for k, v in {**ENS, **REF}.items()}, index=te_t)
add(err.groupby(err.index.month).mean().round(0), "A4. MAE by month (seed ensembles)", "A4_mae_by_month.csv")
add(err.groupby(err.index.hour).mean().round(0), "A5. MAE by hour (seed ensembles)", "A5_mae_by_hour.csv")

# %%
# ---------------- B. Model-level size check (real data) ----------------
rows = []
for (d, k, s), R in RUNS.items():
    if d != "real" or not R["ABL"]: continue
    grp = np.array([group_of(f) for f in R["feats"]])
    for e, A in R["A"].items():
        if "unsigned" in e: continue
        for g, delta in R["ABL"].items():
            a = A[:, grp == g].sum(1)
            if np.std(delta) < 1e-6 or np.std(a) < 1e-9: continue
            rows.append(dict(explainer=e, seed=s, group=g, r=np.corrcoef(a, delta)[0, 1], b=np.cov(a, delta)[0, 1] / np.var(delta, ddof=1),
                             effect_sd_MW=np.std(delta)))
BB = pd.DataFrame(rows)
if len(BB):
    tab = BB.groupby(["explainer", "group"])[["r", "b", "effect_sd_MW"]].mean().round(3)
    add(tab, "B1. Model-level size check: group attribution vs the model's own effect of that group", "B1_size_check_by_group.csv",
        "b = 1: the explanation reports the full size of the model's own effect; r: pattern. Effects are interventional (group replaced by background inputs).")
    main = BB[BB.group.isin(["Demand history", "Hour of day", "Weekday", "Temperature", "Holiday and special days"])]
    summ = main.assign(size_err=lambda d_: (1 - d_.b).abs()).groupby(["explainer", "seed"]).agg(r=("r", "mean"), size_faithfulness=("size_err", lambda v: 1 - v.mean()))
    add(summ.groupby("explainer").agg(["mean", "std"]).round(3), "B2. Size faithfulness to the model (1 − mean |1 − b| over the five main groups)", "B2_size_faithfulness.csv")

# %%
# ---------------- C. Injected ground truth (real data + known effects) ----------------
BENCH_CORR = {"TFT, SHAP": 0.80, "Transformer, SHAP": 0.74, "LSTM, SHAP": 0.73, "XGBoost, TreeSHAP": 0.45}
rows = []
for (d, k, s), R in RUNS.items():
    if d != "injected": continue
    S = R["sample"]; tr = TRUTH_INJ.iloc[S]; feats = R["feats"]
    for e, A in R["A"].items():
        row = dict(explainer=e, model=k, seed=s)
        for comp, feat, truth_col in [("event", "event_flag", "C_event"), ("driver", "driver_z", "C_z")]:
            a = A[:, feats.index(feat)]; c = tr[truth_col].values - tr[truth_col].values.mean()
            row[f"r_{comp}"] = np.corrcoef(a, c)[0, 1] if np.std(a) > 0 else np.nan
            row[f"b_{comp}"] = np.cov(a, c)[0, 1] / np.var(c, ddof=1) if "unsigned" not in e else np.nan
        dz, dd = np.abs(A[:, feats.index("driver_z")]).mean(), np.abs(A[:, feats.index("decoy_d")]).mean()
        row["decoy_credit_ratio"] = dd / max(dz, 1e-9)
        row["magnitude_correctness"] = 1 - np.mean([abs(1 - row["b_event"]), abs(1 - row["b_driver"])]) if "unsigned" not in e else np.nan
        rows.append(row)
INJ = pd.DataFrame(rows)
if len(INJ):
    tabC = INJ.groupby("explainer")[["r_event", "b_event", "r_driver", "b_driver", "decoy_credit_ratio", "magnitude_correctness"]].agg(["mean", "std"]).round(3)
    add(tabC, "C1. Injected ground truth: recovery of a known event and a known nonlinear driver, and credit to a decoy",
        "C1_injected_truth.csv", "b = 1: the full injected effect is credited to its feature. decoy_credit_ratio: mean |attribution| of the decoy relative to the driver (0 is correct).")
    rk = INJ[INJ.explainer.isin(BENCH_CORR)].groupby("explainer")["magnitude_correctness"].mean()
    cmp = pd.DataFrame({"injected_real_data": rk, "benchmark_chapter8": pd.Series(BENCH_CORR)}).dropna()
    if len(cmp) >= 3:
        rho = stats.spearmanr(cmp.iloc[:, 0], cmp.iloc[:, 1]).statistic
        cmp.loc["Spearman between rankings", :] = [rho, np.nan]
    add(cmp.round(3), "C2. Does the benchmark ranking hold on real data with injected truth?", "C2_ranking_transfer.csv")
    accI = LOGDF[(LOGDF.stage == "main") & (LOGDF.dataset == "injected")].groupby("model")["MAE"].agg(["mean", "std"]).round(0)
    add(accI, "C3. Accuracy on the injected data (MW)", "C3_accuracy_injected.csv")
    if len(INJ.dropna(subset=["magnitude_correctness"])) >= 6:
        pv = INJ.pivot_table(index="seed", columns="explainer", values="magnitude_correctness")
        rows = []
        for a_, b_ in [("TFT, SHAP", "XGBoost, TreeSHAP"), ("TFT, SHAP", "LSTM, SHAP"), ("TFT, SHAP", "Transformer, SHAP"),
                       ("LSTM, SHAP", "LSTM, IG"), ("TFT, SHAP", "TFT, IG")]:
            if a_ in pv and b_ in pv:
                rows.append(dict(comparison=f"{a_} vs {b_}", seeds=len(pv), mean_difference=(pv[a_] - pv[b_]).mean(),
                                 a_better_share=((pv[a_] - pv[b_]) > 0).mean()))
        if rows: add(pd.DataFrame(rows).set_index("comparison").round(3), "C4. Paired comparison of correctness on injected data (by seed)", "C4_injected_pairs.csv")

# %%
# ---------------- D. Natural experiments on real drivers ----------------
loc = IDX.tz_convert(TZ); Yr = LOAD.values; Xr = DATA["real"]["X"]
pos_of = pd.Series(np.arange(len(IDX)), index=IDX)
def shift_rows(rows_, hours): return rows_ + hours
# D1: holiday effect from holidays on weekdays: holiday hour minus the mean of the same weekday and hour one week before and after
isspecial = (Xr[["holiday", "bridge", "pre_holiday", "post_holiday", "christmas"]].sum(axis=1).values > 0)
hol_rows = np.flatnonzero((Xr["holiday"].values == 1) & (loc.dayofweek < 5) & (Xr["christmas"].values == 0))
nat = np.full(len(IDX), np.nan)
for r_ in hol_rows:
    a_, b_ = r_ - 168, r_ + 168
    if a_ < 0 or b_ >= len(IDX) or isspecial[a_] or isspecial[b_]: continue
    nat[r_] = Yr[r_] - 0.5 * (Yr[a_] + Yr[b_])
nat_tr = np.nanmean(nat[PART == 0]); nat_te_mask = ~np.isnan(nat) & (PART == 2)
rows = []
for (d, k, s), R in RUNS.items():
    if d != "real": continue
    S = R["sample"]; m_ = nat_te_mask[S]
    if m_.sum() < 10: continue
    for e, A in R["A"].items():
        if "unsigned" in e: continue
        a = A[m_, R["feats"].index("holiday")]; c = nat[S][m_]
        rows.append(dict(explainer=e, seed=s, hours=int(m_.sum()), natural_effect_MW=c.mean(), attribution_MW=a.mean(),
                         ratio=a.mean() / c.mean(), r=np.corrcoef(a, c)[0, 1] if np.std(a) > 0 else np.nan))
NH = pd.DataFrame(rows)
if len(NH):
    add(NH.groupby("explainer")[["hours", "natural_effect_MW", "attribution_MW", "ratio", "r"]].mean().round(3),
        "D1. Holiday effect: natural estimate (holiday vs same weekday one week before and after) vs attribution to the holiday indicator, test holidays",
        "D1_holiday_natural.csv", f"Natural holiday effect over the training period: {nat_tr:.0f} MW (mean over weekday-holiday hours). ratio = 1: the attribution has the natural size.")
# D2: Sunday drop: mean demand on Sundays minus the mean on Tuesday-Thursday of the same week, same hour (no special days)
wk = pd.DataFrame({"y": Yr, "week": loc.isocalendar().week.values, "year": loc.isocalendar().year.values, "dow": loc.dayofweek,
                   "hour": loc.hour, "special": isspecial, "part": PART})
wk = wk[~wk.special]
gsun = wk[wk.dow == 6].groupby(["year", "week", "hour"]).y.mean(); gmid = wk[wk.dow.isin([1, 2, 3])].groupby(["year", "week", "hour"]).y.mean()
gap = (gsun - gmid).dropna()
test_keys = set(map(tuple, wk[wk.part == 2][["year", "week"]].drop_duplicates().values))
nat_sun = gap[[(y_, w_) in test_keys for y_, w_, h_ in gap.index]].mean()
rows = []
for (d, k, s), R in RUNS.items():
    if d != "real": continue
    S = R["sample"]; dw = loc.dayofweek.values[S]; grp = np.array([group_of(f) for f in R["feats"]])
    for e, A in R["A"].items():
        if "unsigned" in e: continue
        aw = A[:, grp == "Weekday"].sum(1)
        if (dw == 6).sum() < 10: continue
        mg = aw[dw == 6].mean() - aw[np.isin(dw, [1, 2, 3])].mean()
        rows.append(dict(explainer=e, seed=s, natural_sunday_gap_MW=nat_sun, attribution_gap_MW=mg, ratio=mg / nat_sun))
NS = pd.DataFrame(rows)
if len(NS):
    add(NS.groupby("explainer")[["natural_sunday_gap_MW", "attribution_gap_MW", "ratio"]].agg(["mean", "std"]).round(3),
        "D2. Sunday drop: natural estimate vs gap in weekday attribution (Sunday minus Tuesday–Thursday), test period", "D2_sunday_natural.csv",
        "The natural gap also contains other differences between days (e.g. temperature), so it is an approximate reference.")
# D3: temperature sensitivity (MW per degree) from a calendar-controlled regression vs slope implied by the temperature attribution
def cal_design(rows_):
    lc = loc[rows_]; D = pd.DataFrame(index=range(len(rows_)))
    for d_ in range(1, 7): D[f"dow{d_}"] = (lc.dayofweek == d_).astype(float)
    for hh in range(1, 24): D[f"h{hh}"] = (lc.hour == hh).astype(float)
    for mm in range(2, 13): D[f"m{mm}"] = (lc.month == mm).astype(float)
    for c in ["holiday", "pre_holiday", "post_holiday", "bridge", "christmas"]: D[c] = Xr[c].values[rows_]
    D["hdd"], D["cdd"] = Xr["hdd"].values[rows_], Xr["cdd"].values[rows_]; return D
trr = DATA["real"]["I_TR"]; reg = Ridge(alpha=1.0).fit(cal_design(trr), Yr[trr]); cf = pd.Series(reg.coef_, index=cal_design(trr[:3]).columns)
rows = []
for (d, k, s), R in RUNS.items():
    if d != "real": continue
    S = R["sample"]; grp = np.array([group_of(f) for f in R["feats"]]); cdd, hdd = Xr["cdd"].values[S], Xr["hdd"].values[S]
    for e, A in R["A"].items():
        if "unsigned" in e: continue
        at = A[:, grp == "Temperature"].sum(1); row = dict(explainer=e, seed=s)
        for nm, v, mask in [("cooling", cdd, cdd > 0), ("heating", hdd, hdd > 0)]:
            if mask.sum() > 30: row[f"{nm}_MW_per_degree"] = np.polyfit(v[mask], at[mask], 1)[0]
        rows.append(row)
NT = pd.DataFrame(rows)
if len(NT):
    tabT = NT.groupby("explainer").mean(numeric_only=True).drop(columns="seed").round(1)
    tabT.loc["calendar regression (training period)"] = [cf["cdd"], cf["hdd"]] if tabT.shape[1] == 2 else np.nan
    add(tabT, "D3. Temperature sensitivity: implied by the temperature attribution vs a calendar-controlled regression (MW per degree)",
        "D3_temperature_sensitivity.csv", "Heating sensitivity is reported per heating degree (demand rises as temperature falls).")

# %%
# ---------------- E. Signatures and robustness (real data) ----------------
gnames = ["Demand history"] + [g for g in GROUPS if not g.startswith("Injected")]
def gsum(A, grp): return np.stack([A[:, grp == g].sum(1) for g in gnames], 1)
rows = []
for (d, k, s), R in RUNS.items():
    if d != "real": continue
    grp = np.array([group_of(f) for f in R["feats"]])
    for e, A in R["A"].items():
        G = np.abs(gsum(A, grp)); rows.append(dict(explainer=e, seed=s, **dict(zip(gnames, G.sum(0) / G.sum()))))
SH = pd.DataFrame(rows)
if len(SH):
    add(SH.groupby("explainer")[gnames].mean().round(3), "E1. Share of absolute attribution by feature group (real data)", "E1_group_shares.csv")
    BENCH = {"XGBoost, TreeSHAP": (0.50, 0.01), "LSTM, SHAP": (0.26, 0.12), "Transformer, SHAP": (0.24, 0.13), "TFT, SHAP": (0.19, 0.11),
             "TFT, variable selection (unsigned)": (0.20, 0.05)}
    sig = pd.DataFrame({e: dict(history_real=SH[SH.explainer == e]["Demand history"].mean(), history_benchmark=BENCH[e][0],
                                gas_coal_real=SH[SH.explainer == e]["Gas and coal (consequences)"].mean(), gas_coal_benchmark=BENCH[e][1])
                        for e in BENCH if e in set(SH.explainer)}).T.round(3)
    add(sig, "E2. Signature check: shares to demand history and to gas/coal, real data vs benchmark", "E2_signatures.csv")
    fig, ax = plt.subplots(figsize=(10, 4.4))
    SH.groupby("explainer")[gnames].mean().plot(kind="barh", stacked=True, ax=ax, colormap="tab20", width=0.7)
    ax.set_xlabel("share of absolute attribution"); ax.legend(fontsize=7, bbox_to_anchor=(1.01, 1), loc="upper left")
    fig.tight_layout(); fig.savefig(PACK / "fig_E1_group_shares.png"); plt.close(fig)
rows = []
real_runs = {key: R for key, R in RUNS.items() if key[0] == "real"}
for (d, k, s), R in real_runs.items():
    grp = np.array([group_of(f) for f in R["feats"]]); A = R["A"]
    if f"{k}, IG" in A:
        r_ = pair_agreement(gsum(A[f"{k}, SHAP"], grp), gsum(A[f"{k}, IG"], grp), k=3); rows.append(dict(comparison=f"{k}: SHAP vs IG", spearman=r_[0], wtau=r_[1], J3=r_[2], S3=r_[3]))
    if "TFT, variable selection (unsigned)" in A:
        r_ = pair_agreement(A["TFT, variable selection (unsigned)"], A["TFT, SHAP"], k=5); rows.append(dict(comparison="TFT: selection weights vs SHAP (features)", spearman=r_[0], wtau=r_[1], J3=r_[2], S3=r_[3]))
for k in MODELS:
    ss = sorted(s for (d, kk, s) in real_runs if kk == k)
    e = "XGBoost, TreeSHAP" if k == "XGBoost" else f"{k}, SHAP"
    for s1, s2 in itertools.combinations(ss, 2):
        R1, R2 = real_runs[("real", k, s1)], real_runs[("real", k, s2)]; grp = np.array([group_of(f) for f in R1["feats"]])
        r_ = pair_agreement(gsum(R1["A"][e], grp), gsum(R2["A"][e], grp), k=3); rows.append(dict(comparison=f"seed ceiling: {e}", spearman=r_[0], wtau=r_[1], J3=r_[2], S3=r_[3]))
for k1, k2 in itertools.combinations(MODELS, 2):
    for s in SEEDS:
        if ("real", k1, s) in real_runs and ("real", k2, s) in real_runs:
            e1 = "XGBoost, TreeSHAP" if k1 == "XGBoost" else f"{k1}, SHAP"; e2 = "XGBoost, TreeSHAP" if k2 == "XGBoost" else f"{k2}, SHAP"
            R1, R2 = real_runs[("real", k1, s)], real_runs[("real", k2, s)]; grp = np.array([group_of(f) for f in R1["feats"]])
            r_ = pair_agreement(gsum(R1["A"][e1], grp), gsum(R2["A"][e2], grp), k=3); rows.append(dict(comparison=f"{e1} vs {e2}", spearman=r_[0], wtau=r_[1], J3=r_[2], S3=r_[3]))
if rows:
    add(pd.DataFrame(rows).groupby("comparison").mean().round(3), "E3. Agreement: SHAP vs Integrated Gradients, TFT selection weights vs SHAP, seeds and models (group level unless stated)",
        "E3_agreement.csv", "Benchmark reference for TFT selection weights vs SHAP (feature level): Spearman 0.25.")

# %%
# ---------------- F. Benchmark sensitivity ----------------
SL = LOGDF[LOGDF.stage == "sensitivity"] if "stage" in LOGDF else pd.DataFrame()
if len(SL):
    tabF = SL.groupby(["condition", "model"])[["magnitude_correctness", "pattern_correctness", "MAE"]].agg(["mean", "std"]).round(3)
    add(tabF, "F1. Benchmark sensitivity: magnitude correctness under alternative generator settings", "F1_sensitivity.csv",
        "Chapter 8 default (SHAP / TreeSHAP): TFT 0.80, Transformer 0.74, LSTM 0.72, XGBoost 0.44.")
    rk = SL.groupby(["condition", "model"])["magnitude_correctness"].mean().unstack()
    base = pd.Series({"TFT": 0.80, "Transformer": 0.74, "LSTM": 0.72, "XGBoost": 0.44})
    rows = [dict(condition=c, spearman_with_chapter8=stats.spearmanr(rk.loc[c, base.index.intersection(rk.columns)],
                                                                      base[base.index.intersection(rk.columns)]).statistic)
            for c in rk.index if rk.loc[c].notna().sum() >= 3]
    if rows: add(pd.DataFrame(rows).set_index("condition").round(3), "F2. Ranking stability: Spearman with the Chapter 8 ranking", "F2_ranking_stability.csv")

# %%
json.dump(dict(run_plan=RUN_PLAN, models=MODELS, seeds=SEEDS, best_params=BEST, train=TRAIN, window=WINDOW, n_explain=N_EXPLAIN,
               shap_nsamples=SHAP_NSAMPLES, ig_steps=IG_STEPS, ablation_k=ABLATION_K, inject=INJECT, sensitivity=SENS,
               split=dict(val_start=str(VAL_START.date()), test_start=str(TEST_START.date())), runs=len(LOGDF)),
          open(PACK / "settings.json", "w"), indent=2, default=str)
LOGDF.to_csv(PACK / "log.csv", index=False)
(PACK / "RESULTS_FINAL.md").write_text("\n".join(MDL), encoding="utf8")
zp = PACK.with_suffix(".zip")
with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
    for f in PACK.iterdir(): z.write(f, f.name)
print("Results pack:", zp)


# ======================================================================================================================
# ==== PART 3: Part 3. Intrinsic and post-hoc explanations of the same RETAIN model (real data, fixed-lag protocol)
# ======================================================================================================================

# %% [markdown]
# # Part 3. Intrinsic and post-hoc explanations of the same RETAIN model (real data, fixed-lag protocol)
#
# Trains RETAIN and the comparison models on the real Spanish data under the fixed 36-hour lag protocol, then compares, on the same 1,000 test windows: the attention score used in earlier work, the exact RETAIN contribution (Equation 1 of the paper), SHAP on the same RETAIN model, and SHAP on an LSTM.
#
# **In the paper:** Section 6.6 and the agreement figure. Part 1 reports the agreement between the exact contribution and SHAP for every seed and dataset. This part adds the steps in between: the attention score of earlier work, the same model with the exact contribution, and SHAP on a second network.
#
# | Part | Content |
# |---|---|
# | A | Leakage audit of the feature builder (`src/sequence_utils.py`) |
# | B | One protocol for every model: random search on the validation period, then all seeds, with naive forecasts, the operator forecast and Diebold-Mariano tests |
# | C | Exact RETAIN contribution, SHAP on the same model, and the step-by-step comparison of Figure 5 |
#
# **Notes.** This part imports the modules in `src/`. Part B trains six models (LSTM, a correlation-constrained LSTM, RETAIN, GRU, Transformer, XGBoost) although Part C uses only RETAIN and the LSTM. The models of one seed are trained one after another from a single random state, so removing a model would change the RETAIN weights.
#
# Results are written to `results/retain_agreement/`.
#
# **Settings.** The plan (`smoke` or `deep`), the number of seeds and the training length come from the command line (`--plan`, `--seeds`, `--max-epochs`, ...). Everything else is set in the first lines of this part. Finished runs are saved and skipped when the part is run again.

# %%
# The launcher passes the shared settings as GLOBAL_SETTINGS (plan, seeds, training length, explained hours, search size).
import json
from pathlib import Path
print(GLOBAL_SETTINGS)

# %%
import sys, json, numpy as np, pandas as pd, torch
sys.path.append("src")
from sequence_utils import (HORIZONS, load_raw, build_dataset, audit_leakage,
                            make_sequences, make_time_splits, split_mask)
from eval_protocol import align_on_time, scores, seed_summary, dm_matrix
from attribution_utils import (retain_exact_contributions, shap_gradient, collapse_time,
                               agreement_suite, decompose_disagreement, dinn_diagnostics)
from lstm_models import LSTMBaseline, RETAINLSTM, dinn_loss, dinn_feature_importance, retain_feature_importance
from unified_models import GRUPlain, TransformerPlain   # buffer-free, same architectures
from train_utils import set_seed, train_improved_lstm, predict_with_model, smooth_l1_loss_fn
from xgboost import XGBRegressor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET = "total load actual"      # switch to "price actual" for the price version
HORIZON = "da_strict"             # fair comparison vs TSO; also run "nowcast" and "h24"
SEQ_LEN = 24
SEEDS = [0, 1, 2, 3, 4]

# %% [markdown]
# ## A. Leakage audit
#
# `audit_leakage` perturbs the target at a row and checks which features change too early.
# It must return `[]` for every horizon. It is the automatic leakage audit of Appendix A of the paper.

# %%
raw = load_raw("data", TARGET)
for h, L in HORIZONS.items():
    _, feats = build_dataset("data", TARGET, h)
    leaks = audit_leakage(raw, TARGET, L, feats)
    print(f"{h:10s} min_lag={L:3d}  n_features={len(feats)}  leaky={leaks}")
    assert leaks == [], f"leakage at {h}: {leaks}"

# %% [markdown]
# ## B. One protocol for every model

# %%
def prepare(data_dir, horizon, target=TARGET):
    df, feats = build_dataset(data_dir, target, horizon)
    X_seq, y_seq, t_seq = make_sequences(df[feats], df[target], SEQ_LEN, times=df["time"])
    val_start, test_start = make_time_splits(t_seq)
    m = split_mask(t_seq, val_start, test_start)
    d = dict(df=df, feats=feats, X=X_seq, y=y_seq, t=t_seq, m=m)
    # tabular rows for XGBoost = the target row of each window -> identical timestamps
    d["X_tab"] = X_seq[:, -1, :]
    return d

D = prepare("data", HORIZON)
print({k: int(v.sum()) for k, v in D["m"].items()}, "| test starts", D["t"][D["m"]["test"]][0])

# %% [markdown]
# Hyperparameters are selected by the random search in the next cell, on the audited features of Part A.
# All neural models go through the SAME `train_improved_lstm` loop (AdamW, SmoothL1,
# plateau LR, clipping). `unified_models` gives the GRU and the Transformer the same recipe.

# %%
# ===== tune every model once, then train and evaluate all seeds =====
import sys, os, json, importlib, time
import numpy as np, pandas as pd, torch
sys.path.append("src")

import sequence_utils, eval_protocol, attribution_utils, unified_models
for _mod in (sequence_utils, eval_protocol, attribution_utils, unified_models):
    importlib.reload(_mod)

from sequence_utils import (HORIZONS, CALENDAR_COLS, load_raw, build_dataset, audit_leakage,
                            make_sequences, make_time_splits, split_mask)
from eval_protocol import align_on_time, scores, seed_summary, dm_matrix
from lstm_models import LSTMBaseline, RETAINLSTM, dinn_loss
from unified_models import GRUPlain, TransformerPlain
from train_utils import set_seed, train_improved_lstm, predict_with_model, smooth_l1_loss_fn
from xgboost import XGBRegressor
from sklearn.model_selection import ParameterSampler
from sklearn.metrics import mean_absolute_error as mae_fn

# ---------------- settings ----------------
DATA_DIR = "data"
TARGET = "total load actual"
HORIZON = "da_strict"
SEQ_LEN = 24
_smoke = GLOBAL_SETTINGS["plan"] == "smoke"
SEEDS = [0] if _smoke else list(range(GLOBAL_SETTINGS["seeds"]))
N_TRIALS = 1 if _smoke else GLOBAL_SETTINGS["tune_trials"]          # candidates per model (the whole grid is used if it is smaller)
TUNE_EPOCHS, TUNE_PATIENCE = (2, 1) if _smoke else (GLOBAL_SETTINGS["tune_epochs"], 6)
FINAL_EPOCHS, FINAL_PATIENCE = (2, 1) if _smoke else (GLOBAL_SETTINGS["max_epochs"], GLOBAL_SETTINGS["patience"])
DINN_LAMBDA = 0.3
RUN_NOWCAST_CHECK = not _smoke
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("results/retain_agreement", exist_ok=True)
os.makedirs("results/retain_agreement/saved_models", exist_ok=True)
print("device:", device)

# ---------------- 0. checks ----------------
assert "doy_sin" in CALENDAR_COLS, "src/sequence_utils.py is still the old version: replace it and restart the kernel"
_raw = load_raw(DATA_DIR, TARGET)
for _h, _L in HORIZONS.items():
    _, _feats = build_dataset(DATA_DIR, TARGET, _h)
    _leaks = audit_leakage(_raw, TARGET, _L, _feats)
    print(f"leakage audit {_h:10s} n_features={len(_feats):3d} leaky={_leaks}")
    assert not _leaks, f"leakage at {_h}: {_leaks}"
del _raw

# ---------------- helpers ----------------
def prepare(data_dir, horizon, target=TARGET):
    df, feats = build_dataset(data_dir, target, horizon)
    X_seq, y_seq, t_seq = make_sequences(df[feats], df[target], SEQ_LEN, times=df["time"])
    val_start, test_start = make_time_splits(t_seq)
    d = dict(df=df, feats=feats, X=X_seq, y=y_seq, t=t_seq, m=split_mask(t_seq, val_start, test_start))
    d["X_tab"] = X_seq[:, -1, :]
    return d

def norm(mdl, X):
    return (X - mdl._feature_mean) / mdl._feature_std

def dinn_loss_fn(pred, y, xb):
    return dinn_loss(pred, y, xb[:, -1, :], lam=DINN_LAMBDA)

CLASSES = {"LSTM": (LSTMBaseline, "smoothl1"), "DINN-LSTM": (LSTMBaseline, "dinn"),
           "RETAIN": (RETAINLSTM, "smoothl1"), "GRU": (GRUPlain, "smoothl1"),
           "Transformer": (TransformerPlain, "smoothl1")}
ARCH = {"LSTM":        {"hidden_size": [64, 128], "num_layers": [1, 2], "dropout": [0.1, 0.3]},
        "DINN-LSTM":   {"hidden_size": [64, 128], "num_layers": [1, 2], "dropout": [0.1, 0.3]},
        "RETAIN":      {"hidden_size": [64, 128], "dropout": [0.2, 0.3]},
        "GRU":         {"hidden_size": [64, 128], "num_layers": [1, 2], "dropout": [0.1, 0.3]},
        "Transformer": {"d_model": [32, 64], "nhead": [4], "num_layers": [1, 2], "dropout": [0.1, 0.2]}}
TRAIN = {"lr": [3e-4, 1e-3, 3e-3], "weight_decay": [1e-6, 1e-4], "batch_size": [128, 256]}
OLD_BEST = {
    "LSTM":   ({"hidden_size": 64, "num_layers": 1, "dropout": 0.3}, {"lr": 1e-3, "weight_decay": 1e-6, "batch_size": 128}),
    "RETAIN": ({"hidden_size": 128, "dropout": 0.3}, {"lr": 3e-4, "weight_decay": 1e-4, "batch_size": 256}),
}
XGB_SPACE = {"max_depth": [4, 6, 8], "learning_rate": [0.03, 0.05], "min_child_weight": [1, 5],
             "subsample": [0.8], "colsample_bytree": [0.6, 0.8]}

def tune(D):
    m, X, y = D["m"], D["X"], D["y"]
    models, train_by_model, log = {}, {}, []
    for name, (cls, loss) in CLASSES.items():
        lf = dinn_loss_fn if loss == "dinn" else smooth_l1_loss_fn
        cands = ([OLD_BEST[name]] if name in OLD_BEST else []) + [
            ({k: p[k] for k in ARCH[name]}, {k: p[k] for k in TRAIN})
            for p in ParameterSampler({**ARCH[name], **TRAIN}, N_TRIALS, random_state=0)]
        results = []
        # one line per finished trial: an interrupted search resumes here. Each trial starts from set_seed(0),
        # so a trial gives the same result whether or not the trials before it were run in the same run.
        _trial_file = f"results/retain_agreement/tuning_trials_{HORIZON}_{name}.jsonl"
        _done = {}
        if os.path.exists(_trial_file):
            for _l in open(_trial_file):
                if _l.strip(): _r = json.loads(_l); _done[_r["trial"]] = _r["val_MAE"]
            print(f"  {name}: {len(_done)} finished trials loaded from {_trial_file}")
        for i, (kw, tp) in enumerate(cands):
            if i in _done:
                val = float(_done[i]); results.append((val, kw, tp))
                log.append({"model": name, "trial": i, "val_MAE": val, **kw, **tp}); continue
            set_seed(0)
            t0 = time.time()
            mdl, *_ = train_improved_lstm(cls, kw, lf, X[m["train"]], y[m["train"]], X[m["val"]], y[m["val"]],
                                          tp, device, epochs=TUNE_EPOCHS, patience=TUNE_PATIENCE,
                                          lr_patience=3, verbose_every=1000)
            val = mae_fn(y[m["val"]], predict_with_model(mdl, X[m["val"]], device))
            if not np.isfinite(val):
                val = np.inf
            results.append((val, kw, tp))
            log.append({"model": name, "trial": i, "val_MAE": val, **kw, **tp})
            with open(_trial_file, "a") as _fh: _fh.write(json.dumps({"trial": i, "val_MAE": float(val)}, default=float) + "\n")
            print(f"  {name:12s} trial {i + 1}/{len(cands)}  val MAE {val:8.1f}  ({time.time() - t0:.0f}s)")
            del mdl
            if device.type == "cuda":
                torch.cuda.empty_cache()
        val, kw, tp = min(results, key=lambda r: r[0])
        if not np.isfinite(val):
            raise RuntimeError(f"every {name} candidate diverged; check its loss before continuing")
        models[name], train_by_model[name] = (cls, kw, loss), tp
        print(f"BEST {name:12s} val MAE {val:8.1f} | {kw} | {tp}")

    xgb_res = []
    for p in ParameterSampler(XGB_SPACE, 8, random_state=0):
        xg = XGBRegressor(n_estimators=2000, early_stopping_rounds=50, n_jobs=-1, random_state=0, **p)
        xg.fit(D["X_tab"][m["train"]], y[m["train"]], eval_set=[(D["X_tab"][m["val"]], y[m["val"]])], verbose=False)
        val = mae_fn(y[m["val"]], xg.predict(D["X_tab"][m["val"]]))
        xgb_res.append((val, p))
        log.append({"model": "XGBoost", "val_MAE": val, **p})
    val, best_xgb = min(xgb_res, key=lambda r: r[0])
    print(f"BEST {'XGBoost':12s} val MAE {val:8.1f} | {best_xgb}")
    return models, train_by_model, dict(n_estimators=2000, **best_xgb), pd.DataFrame(log)

def run_seed(D, seed, epochs=FINAL_EPOCHS, patience=FINAL_PATIENCE):
    set_seed(seed)
    m, X, y = D["m"], D["X"], D["y"]
    preds, fitted = {}, {}
    for name, (cls, kw, loss) in MODELS.items():
        lf = dinn_loss_fn if loss == "dinn" else smooth_l1_loss_fn
        mdl, *_ = train_improved_lstm(cls, kw, lf, X[m["train"]], y[m["train"]], X[m["val"]], y[m["val"]],
                                      TRAIN_BY_MODEL[name], device, epochs=epochs, patience=patience,
                                      lr_patience=5, verbose_every=1000)
        preds[name], fitted[name] = predict_with_model(mdl, X[m["test"]], device), mdl
    xgb = XGBRegressor(random_state=seed, early_stopping_rounds=50, n_jobs=-1, **XGB_PARAMS)
    xgb.fit(D["X_tab"][m["train"]], y[m["train"]], eval_set=[(D["X_tab"][m["val"]], y[m["val"]])], verbose=False)
    preds["XGBoost"], fitted["XGBoost"] = xgb.predict(D["X_tab"][m["test"]]), xgb
    return preds, fitted

def with_baselines(preds, D):
    yt = D["df"].set_index("time")[TARGET]
    t_test = pd.to_datetime(D["t"][D["m"]["test"]], utc=True)
    out = dict(preds)
    out["naive t-168"] = yt.shift(168, freq="h").reindex(t_test).to_numpy()
    out["naive t-48"] = yt.shift(48, freq="h").reindex(t_test).to_numpy()
    return out

def evaluate(D, seeds):
    y_series = D["df"].set_index("time")[TARGET]
    tso = D["df"].set_index("time")["total load forecast"] if TARGET == "total load actual" else None
    t_test = D["t"][D["m"]["test"]]
    tables, fitted_all, aligned_all = [], {}, {}
    for s in seeds:
        t0 = time.time()
        preds, fitted = run_seed(D, s)
        preds = with_baselines(preds, D)
        al = align_on_time(preds, y_series, {k: t_test for k in preds}, tso=tso)
        tables.append(scores(al))
        fitted_all[s], aligned_all[s] = fitted, al
        print(f"seed {s} done ({time.time() - t0:.0f}s) | MAE:",
              {k: round(v, 1) for k, v in scores(al)["MAE"].items()})
    return seed_summary(tables), fitted_all, aligned_all

# ---------------- 1. tune ----------------
D = prepare(DATA_DIR, HORIZON)
m = D["m"]
print({k: int(v.sum()) for k, v in m.items()}, "| n_features:", len(D["feats"]),
      "| test starts:", D["t"][m["test"]][0])

MODELS, TRAIN_BY_MODEL, XGB_PARAMS, tune_log = tune(D)
tune_log.to_csv(f"results/retain_agreement/tuning_log_{HORIZON}.csv", index=False)
with open(f"results/retain_agreement/tuned_params_{HORIZON}.json", "w") as f:
    json.dump({"models": {k: {"arch": v[1], "train": TRAIN_BY_MODEL[k], "loss": v[2]} for k, v in MODELS.items()},
               "xgboost": XGB_PARAMS, "horizon": HORIZON, "n_features": len(D["feats"]),
               "features": D["feats"]}, f, indent=2)

# ---------------- 2. rerun Part B ----------------
summary, fitted_by_seed, aligned_by_seed = evaluate(D, SEEDS)
summary.to_csv(f"results/retain_agreement/partB_summary_{HORIZON}.csv")
display(summary.round(1))

# ---------------- 3. significance + error structure ----------------
stat, pval = dm_matrix(aligned_by_seed[SEEDS[0]], h=24)
pval.to_csv(f"results/retain_agreement/dm_pvalues_{HORIZON}.csv")
display(pval.round(4))

best_model = summary.drop(index=[i for i in ["TSO", "naive t-168", "naive t-48"] if i in summary.index])["MAE"].idxmin()
al0 = aligned_by_seed[SEEDS[0]]
err = (al0[best_model] - al0["y"]).abs()
by_month = err.groupby(al0.index.month).mean().round(0).rename(f"{best_model} MAE by month")
by_hour = err.groupby(al0.index.hour).mean().round(0).rename(f"{best_model} MAE by hour (UTC)")
by_month.to_csv(f"results/retain_agreement/error_by_month_{HORIZON}.csv")
by_hour.to_csv(f"results/retain_agreement/error_by_hour_{HORIZON}.csv")
display(by_month.to_frame().T)
display(by_hour.to_frame().T)

# ---------------- 4. nowcast sanity check ----------------
if RUN_NOWCAST_CHECK:
    D_now = prepare(DATA_DIR, "nowcast")
    preds_now, _ = run_seed(D_now, SEEDS[0])
    preds_now = with_baselines(preds_now, D_now)
    al_now = align_on_time(preds_now, D_now["df"].set_index("time")[TARGET],
                           {k: D_now["t"][D_now["m"]["test"]] for k in preds_now},
                           tso=D_now["df"].set_index("time")["total load forecast"] if TARGET == "total load actual" else None)
    now_table = scores(al_now).sort_values("MAE")
    now_table.to_csv("results/retain_agreement/nowcast_check.csv")
    print("NOWCAST check (1 seed). Expect roughly your old ~350-470 MW if the pipeline is sound:")
    display(now_table.round(1))
    del D_now

# ---------------- 5. save seed-0 models ----------------
for name, mdl in fitted_by_seed[SEEDS[0]].items():
    if name == "XGBoost":
        mdl.save_model(f"results/retain_agreement/saved_models/xgboost_{HORIZON}_seed0.json")
    else:
        torch.save({"state_dict": mdl.state_dict(), "arch": MODELS[name][1], "train": TRAIN_BY_MODEL[name],
                    "feature_mean": mdl._feature_mean, "feature_std": mdl._feature_std,
                    "target_mean": mdl._target_mean, "target_std": mdl._target_std,
                    "features": D["feats"], "horizon": HORIZON, "seq_len": SEQ_LEN},
                   f"results/retain_agreement/saved_models/{name}_{HORIZON}_seed0.pt")

print("\nDone. Tables in results/retain_agreement/, models in results/retain_agreement/saved_models/.")

# %%
# Is the best model significantly better than the TSO / each other? (h=24: daily error correlation)
stat, pval = dm_matrix(aligned_by_seed[SEEDS[0]], h=24)
pval.round(4)

# %% [markdown]
# Also report, next to this table: horizon (`da_strict` = issued ~12:00 D-1), that weather
# is used as a perfect forecast, the exact test period, and n_seeds. If no model beats the
# TSO at `da_strict`, that is fine and publishable: the paper is about explanation quality,
# and "our explained models sit X% above the operator's benchmark" is an honest framing.

# %% [markdown]
# ## C. Intrinsic vs post-hoc on the SAME model
#
# Earlier work compared the RETAIN attention score with SHAP on a different model. The table below separates
# (1)→(2) the model confound, (2)→(3) the attention score, and (3) the real
# intrinsic-vs-post-hoc gap. Lead with jaccard@5 / wtau, not raw Spearman.

# %%
fit0 = fitted_by_seed[SEEDS[0]]
retain, lstm = fit0["RETAIN"], fit0["LSTM"]
m = D["m"]
norm = lambda mdl, X: (X - mdl._feature_mean) / mdl._feature_std     # same scaling the models trained on
X_tr_R, X_te_R = norm(retain, D["X"][m["train"]]), norm(retain, D["X"][m["test"]])
X_tr_L, X_te_L = norm(lstm,   D["X"][m["train"]]), norm(lstm,   D["X"][m["test"]])

N_EXPLAIN = 100 if GLOBAL_SETTINGS["plan"] == "smoke" else GLOBAL_SETTINGS["n_explain"]
idx = np.random.default_rng(0).choice(len(X_te_R), N_EXPLAIN, replace=False)

ex = retain_exact_contributions(retain, X_te_R[idx], device)
print("completeness error (should be ~1e-6):", ex["max_abs_completeness_error"])
retain_exact = collapse_time(ex["contrib"], "signed")
retain_proxy = retain_feature_importance(retain, X_te_R[idx], device=device)   # v1 method

shap_retain = collapse_time(shap_gradient(retain, X_tr_R, X_te_R[idx], device), "signed")
shap_lstm   = collapse_time(shap_gradient(lstm,   X_tr_L, X_te_L[idx], device), "signed")

decomp = decompose_disagreement(retain_exact, retain_proxy, shap_retain, shap_lstm, k=5)
decomp.round(3)


# ======================================================================================================================
# ==== PART 4: Part 4. Renewable sources under a hidden demand driver
# ======================================================================================================================

# %% [markdown]
# # Part 4. Renewable sources under a hidden demand driver
#
# **In the paper:** Section 6.4, the renewable-sources figure. Results are written to `results/renewable_sources/`.
#
# Same models, settings and explainers as the benchmark in the paper:
# * **XGBoost + TreeSHAP** (tabular features at the target hour)
# * **LSTM, Transformer, TFT + SHAP** (expected gradients, `shap.GradientExplainer`, 300 background windows,
#   200 samples, 24-hour input window; attribution summed over the window)
# * **TFT variable-selection weights** (its built-in importance, unsigned) as an extra line in Experiment A
#
# **Synthetic system (one per hidden intensity and seed).**
# * Demand `D = 28000 + hour + weekly + holiday + temperature + hidden H + AR(0.95) + noise`.
#   `H` is a slow daily AR(0.995) process whose standard deviation is the *hidden-factor intensity* (0 to 3600 MW).
#   No feature measures `H`.
# * Supply split into sources with known equations:
#   * **Solar**: sun position x season x (1 - 0.7 clouds). Never depends on demand.
#   * **Wind (curtailed)**: available = min(15000, 40 ws^3); delivered = min(available, max(R - 9000, 0)),
#     `R = D - solar - nuclear - biomass`. At low demand wind is curtailed, so its output follows demand (and `H`).
#   * **Hydro (reservoir)**: `0.30 (R - wind) + seasonal inflow + noise`.
#   * **Pumped storage**: `0.25 (net load - its 24 h mean)`, clipped to +/- 3000 MW.
#   * **Biomass** about 600 MW flat. **Nuclear** 7000 MW with random unit outages. Gas and coal cover the rest (not features).
#
# **Experiment A (day-ahead demand).** Features: calendar, weather, demand lags 36/48/168 h and every source lagged 36 h.
# Truth: the sources have **zero** effect on demand. Y axis: share of |attribution| given to each source.
#
# **Experiment B (hydro dispatch).** Target: hydro at hour t. Features: calendar, weather, same-hour wind, solar,
# nuclear, biomass. Demand is not given. Truth: each of wind, solar, nuclear moves hydro by exactly **-0.30 MW per MW**.
# Y axis: correctness slope `b` (attribution regressed on the true contribution, 1 = correct, < 0 = sign flipped).
#
# Split: train 2015-2016, validation 2017 (early stopping), test 2018. Resumable: finished runs are read from `runs.jsonl`.
#
# **Settings.** The plan (`smoke` or `deep`), the number of seeds and the training length come from the command line (`--plan`, `--seeds`, `--max-epochs`, ...). Everything else is set in the first lines of this part. Finished runs are saved and skipped when the part is run again.

# %%
# The launcher passes the shared settings as GLOBAL_SETTINGS (plan, seeds, training length, explained hours, search size).
import json
from pathlib import Path
print(GLOBAL_SETTINGS)

# %%
# ================= SETTINGS =================
RUN_PLAN = "smoke" if GLOBAL_SETTINGS["plan"] == "smoke" else "deep"
PLANS = {"smoke": ([0, 2400], [0]),
         "core":  ([0, 600, 1200, 2400, 3600], [0, 1, 2]),
         "all":   ([0, 600, 1200, 2400, 3600], [0, 1, 2, 3, 4]),
         "deep":  ([0, 600, 1200, 2400, 3600], list(range(GLOBAL_SETTINGS["seeds"])))}
MODELS_TO_RUN = ["XGBoost", "LSTM", "Transformer", "TFT"]
EXPERIMENTS = ["A", "B"]
WINDOW = 24
# Same configurations as the benchmark runs of the paper
ARCH = dict(LSTM=dict(hidden=128, layers=2, dropout=0.1),
            RETAIN=dict(hidden=128, dropout=0.2),
            Transformer=dict(d_model=64, heads=4, layers=2, dropout=0.1),
            TFT=dict(d_model=32, heads=4, dropout=0.1))
TRAIN = dict(lr=1e-3, weight_decay=1e-4, batch=256, max_epochs=GLOBAL_SETTINGS["max_epochs"], patience=GLOBAL_SETTINGS["patience"], clip=1.0)
SHAP_BACKGROUND, SHAP_NSAMPLES = 300, 200
N_EXPLAIN = GLOBAL_SETTINGS["n_explain"]              # test hours explained per run (same hours for every model)
OUT = "results/renewable_sources" + ("_smoke" if RUN_PLAN == "smoke" else "")
if RUN_PLAN == "smoke":
    TRAIN.update(max_epochs=3, patience=2); N_EXPLAIN, SHAP_NSAMPLES = 150, 50

# %%
import os, json, time, warnings
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
warnings.filterwarnings("ignore")
import xgboost as xgb, shap
HAVE_SHAP = True
os.makedirs(OUT, exist_ok=True)
LOG = os.path.join(OUT, "runs.jsonl")
HIDDEN_LEVELS, SEEDS = PLANS[RUN_PLAN]

# %% [markdown]
# ## 1. Data generator

# %%
def ar(n, phi, sd, rng):
    e = rng.normal(0, sd * np.sqrt(1 - phi ** 2), n); x = np.zeros(n); x[0] = rng.normal(0, sd)
    for i in range(1, n): x[i] = phi * x[i - 1] + e[i]
    return x

def generate(hidden_sd, seed):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2015-01-01", "2018-12-31 23:00", freq="h"); n = len(idx)
    hour, dow, doy = idx.hour.values, idx.dayofweek.values, idx.dayofyear.values
    day = (idx.normalize() - idx[0].normalize()).days.values
    nd = day.max() + 1
    season = np.sin(2 * np.pi * (doy - 80) / 365.25)            # +1 mid summer, -1 mid winter
    hol_days = rng.choice(nd, 12 * 4, replace=False)
    holiday = np.isin(day, hol_days).astype(float)
    # weather
    temp = 16 + 9 * season + 4 * np.sin(2 * np.pi * (hour - 9) / 24) + np.repeat(ar(nd, 0.8, 3, rng), 24)[:n] + ar(n, 0.9, 0.8, rng)
    clouds = np.clip(0.4 + np.repeat(ar(nd, 0.6, 0.25, rng), 24)[:n] + ar(n, 0.95, 0.08, rng), 0, 1)
    ws = np.clip(7 + ar(n, 0.97, 3, rng) - 1.0 * season, 0, 25)
    humidity = np.clip(65 - 15 * season + ar(n, 0.95, 8, rng), 20, 100)
    # demand components (as in the paper benchmark, with the gas share removed)
    c_hour = 3500 * np.sin(2 * np.pi * (hour - 6) / 24) - 1200 * np.cos(4 * np.pi * (hour - 6) / 24)
    c_week = np.where(dow == 5, -1800, np.where(dow == 6, -3200, 0.0))
    c_hol = -2500 * holiday
    c_temp = 250 * np.clip(16 - temp, 0, None) + 150 * np.clip(temp - 22, 0, None) ** 1.3
    H = np.repeat(ar(nd, 0.995, 1.0, rng), 24)[:n]
    H = (H - H.mean()) / (H.std() + 1e-9) * hidden_sd if hidden_sd > 0 else np.zeros(n)
    c_ar = ar(n, 0.95, 300, rng)
    D = 28000 + c_hour + c_week + c_hol + c_temp + H + c_ar + rng.normal(0, 150, n)
    # supply
    elev = np.clip(np.sin(2 * np.pi * (hour - 6) / 24), 0, None) * (0.75 + 0.25 * season)
    solar = 6000 * elev * (1 - 0.7 * clouds) + rng.normal(0, 50, n) * (elev > 0)
    nuclear = 7000 - 1000 * (np.repeat(rng.random(nd) < 0.0, 24)[:n])
    out = np.zeros(nd); i = 0
    while i < nd:                                              # unit outages lasting 2-5 weeks
        if rng.random() < 0.01:
            L = rng.integers(14, 36); out[i:i + L] += rng.integers(1, 3); i += L
        else: i += 1
    nuclear = 7000 - 1000 * np.repeat(out, 24)[:n]
    biomass = 600 + rng.normal(0, 30, n)
    R = D - solar - nuclear - biomass
    wind_avail = np.minimum(15000, 40 * ws ** 3)
    wind = np.clip(np.minimum(wind_avail, np.clip(R - 9000, 0, None)) + rng.normal(0, 100, n), 0, None)
    netload = R - wind
    hydro_noise = ar(n, 0.98, 400, rng)
    hydro = np.clip(0.30 * netload + 1500 * np.sin(2 * np.pi * (doy - 20) / 365.25) + hydro_noise, 0, 12000)
    roll = pd.Series(netload).rolling(24, min_periods=1).mean().values
    pumped = np.clip(0.25 * (netload - roll), -3000, 3000) + rng.normal(0, 80, n)
    df = pd.DataFrame(dict(demand=D, solar=solar, wind=wind, hydro=hydro, pumped=pumped, biomass=biomass,
                           nuclear=nuclear, temp=temp, clouds=clouds, wind_speed=ws, humidity=humidity,
                           hour=hour, dow=dow, month=idx.month.values, holiday=holiday,
                           c_hour=c_hour, c_week=c_week, c_hol=c_hol, c_temp=c_temp, H=H,
                           hydro_noise=hydro_noise), index=idx)
    return df

SOURCES = ["hydro", "pumped", "wind", "solar", "biomass", "nuclear"]
LABEL = {"hydro": "Hydro (reservoir)", "pumped": "Pumped storage", "wind": "Wind (curtailed)",
         "solar": "Solar", "biomass": "Biomass", "nuclear": "Nuclear"}

# %% [markdown]
# ## 2. Features for the two experiments

# %%
def calendar(df):
    X = pd.DataFrame(index=df.index)
    X["hour_sin"], X["hour_cos"] = np.sin(2 * np.pi * df.hour / 24), np.cos(2 * np.pi * df.hour / 24)
    for h in range(24): X[f"h{h}"] = (df.hour == h).astype(float)
    X["sat"], X["sun"] = (df.dow == 5).astype(float), (df.dow == 6).astype(float)
    X["holiday"] = df.holiday
    X["month_sin"], X["month_cos"] = np.sin(2 * np.pi * df.month / 12), np.cos(2 * np.pi * df.month / 12)
    return X

def weather(df):
    X = pd.DataFrame(index=df.index)
    X["temp"] = df.temp; X["hdd"] = np.clip(16 - df.temp, 0, None); X["cdd"] = np.clip(df.temp - 22, 0, None)
    X["cdd13"] = X["cdd"] ** 1.3
    X["clouds"], X["wind_speed"], X["humidity"] = df.clouds, df.wind_speed, df.humidity
    return X

def features_A(df):
    """Day-ahead demand forecast. Lags >= 36 h (issued 12:00 the day before)."""
    X = pd.concat([calendar(df), weather(df)], axis=1)
    for L in (36, 48, 168): X[f"load_lag{L}"] = df.demand.shift(L)
    X["load_roll24_lag36"] = df.demand.shift(36).rolling(24).mean()
    for s in SOURCES: X[f"{s}_lag36"] = df[s].shift(36)
    groups = {c: ("calendar" if c in calendar(df.iloc[:1]).columns else
                  "weather" if c in weather(df.iloc[:1]).columns else
                  "demand_history" if c.startswith("load") else c.replace("_lag36", "")) for c in X.columns}
    return X, df.demand, groups

def features_B(df):
    """Hydro dispatch from same-hour forecasts of the other sources. Demand is not observed."""
    X = pd.concat([calendar(df), weather(df)], axis=1)
    X["doy_sin"] = np.sin(2 * np.pi * (df.index.dayofyear - 20) / 365.25)
    X["doy_cos"] = np.cos(2 * np.pi * (df.index.dayofyear - 20) / 365.25)
    for s in ["wind", "solar", "nuclear", "biomass"]: X[s] = df[s]
    groups = {c: (c if c in ["wind", "solar", "nuclear", "biomass"] else
                  "weather" if c in weather(df.iloc[:1]).columns else "calendar") for c in X.columns}
    return X, df.hydro, groups

# %% [markdown]
# ## 3. XGBoost + TreeSHAP and the sequence models
# XGBoost uses the paper's settings (2000 trees max, depth 6, learning rate 0.05, early stopping on 2017).
# The neural networks are copied unchanged from the benchmark notebook.

# %%
class XGBTreeSHAP:
    def fit(self, X, y, Xv, yv, seed):
        self.m = xgb.XGBRegressor(n_estimators=2000, max_depth=6, learning_rate=0.05, subsample=0.8,
                                  colsample_bytree=0.8, min_child_weight=5, early_stopping_rounds=50,
                                  n_jobs=-1, verbosity=0, random_state=seed).fit(X, y, eval_set=[(Xv, yv)], verbose=False)
        self.ex = shap.TreeExplainer(self.m); return self
    def predict(self, X): return self.m.predict(X)
    def explain(self, X): return np.asarray(self.ex.shap_values(X))

# %%
import torch, torch.nn as nn
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", DEVICE, "| torch", torch.__version__, "| shap", shap.__version__ if HAVE_SHAP else "missing")
assert HAVE_SHAP, "pip install shap"

class LSTMNet(nn.Module):
    def __init__(self, F, hidden, layers, dropout):
        super().__init__()
        self.rnn = nn.LSTM(F, hidden, layers, batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(self, x):
        h, _ = self.rnn(x); return self.head(h[:, -1]).squeeze(-1)

class RETAINNet(nn.Module):
    def __init__(self, F, hidden, dropout):
        super().__init__()
        self.emb = nn.Linear(F, hidden)
        self.gru_a = nn.GRU(hidden, hidden, batch_first=True); self.gru_b = nn.GRU(hidden, hidden, batch_first=True)
        self.att_a = nn.Linear(hidden, 1); self.att_b = nn.Linear(hidden, hidden)
        self.drop = nn.Dropout(dropout); self.out = nn.Linear(hidden, 1)
    def attention(self, x):
        v = self.emb(x)
        rv = torch.flip(self.drop(v), [1])                       # reverse time
        g, _ = self.gru_a(rv); h, _ = self.gru_b(rv)
        g, h = torch.flip(g, [1]), torch.flip(h, [1])
        alpha = torch.softmax(self.att_a(g).squeeze(-1), dim=1)   # (n, T)
        beta = torch.tanh(self.att_b(h))                         # (n, T, H)
        return v, alpha, beta
    def forward(self, x):
        v, a, b = self.attention(x)
        return self.out((a.unsqueeze(-1) * b * v).sum(1)).squeeze(-1)
    def contributions(self, x):
        """Exact contribution (n, T, F) and the feature-independent bias (n,)."""
        v, a, b = self.attention(x)
        w, W = self.out.weight.squeeze(0), self.emb.weight          # (H,), (H, F)
        coef = torch.einsum("nt,nth,h,hf->ntf", a, b, w, W)
        bias = ((a.unsqueeze(-1) * b * self.emb.bias).sum(1) * w).sum(-1) + self.out.bias
        return x * coef, bias

class PosEnc(nn.Module):
    def __init__(self, d, T=512):
        super().__init__()
        pe = torch.zeros(T, d); pos = torch.arange(T).unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2) * (-np.log(10000.0) / d))
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer("pe", pe)
    def forward(self, x): return x + self.pe[: x.shape[1]]

class EncLayer(nn.Module):
    def __init__(self, d, heads, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.ReLU(), nn.Linear(4 * d, d))
        self.n1, self.n2, self.drop = nn.LayerNorm(d), nn.LayerNorm(d), nn.Dropout(dropout)
        self.last_w = None
    def forward(self, x, keep=False):
        a, w = self.attn(x, x, x, need_weights=keep, average_attn_weights=True)
        if keep: self.last_w = w                                  # (n, T, T)
        x = self.n1(x + self.drop(a)); return self.n2(x + self.drop(self.ff(x)))

class TransformerNet(nn.Module):
    def __init__(self, F, d_model, heads, layers, dropout):
        super().__init__()
        self.inp, self.pos = nn.Linear(F, d_model), PosEnc(d_model)
        self.layers = nn.ModuleList([EncLayer(d_model, heads, dropout) for _ in range(layers)])
        self.head = nn.Linear(d_model, 1)
    def forward(self, x, keep=False):
        z = self.pos(self.inp(x))
        for L in self.layers: z = L(z, keep)
        return self.head(z[:, -1]).squeeze(-1)
    def attention_score(self, x):
        """Section 4.2.2: attention of the last position on each step x |input|, summed over time (unsigned)."""
        self.forward(x, keep=True)
        w = self.layers[-1].last_w[:, -1, :]                      # (n, T)
        return (w.unsqueeze(-1) * x.abs()).sum(1)

class GLU(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__(); self.l = nn.Linear(d_in, 2 * d_out)
    def forward(self, x):
        a, b = self.l(x).chunk(2, dim=-1); return a * torch.sigmoid(b)

class GRN(nn.Module):
    """Gated residual network of the TFT."""
    def __init__(self, d_in, d_hid, d_out, dropout):
        super().__init__()
        self.skip = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()
        self.l1, self.l2 = nn.Linear(d_in, d_hid), nn.Linear(d_hid, d_hid)
        self.drop, self.glu, self.norm = nn.Dropout(dropout), GLU(d_hid, d_out), nn.LayerNorm(d_out)
    def forward(self, a):
        h = self.drop(self.l2(nn.functional.elu(self.l1(a))))
        return self.norm(self.skip(a) + self.glu(h))

class InterpretableMHA(nn.Module):
    """TFT interpretable multi-head attention: per-head queries/keys, shared values, heads averaged."""
    def __init__(self, d, heads, dropout):
        super().__init__()
        self.dk = d // heads
        self.q = nn.ModuleList([nn.Linear(d, self.dk) for _ in range(heads)])
        self.k = nn.ModuleList([nn.Linear(d, self.dk) for _ in range(heads)])
        self.v, self.o, self.drop = nn.Linear(d, self.dk), nn.Linear(self.dk, d), nn.Dropout(dropout)
    def forward(self, q, k):
        v, outs, atts = self.v(k), [], []
        for qi, ki in zip(self.q, self.k):
            a = torch.softmax(qi(q) @ ki(k).transpose(1, 2) / self.dk ** 0.5, dim=-1)
            atts.append(a); outs.append(self.drop(a) @ v)
        return self.o(torch.stack(outs).mean(0)), torch.stack(atts).mean(0)

class TFTNet(nn.Module):
    """Temporal Fusion Transformer (Lim et al., 2021), single-horizon version for one target hour:
    variable selection network -> LSTM encoder with gated skip -> enrichment GRN ->
    interpretable attention from the last step -> gated position-wise GRN -> output.
    Simplifications: no static covariates; one GRN shared by all variables (the original uses one per variable)."""
    def __init__(self, F, d_model, heads, dropout):
        super().__init__()
        d = d_model
        self.emb_w, self.emb_b = nn.Parameter(torch.randn(F, d) * 0.1), nn.Parameter(torch.zeros(F, d))
        self.vsn, self.var_grn = GRN(F * d, d, F, dropout), GRN(d, d, d, dropout)
        self.lstm = nn.LSTM(d, d, batch_first=True)
        self.gate1, self.norm1 = GLU(d, d), nn.LayerNorm(d)
        self.enrich = GRN(d, d, d, dropout)
        self.attn = InterpretableMHA(d, heads, dropout)
        self.gate2, self.norm2 = GLU(d, d), nn.LayerNorm(d)
        self.ff, self.gate3, self.norm3 = GRN(d, d, d, dropout), GLU(d, d), nn.LayerNorm(d)
        self.out = nn.Linear(d, 1)
        self.last_w = None
    def select(self, x):
        e = x.unsqueeze(-1) * self.emb_w + self.emb_b                         # (n, T, F, d)
        n, T, F_, d = e.shape
        w = torch.softmax(self.vsn(e.reshape(n, T, F_ * d)), dim=-1)           # (n, T, F) selection weights
        return (w.unsqueeze(-1) * self.var_grn(e)).sum(2), w
    def forward(self, x, keep=False):
        z, w = self.select(x)
        h, _ = self.lstm(z)
        h = self.enrich(self.norm1(z + self.gate1(h)))
        last = h[:, -1:]
        a, _ = self.attn(last, h)
        y = self.norm2(last + self.gate2(a))
        y = self.norm3(y + self.gate3(self.ff(y)))
        if keep: self.last_w = w
        return self.out(y[:, 0]).squeeze(-1)
    def selection_weights(self, x):
        """TFT's built-in importance: variable-selection weights averaged over the window (unsigned, sum to 1)."""
        self.forward(x, keep=True); return self.last_w.mean(1)

def build_net(kind, F):
    a = ARCH[kind]
    if kind == "LSTM": return LSTMNet(F, a["hidden"], a["layers"], a["dropout"])
    if kind == "RETAIN": return RETAINNet(F, a["hidden"], a["dropout"])
    if kind == "TFT": return TFTNet(F, a["d_model"], a["heads"], a["dropout"])
    return TransformerNet(F, a["d_model"], a["heads"], a["layers"], a["dropout"])

class Out2D(nn.Module):
    def __init__(self, net): super().__init__(); self.net = net
    def forward(self, x): return self.net(x).unsqueeze(-1)

class SeqModel:
    """Train / predict / explain one network on standardized windows (n, T, F)."""
    def __init__(self, kind, F, seed):
        torch.manual_seed(seed); np.random.seed(seed)
        self.kind, self.net = kind, build_net(kind, F).to(DEVICE)
    def _t(self, a): return torch.as_tensor(a, dtype=torch.float32, device=DEVICE)
    def fit(self, Xtr, ytr, Xva, yva):
        opt = torch.optim.AdamW(self.net.parameters(), lr=TRAIN["lr"], weight_decay=TRAIN["weight_decay"])
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)
        loss_fn = nn.HuberLoss(); best, best_state, bad = np.inf, None, 0
        Xtr_t, ytr_t = self._t(Xtr), self._t(ytr); n = len(Xtr)
        self.epochs = 0
        for ep in range(TRAIN["max_epochs"]):
            self.net.train(); perm = torch.randperm(n, device=DEVICE)
            for i in range(0, n, TRAIN["batch"]):
                idx = perm[i:i + TRAIN["batch"]]
                opt.zero_grad(); loss = loss_fn(self.net(Xtr_t[idx]), ytr_t[idx]); loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), TRAIN["clip"]); opt.step()
            val = float(np.mean(np.abs(self.predict(Xva) - yva)))
            sched.step(val); self.epochs = ep + 1
            if val < best - 1e-5:
                best, bad = val, 0; best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
            else:
                bad += 1
                if bad >= TRAIN["patience"]: break
        self.net.load_state_dict(best_state); self.net.eval(); return self
    @torch.no_grad()
    def predict(self, Xw, bs=2048):
        self.net.eval()
        return np.concatenate([self.net(self._t(Xw[i:i + bs])).cpu().numpy() for i in range(0, len(Xw), bs)])
    def shap(self, Xw, background, seed=0):
        self.net.eval()
        wrapped = Out2D(self.net).eval()                          # shap expects (n, 1) outputs
        with torch.backends.cudnn.flags(enabled=False):          # RNN backward in eval mode on GPU
            ex = shap.GradientExplainer(wrapped, self._t(background))
            sv = ex.shap_values(self._t(Xw), nsamples=SHAP_NSAMPLES, rseed=seed)
        sv = np.asarray(sv[0] if isinstance(sv, list) else sv)
        return sv.reshape(len(Xw), Xw.shape[1], Xw.shape[2])     # (n, T, F)
    @torch.no_grad()
    def intrinsic(self, Xw):
        self.net.eval(); x = self._t(Xw)
        if self.kind == "RETAIN":
            c, bias = self.net.contributions(x)
            err = (c.sum((1, 2)) + bias - self.net(x)).abs().max().item()
            assert err < 1e-3, f"RETAIN identity violated ({err:.2e})"
            return c.cpu().numpy(), err
        if self.kind == "Transformer":
            return self.net.attention_score(x).cpu().numpy(), None
        if self.kind == "TFT":
            return self.net.selection_weights(x).cpu().numpy(), None
        return None, None

# %%
def windows(Xs, idx, T=WINDOW):
    return np.stack([Xs[i - T + 1:i + 1] for i in idx]).astype(np.float32)

# %% [markdown]
# ## 4. Scoring against the ground truth
# * **share::g**: sum of |attribution| to group g over the explained hours, divided by the total over all features.
# * **b_s** (Experiment B, signed explainers only): slope of the attribution to source s regressed on its true
#   contribution -0.30 (x_s - training mean). 1 = correct, 0 = no effect found, negative = sign flipped.
# * **implied_effect_s** = -0.30 x b_s, the MW of hydro per MW of source s that the explanation implies.

# %%
def score(A, cols, groups, Xe_raw, exp, means, signed=True):
    G = sorted(set(groups.values())); r = {}
    tot = np.abs(A).sum()
    gA = {g: A[:, [i for i, c in enumerate(cols) if groups[c] == g]].sum(1) for g in G}
    for g in G: r[f"share::{g}"] = float(np.abs(gA[g]).sum() / tot)
    if exp == "B" and signed:
        for s in ["wind", "solar", "nuclear"]:
            c = -0.30 * (Xe_raw[:, cols.index(s)] - means[s])
            r[f"b_{s}"] = float(np.polyfit(c, gA[s], 1)[0]) if c.std() > 1e-6 else np.nan
            r[f"implied_effect_{s}"] = -0.30 * r[f"b_{s}"]
    return r

def done_keys():
    if not os.path.exists(LOG): return set()
    return {(d["exp"], d["hidden"], d["seed"], d["model"]) for d in map(json.loads, open(LOG))}

# %% [markdown]
# ## 5. Run the sweep (resumable)

# %%
done, t_all, n_new = done_keys(), time.time(), 0
print(f"plan '{RUN_PLAN}': {len(HIDDEN_LEVELS)} intensities x {len(SEEDS)} seeds x {len(EXPERIMENTS)} experiments "
      f"x {len(MODELS_TO_RUN)} models; {len(done)} model runs already done")
for hidden in HIDDEN_LEVELS:
    for seed in SEEDS:
        todo = [(e, k) for e in EXPERIMENTS for k in MODELS_TO_RUN if (e, hidden, seed, k) not in done]
        if not todo: continue
        df = generate(hidden, seed)
        for exp in EXPERIMENTS:
            kinds = [k for e, k in todo if e == exp]
            if not kinds: continue
            X, y, groups = (features_A if exp == "A" else features_B)(df)
            ok = X.notna().all(1); X, y = X[ok], y[ok]; cols, F = list(X.columns), X.shape[1]
            yr = X.index.year
            tr, va, te = np.where(yr <= 2016)[0], np.where(yr == 2017)[0], np.where(yr == 2018)[0]
            tr = tr[tr >= WINDOW - 1]
            sample = np.sort(np.random.default_rng(seed + 99).choice(te, min(N_EXPLAIN, len(te)), replace=False))
            Xv, yv = X.values, y.values
            means = dict(zip(cols, Xv[tr].mean(0)))
            mu, sd = Xv[tr].mean(0), Xv[tr].std(0) + 1e-9
            ym, ysd = yv[tr].mean(), yv[tr].std()
            Xs, ys = ((Xv - mu) / sd).astype(np.float32), ((yv - ym) / ysd).astype(np.float32)
            extra = {}
            if exp == "B": extra["corr_wind_H"] = float(np.corrcoef(df.wind, df.H)[0, 1]) if hidden else 0.0
            neural = [k for k in kinds if k != "XGBoost"]
            if neural:
                Wtr, Wva, Wte, Wex = windows(Xs, tr), windows(Xs, va), windows(Xs, te), windows(Xs, sample)
                bg = Wtr[np.random.default_rng(seed).choice(len(Wtr), SHAP_BACKGROUND, replace=False)]
            for kind in kinds:
                t0 = time.time(); expl = {}
                if kind == "XGBoost":
                    m = XGBTreeSHAP().fit(Xv[tr], yv[tr], Xv[va], yv[va], seed)
                    mae = float(np.mean(np.abs(m.predict(Xv[te]) - yv[te])))
                    expl["XGBoost, TreeSHAP"] = (m.explain(Xv[sample]), True)
                else:
                    m = SeqModel(kind, F, seed).fit(Wtr, ys[tr], Wva, ys[va])
                    mae = float(np.mean(np.abs(m.predict(Wte) * ysd + ym - yv[te])))
                    expl[f"{kind}, SHAP"] = (m.shap(Wex, bg, seed).sum(1) * ysd, True)
                    if kind == "TFT":
                        expl["TFT, variable selection (unsigned)"] = (m.intrinsic(Wex)[0], False)
                for ename, (A, signed) in expl.items():
                    rec = dict(exp=exp, hidden=hidden, seed=seed, model=kind, explainer=ename, mae=mae,
                               fit_seconds=round(time.time() - t0), **extra,
                               **score(A, cols, groups, Xv[sample], exp, means, signed))
                    open(LOG, "a").write(json.dumps(rec) + "\n")
                n_new += 1
                print(f"{exp} hidden={hidden:5d} seed={seed} {kind:11s} MAE={mae:7.0f} MW  {time.time()-t0:5.0f}s")
if n_new:
    per = (time.time() - t_all) / n_new
    full = 5 * 5 * len(EXPERIMENTS) * len(MODELS_TO_RUN)
    print(f"\n{n_new} model runs, ~{per:.0f}s each. Estimate: core ~{per*full*3/5/3600:.1f} h, all ~{per*full/3600:.1f} h"
          + ("  (smoke uses short training: real runs take several times longer)" if RUN_PLAN == "smoke" else ""))
res = pd.DataFrame([json.loads(l) for l in open(LOG)])
res.to_csv(os.path.join(OUT, "all_runs.csv"), index=False); print(len(res), "explainer rows")

# %% [markdown]
# ## 6. Figures

# %%
COL = {"XGBoost, TreeSHAP": "#DD8452", "LSTM, SHAP": "#4C72B0", "Transformer, SHAP": "#8172B3",
       "TFT, SHAP": "#55A868", "TFT, variable selection (unsigned)": "#55A868"}
STYLE = {"TFT, variable selection (unsigned)": dict(ls=":", marker="s")}
ORDER = [e for e in COL if e in set(res.explainer)]

def band(ax, d, key, name, scale=1.0):
    if key not in d or d[key].isna().all(): return
    g = d.groupby("hidden")[key]; m, s = g.mean() * scale, g.std().fillna(0) * scale
    st = dict(ls="-", marker="o"); st.update(STYLE.get(name, {}))
    ax.plot(m.index, m.values, color=COL[name], label=name, lw=2, **st)
    ax.fill_between(m.index, (m - s).values, (m + s).values, color=COL[name], alpha=0.12)

A_, B_ = res[res.exp == "A"], res[res.exp == "B"]

# Figure R1: Experiment A, credit wrongly given to each source
fig, axs = plt.subplots(2, 3, figsize=(14, 7.5), sharex=True)
for ax, s in zip(axs.flat, SOURCES):
    for name in ORDER: band(ax, A_[A_.explainer == name], f"share::{s}", name, 100)
    ax.axhline(0, ls="--", c="k", lw=1, label="Ground truth (0 %)"); ax.set_title(LABEL[s]); ax.grid(alpha=0.3)
for ax in axs[1]: ax.set_xlabel("Hidden-factor intensity (std of H, MW)")
for ax in axs[:, 0]: ax.set_ylabel("Share of |attribution| (%)")
axs[0, 0].legend(fontsize=7.5)
fig.suptitle("Experiment A. Demand forecast: credit wrongly given to lagged generation, by source\n"
             "(true effect of every source on demand = 0)", fontsize=12)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_R1_renewable_shares.png"), dpi=160)

# Figure R2: Experiment B, effect of one source on another (signed explainers)
SIGNED = [e for e in ORDER if "unsigned" not in e]
fig, axs = plt.subplots(1, 4, figsize=(17, 4.4))
for ax, s in zip(axs[:3], ["wind", "solar", "nuclear"]):
    for name in SIGNED: band(ax, B_[B_.explainer == name], f"b_{s}", name)
    ax.axhline(1, ls="--", c="k", lw=1, label="Correct (b = 1)"); ax.axhline(0, c="grey", lw=0.8)
    ax.set_title(f"Effect of {LABEL[s].split(' (')[0].lower()} on hydro"); ax.grid(alpha=0.3)
    ax.set_xlabel("Hidden-factor intensity (MW)")
axs[0].set_ylabel("Correctness slope b\n(1 = true -0.30 MW/MW recovered)"); axs[0].legend(fontsize=7.5)
for name in SIGNED: band(axs[3], B_[B_.explainer == name], "implied_effect_wind", name)
axs[3].axhline(-0.30, ls="--", c="k", lw=1); axs[3].axhline(0, c="grey", lw=0.8)
axs[3].set_title("Implied wind to hydro effect (truth -0.30)"); axs[3].set_xlabel("Hidden-factor intensity (MW)")
axs[3].set_ylabel("MW hydro per MW wind"); axs[3].grid(alpha=0.3)
fig.suptitle("Experiment B. Hydro dispatch: how a hidden demand factor distorts the effect of one source on another", fontsize=12)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_R2_source_on_source.png"), dpi=160)

# Figure R3: overview
fig, axs = plt.subplots(1, 4, figsize=(18, 4.2))
A2 = A_.copy(); A2["share::all_sources"] = A2[[f"share::{s}" for s in SOURCES]].sum(1)
for name in ORDER:
    d = A2[A2.explainer == name]
    band(axs[0], d, "share::all_sources", name, 100); band(axs[1], d, "share::demand_history", name, 100)
    if "unsigned" not in name:
        band(axs[2], d, "mae", name); band(axs[3], B_[B_.explainer == name], "mae", name)
for ax, t, yl in zip(axs, ["A: all generation lags (truth 0 %)", "A: demand history", "A: test MAE (demand)", "B: test MAE (hydro)"],
                     ["%", "%", "MW", "MW"]):
    ax.set_title(t); ax.set_xlabel("Hidden-factor intensity (MW)"); ax.set_ylabel(yl); ax.grid(alpha=0.3)
axs[0].axhline(0, ls="--", c="k", lw=1); axs[0].legend(fontsize=7.5)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_R3_overview.png"), dpi=160)
plt.show()

# %% [markdown]
# ## 7. Tables and results pack

# %%
keysA = [f"share::{s}" for s in SOURCES] + ["share::demand_history", "mae"]
tA = A_.groupby(["explainer", "hidden"])[keysA].mean()
tA[[k for k in keysA if k.startswith("share")]] *= 100
tA = tA.round(2); tA.to_csv(os.path.join(OUT, "tableA_shares_by_intensity.csv"))
keysB = [k for k in ["b_wind", "b_solar", "b_nuclear", "implied_effect_wind", "implied_effect_solar", "mae", "corr_wind_H"] if k in B_]
tB = B_[B_.explainer.isin(SIGNED)].groupby(["explainer", "hidden"])[keysB].mean().round(3)
tB.to_csv(os.path.join(OUT, "tableB_effects_by_intensity.csv"))
print(tA.to_string()); print(); print(tB.to_string())
import shutil; shutil.make_archive(OUT + "_pack", "zip", OUT); print("saved", OUT + "_pack.zip")


# ======================================================================================================================
# ==== PART 5: Part 5. Replication on a second country: Portugal
# ======================================================================================================================

# %% [markdown]
# # Part 5. Replication on a second country: Portugal
#
# **In the paper:** Section 6.7. Results are written to `results/portugal_replication/`. The model settings in `BEST` and the Spanish values in the comparison table (section 9 of this part) are typed in by hand: update them from the tables of Part 1 once the deep run has finished.
#
# This notebook repeats the paper's validation on **Portuguese** electricity demand with the method that was most correct in Spain, **TFT + SHAP**, and keeps **XGBoost + TreeSHAP** as the reference (the least correct in Spain). Nothing is tuned or redesigned. The pipeline, features, scoring, explainers and model settings are those of the Spanish study. Only the data, the holidays, the time zone and the calibration of the synthetic generator change.
#
# | Step | What is done | Spanish counterpart |
# |---|---|---|
# | 1 | Download Portuguese load, TSO day-ahead forecast and generation (ENTSO-E, or OPSD without a key) and weather for five cities (Open-Meteo) for 2015 to 2018. Save them in the same format as the Spanish Kaggle files | Kaggle Jhana dataset |
# | 2 | Same operational protocol (forecast issued at 12:00 on D-1), same features and the same leakage audit | Section 3 |
# | 3 | **Synthetic benchmark calibrated to Portugal**: same generator, with its demand profile, weekend and holiday drops, temperature response and climate fitted to the Portuguese training data, and every other size scaled by the ratio of mean demand. Default version plus the hidden-driver sweep, five seeds | Section 5, Table 5, Figure 3 |
# | 4 | **Real data**: accuracy against the TSO forecast and naive forecasts, size check, natural experiments (Sunday drop, holidays, temperature), signatures | Section 6.5, Table 7 |
# | 5 | **Injected truth** in the real Portuguese data: the same event, nonlinear driver and decoy, scaled to Portuguese demand | Table 6 |
# | 6 | **Spain vs Portugal**: every headline number side by side, with automatic checks of each finding and a written conclusion | |
#
# **Settings reused from Spain without tuning.** Real data: TFT model dimension 32, 4 heads, dropout 0.2, learning rate 0.003. XGBoost depth 8, learning rate 0.03, minimum child weight 1, column sample 0.8. Benchmark: the architectures of the Spanish benchmark runs. SHAP: expected gradients with 300 background windows and 200 samples, 1,000 explained test hours per run.
#
# **Optional.** Put an ENTSO-E API key in `ENTSOE_API_KEY` to use the same source and columns as the Spanish data (needs `entsoe-py`). Without it, Open Power System Data is used.
#
# **If a download fails** (firewall or proxy), download the files in a browser and put them in `data/portugal/raw/`: the OPSD file as `opsd_time_series_60min_singleindex.csv` (from data.open-power-system-data.org, time series, version 2020-10-06, 60-minute single index), and for each city the Open-Meteo response as `openmeteo_<City>.json` (the URLs are in section 1). The notebook then uses them without downloading.
#
# **Settings.** The plan (`smoke` or `deep`), the number of seeds and the training length come from the command line (`--plan`, `--seeds`, `--max-epochs`, ...). Everything else is set in the first lines of this part. Finished runs are saved and skipped when the part is run again.

# %%
# The launcher passes the shared settings as GLOBAL_SETTINGS (plan, seeds, training length, explained hours, search size).
import json
from pathlib import Path
print(GLOBAL_SETTINGS)

# %%
# ============================== CONFIGURATION ==============================
from pathlib import Path
COUNTRY, COUNTRY_NAME = "PT", "Portugal"
TZ = "Europe/Lisbon"
START_REAL, END_REAL = "2015-01-01", "2018-12-31"     # same years as the Spanish study
DATA_DIR = Path("data/portugal")
ENERGY_CSV, WEATHER_CSV = DATA_DIR / "energy_dataset.csv", DATA_DIR / "weather_features.csv"
ENTSOE_API_KEY = ""                    # optional: same source and columns as the Spanish data. Empty: OPSD is used
OUT_DIR = Path("results/portugal_replication")
ISSUE_HOUR, LAST_KNOWN_HOUR = 12, 11
SPLIT = (0.64, 0.16, 0.20)

RUN_PLAN = "smoke" if GLOBAL_SETTINGS["plan"] == "smoke" else "full"
MODELS = ["TFT", "XGBoost"]            # TFT + SHAP under test, XGBoost + TreeSHAP as reference. Add "LSTM", "Transformer" for the full set
SEEDS = list(range(GLOBAL_SETTINGS["seeds"]))
WINDOW = 24
N_EXPLAIN = GLOBAL_SETTINGS["n_explain"]             # random test hours explained per run (plus holiday or event hours)
SHAP_BACKGROUND, SHAP_NSAMPLES = 300, 200
RUN_IG = False                         # Integrated Gradients as a second explainer (not needed for the replication)
IG_STEPS = 32
ABLATION_K = 16                        # background draws for the model-level size check
TRAIN = dict(lr=1e-3, weight_decay=1e-4, batch=256, max_epochs=GLOBAL_SETTINGS["max_epochs"], patience=GLOBAL_SETTINGS["patience"], clip=1.0)

# No tuning: the configurations selected on the Spanish validation period (paper, Table 2)
BEST = {"TFT": dict(d_model=32, heads=4, dropout=0.2, lr=0.003),
        "XGBoost": dict(max_depth=8, lr=0.03, min_child_weight=1, colsample=0.8),
        "LSTM": dict(hidden=256, layers=1, dropout=0.2, lr=0.003),
        "Transformer": dict(d_model=128, heads=4, layers=3, dropout=0.2, lr=0.001)}
# Architectures of the Spanish benchmark runs (synthetic data)
ARCH = dict(LSTM=dict(hidden=128, layers=2, dropout=0.1), Transformer=dict(d_model=64, heads=4, layers=2, dropout=0.1),
            TFT=dict(d_model=32, heads=4, dropout=0.1), RETAIN=dict(hidden=128, dropout=0.2))
BENCH_ARCH = {k: dict(v) for k, v in ARCH.items()}

# Synthetic benchmark: hidden-driver size in Spanish MW (scaled to Portugal by the ratio of mean demand)
SYN_CONDITIONS = {"default": 1200, "latent_0": 0, "latent_600": 600, "latent_2400": 2400}
# Injected truth in Spanish MW (scaled to Portugal in section 3)
INJECT = dict(seed=2026, n_event_days=60, event_mw=-1500, event_hours=(8, 21),
              driver_mw=900, driver_temp_interaction=0.4, decoy_corr=0.8)
RUN_SYNTHETIC, RUN_REAL, RUN_INJECTED = True, True, True

if RUN_PLAN == "smoke":
    SEEDS, N_EXPLAIN, SHAP_NSAMPLES, IG_STEPS, ABLATION_K = [0], 80, 30, 8, 4
    SYN_CONDITIONS = {"default": 1200, "latent_0": 0}
    TRAIN.update(max_epochs=2, patience=1)
RUN_DIR = OUT_DIR / RUN_PLAN

# %%
import itertools, json, time, warnings, zipfile, copy, urllib.request
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import Ridge
warnings.filterwarnings("ignore")
for d in ("attributions", "forecasts"): (RUN_DIR / d).mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 200, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
try:
    show = display
except NameError:
    show = print
HAVE_XGB = HAVE_SHAP = False
try:
    import xgboost as xgb; HAVE_XGB = True
except ImportError: pass
try:
    import shap; HAVE_SHAP = True
except ImportError: pass
print(f"xgboost {HAVE_XGB} | shap {HAVE_SHAP}")

LOG = RUN_DIR / "log.jsonl"            # one JSON line per finished step (robust to changing columns)
def logged():
    if not LOG.exists(): return []
    return [json.loads(l) for l in LOG.read_text().splitlines() if l.strip()]
def log(row):
    with open(LOG, "a") as f: f.write(json.dumps(row, default=float) + "\n")
def is_done(**key):
    return any(all(r.get(k) == v for k, v in key.items()) for r in logged())

# %% [markdown]
# ## 1. Data: download once and save in the format of the Spanish files

# %%
# Download once and save in the format of the Spanish Kaggle files (energy_dataset.csv, weather_features.csv),
# so that the Spanish loading and feature code below runs unchanged.
import urllib.request
OPSD_URL = "https://data.open-power-system-data.org/time_series/2020-10-06/time_series_60min_singleindex.csv"
METEO_URL = ("https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={s}&end_date={e}"
             "&hourly=temperature_2m,relative_humidity_2m,pressure_msl,wind_speed_10m,cloud_cover"
             "&wind_speed_unit=ms&timezone=GMT")
CITY_COORDS = {"Lisbon": (38.72, -9.14), "Porto": (41.15, -8.61), "Braga": (41.55, -8.42),
               "Coimbra": (40.21, -8.43), "Faro": (37.02, -7.93)}
RAW = DATA_DIR / "raw"; RAW.mkdir(parents=True, exist_ok=True)
T0, T1 = pd.Timestamp(START_REAL, tz="UTC") - pd.Timedelta(days=1), pd.Timestamp(END_REAL, tz="UTC") + pd.Timedelta(days=2)

def _download(url, dest):
    if not dest.exists():
        print("downloading", url[:90], "...")
        tmp = dest.with_suffix(".part"); urllib.request.urlretrieve(url, tmp); tmp.rename(dest)
    return dest

def energy_from_entsoe(key):
    from entsoe import EntsoePandasClient
    cl = EntsoePandasClient(api_key=key); out = pd.DataFrame()
    lf = cl.query_load_and_forecast(COUNTRY, start=T0, end=T1)
    out["total load actual"], out["total load forecast"] = lf["Actual Load"], lf["Forecasted Load"]
    g = cl.query_generation(COUNTRY, start=T0, end=T1)
    if isinstance(g.columns, pd.MultiIndex):
        g = g.xs("Actual Aggregated", axis=1, level=1) if "Actual Aggregated" in g.columns.get_level_values(1) else g.droplevel(1, axis=1)
    for src, col in [("Fossil Gas", "generation fossil gas"), ("Fossil Hard Coal", "generation fossil hard coal"),
                     ("Hydro Water Reservoir", "generation hydro water reservoir"), ("Nuclear", "generation nuclear"),
                     ("Solar", "generation solar"), ("Wind Onshore", "generation wind onshore")]:
        if src in g: out[col] = g[src]
    ws = cl.query_wind_and_solar_forecast(COUNTRY, start=T0, end=T1)
    for src, col in [("Solar", "forecast solar day ahead"), ("Wind Onshore", "forecast wind onshore day ahead")]:
        if src in ws: out[col] = ws[src]
    out.index = pd.to_datetime(out.index, utc=True)
    return out.resample("h").mean()

def energy_from_opsd():
    p = _download(OPSD_URL, RAW / "opsd_time_series_60min_singleindex.csv")
    cols = set(pd.read_csv(p, nrows=0).columns)
    pick = lambda *names: next((c for c in names if c in cols), None)
    m = {"total load actual": pick(f"{COUNTRY}_load_actual_entsoe_transparency"),
         "total load forecast": pick(f"{COUNTRY}_load_forecast_entsoe_transparency"),
         "generation solar": pick(f"{COUNTRY}_solar_generation_actual"),
         "generation wind onshore": pick(f"{COUNTRY}_wind_onshore_generation_actual", f"{COUNTRY}_wind_generation_actual")}
    m = {k: v for k, v in m.items() if v}
    assert "total load actual" in m, f"no {COUNTRY} load column in the OPSD file"
    df = pd.read_csv(p, usecols=["utc_timestamp"] + list(m.values()))
    df.index = pd.to_datetime(df.pop("utc_timestamp"), utc=True)
    return df.rename(columns={v: k for k, v in m.items()})

def weather_from_openmeteo():
    rows = []
    for city, (lat, lon) in CITY_COORDS.items():
        p = _download(METEO_URL.format(lat=lat, lon=lon, s=T0.date(), e=T1.date()), RAW / f"openmeteo_{city}.json")
        hh = pd.DataFrame(json.loads(p.read_text())["hourly"])
        hh["time"] = pd.to_datetime(hh["time"], utc=True)
        rows.append(pd.DataFrame({"time": hh["time"], "city_name": city, "temp": hh["temperature_2m"] + 273.15,
                                  "humidity": hh["relative_humidity_2m"], "pressure": hh["pressure_msl"],
                                  "wind_speed": hh["wind_speed_10m"], "clouds_all": hh["cloud_cover"]}))
    return pd.concat(rows)

def iso(t): return t.strftime("%Y-%m-%d %H:%M:%S+00:00")
if not ENERGY_CSV.exists():
    E_raw = energy_from_entsoe(ENTSOE_API_KEY) if ENTSOE_API_KEY else energy_from_opsd()
    E_raw = E_raw.loc[pd.Timestamp(START_REAL, tz=TZ).tz_convert("UTC"):pd.Timestamp(END_REAL + " 23:00", tz=TZ).tz_convert("UTC")]
    E_raw.insert(0, "time", iso(E_raw.index)); E_raw.to_csv(ENERGY_CSV, index=False)
    (DATA_DIR / "SOURCE.txt").write_text("ENTSO-E Transparency (entsoe-py)" if ENTSOE_API_KEY else f"OPSD {OPSD_URL}")
if not WEATHER_CSV.exists():
    W_raw = weather_from_openmeteo(); W_raw.insert(0, "dt_iso", iso(pd.DatetimeIndex(W_raw.pop("time"))))
    W_raw.to_csv(WEATHER_CSV, index=False)
print("energy source:", (DATA_DIR / "SOURCE.txt").read_text() if (DATA_DIR / "SOURCE.txt").exists() else "existing file")
print("energy columns:", list(pd.read_csv(ENERGY_CSV, nrows=0).columns))

# %% [markdown]
# ## 2. Data and features (operational protocol), unchanged from the Spanish study

# %%
def load_energy(path):
    df = pd.read_csv(path); df["time"] = pd.to_datetime(df["time"], utc=True)
    return df.drop_duplicates("time").set_index("time").sort_index()

def load_weather(path):
    w = pd.read_csv(path); w["city_name"] = w["city_name"].astype(str).str.strip()
    w["time"] = pd.to_datetime(w["dt_iso"], utc=True); w = w.drop_duplicates(["time", "city_name"])
    w["temp"] = w["temp"] - 273.15
    if "pressure" in w: w.loc[(w["pressure"] < 900) | (w["pressure"] > 1100), "pressure"] = np.nan      # known outliers in this dataset
    if "wind_speed" in w: w.loc[w["wind_speed"] > 50, "wind_speed"] = np.nan
    cols = [c for c in ["temp", "humidity", "pressure", "wind_speed", "clouds_all"] if c in w]
    return w.groupby("time")[cols].mean().sort_index()

def country_holidays(years):
    """National holidays of Portugal. Four of them (Corpus Christi, 5 October, 1 November, 1 December) were suspended
    in 2013 to 2015 and restored in 2016; the holidays package and the fallback list both follow this."""
    try:
        import holidays as _h
        return set(_h.country_holidays(COUNTRY, years=list(years)).keys())
    except ImportError:
        fixed = [(1, 1), (4, 25), (5, 1), (6, 10), (8, 15), (12, 8), (12, 25)]
        restored = [(10, 5), (11, 1), (12, 1)]
        good_friday = {2015: (4, 3), 2016: (3, 25), 2017: (4, 14), 2018: (3, 30), 2019: (4, 19)}
        easter = {2015: (4, 5), 2016: (3, 27), 2017: (4, 16), 2018: (4, 1), 2019: (4, 21)}
        corpus = {2016: (5, 26), 2017: (6, 15), 2018: (5, 31), 2019: (6, 20)}
        out = {pd.Timestamp(y, m, d).date() for y in years for m, d in fixed}
        out |= {pd.Timestamp(y, m, d).date() for y in years if y >= 2016 for m, d in restored}
        for dct in (good_friday, easter, corpus):
            out |= {pd.Timestamp(y, *dct[y]).date() for y in years if y in dct}
        return out

def to_daily(s):
    loc = s.index.tz_convert(TZ)
    df = pd.DataFrame({"v": s.values, "day": loc.tz_localize(None).normalize(), "hour": loc.hour})
    M = df.groupby(["day", "hour"])["v"].mean().unstack("hour").reindex(columns=range(24))
    n = df.groupby("day")["hour"].nunique(); short = n.index[n == 23]
    if len(short): M.loc[short] = M.loc[short].interpolate(axis=1, limit=1, limit_area="inside")
    M.index = pd.DatetimeIndex(M.index); return M

E = load_energy(ENERGY_CSV); W = load_weather(WEATHER_CSV)
IDX = pd.date_range(E.index.min(), E.index.max(), freq="h", tz="UTC")
E, W = E.reindex(IDX), W.reindex(IDX).interpolate(limit=6, limit_area="inside")
GEN = {"gas": "generation fossil gas", "coal": "generation fossil hard coal", "nuclear": "generation nuclear",
       "hydro": "generation hydro water reservoir", "solar": "generation solar", "wind": "generation wind onshore"}
GEN = {k: v for k, v in GEN.items() if v in E}
LOAD = E["total load actual"].interpolate(limit=6, limit_area="inside")
HOLS = country_holidays(range(IDX.min().year, IDX.max().year + 1))
RATIO = float(LOAD.mean() / 28000)                          # Portuguese mean demand relative to the Spanish generator (28,000 MW)
print(len(IDX), "hours;", list(W.columns), "| generation:", list(GEN), f"| mean demand {LOAD.mean():.0f} MW, ratio to Spain {RATIO:.3f}")

# %%
def build_features(load):
    loc = IDX.tz_convert(TZ); day = loc.tz_localize(None).normalize(); h = loc.hour.values
    L = to_daily(load); days = pd.date_range(L.index.min(), L.index.max(), freq="D"); L = L.reindex(days)
    pos = pd.Series(np.arange(len(days)), index=days)
    di = pos.reindex(day).values                                   # day index of each hour
    Lv = L.values
    def day_shift(k): return np.where(di - k >= 0, di - k, -1).astype(int)
    def take(M, k, hour):                                          # M[day - k, hour]
        d = day_shift(k); out = np.full(len(d), np.nan); ok = (d >= 0) & ~np.isnan(di)
        out[ok] = M[d[ok], hour[ok] if np.ndim(hour) else hour]; return out
    X = pd.DataFrame(index=IDX)
    X["hour_sin"], X["hour_cos"] = np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24)
    X["dow_sin"], X["dow_cos"] = np.sin(2 * np.pi * loc.dayofweek / 7), np.cos(2 * np.pi * loc.dayofweek / 7)
    X["doy_sin"], X["doy_cos"] = np.sin(2 * np.pi * loc.dayofyear / 365.25), np.cos(2 * np.pi * loc.dayofyear / 365.25)
    X["weekend"] = (loc.dayofweek >= 5).astype(float)
    X["summer_time"] = np.array([bool(t.dst()) for t in loc], float)
    inh = lambda dd: np.array([x.date() in HOLS for x in dd], float)
    X["holiday"] = inh(day); X["pre_holiday"] = inh(day + pd.Timedelta(days=1)); X["post_holiday"] = inh(day - pd.Timedelta(days=1))
    X["bridge"] = (((loc.dayofweek == 0) & (X["pre_holiday"] == 1)) | ((loc.dayofweek == 4) & (X["post_holiday"] == 1))).astype(float)
    X["christmas"] = (((loc.month == 12) & (loc.day >= 24)) | ((loc.month == 1) & (loc.day <= 6))).astype(float)
    T = W["temp"]
    X["temp"] = T.values; X["hdd"], X["cdd"] = np.clip(18 - T, 0, None).values, np.clip(T - 18, 0, None).values
    Td = to_daily(T).reindex(days)
    for f in ("mean", "max", "min"):
        X[f"temp_day_{f}"] = getattr(Td, f)(axis=1).reindex(day).values
    X["temp_ewm24"], X["temp_ewm72"] = T.ewm(halflife=24).mean().values, T.ewm(halflife=72).mean().values
    for c in [c for c in W.columns if c != "temp"]:
        X[c] = W[c].values
    for c, nm in [("forecast solar day ahead", "solar_fc"), ("forecast wind onshore day ahead", "wind_fc")]:
        if c in E: X[nm] = E[c].interpolate(limit=6, limit_area="inside").values
    # demand history relative to the issue time (12:00 on D-1; last observation 11:00 on D-1)
    LK = LAST_KNOWN_HOUR
    X["load_last_obs"] = take(Lv, 1, LK)
    X["load_change_24h"] = take(Lv, 1, LK) - take(Lv, 2, LK)
    win24 = np.full((len(days), 24), np.nan)                       # 24 h before issue of day d: D-2 12..23 + D-1 0..11
    win24[2:, :24 - LK - 1] = Lv[:-2, LK + 1:]; win24[2:, 24 - LK - 1:] = Lv[1:-1, :LK + 1]
    for f in ("mean", "max", "min", "std"):
        X[f"load_24h_{f}"] = getattr(np, "nan" + f)(win24, axis=1)[np.where(np.isnan(di), 0, di).astype(int)]
    wk = np.full(len(days), np.nan)
    for d in range(8, len(days)):
        wk[d] = np.nanmean(np.r_[Lv[d - 8, LK + 1:], Lv[d - 7:d - 1].ravel(), Lv[d - 1, :LK + 1]])
    X["load_week_mean"] = wk[np.where(np.isnan(di), 0, di).astype(int)]
    for f in ("mean", "max", "min"):
        X[f"load_D2_{f}"] = take(getattr(np, "nan" + f)(Lv, axis=1)[:, None], 2, 0)
    X["load_same_hour_7d"], X["load_same_hour_14d"] = take(Lv, 7, h), take(Lv, 14, h)
    X["load_same_hour_latest"] = np.where(h <= LK, take(Lv, 1, h), take(Lv, 2, h))
    X["latest_is_D1"] = (h <= LK).astype(float)
    for k, c in GEN.items():
        G = to_daily(E[c].interpolate(limit=6, limit_area="inside")).reindex(days).values
        X[f"gen_{k}_at_issue"] = take(G, 1, LK)
    X.loc[np.isnan(di)] = np.nan
    return X

GROUPS = {"Hour of day": ["hour_sin", "hour_cos"], "Weekday": ["dow_sin", "dow_cos", "weekend"],
          "Holiday and special days": ["holiday", "pre_holiday", "post_holiday", "bridge", "christmas"],
          "Season": ["doy_sin", "doy_cos", "summer_time"],
          "Temperature": ["temp", "hdd", "cdd", "temp_day_mean", "temp_day_max", "temp_day_min", "temp_ewm24", "temp_ewm72"],
          "Other weather": ["humidity", "pressure", "wind_speed", "clouds_all"],
          "Renewable forecasts": ["solar_fc", "wind_fc"],
          "Gas and coal (consequences)": ["gen_gas_at_issue", "gen_coal_at_issue"],
          "Other generation": ["gen_nuclear_at_issue", "gen_hydro_at_issue", "gen_solar_at_issue", "gen_wind_at_issue"]}
def group_of(f):
    if f.startswith("load_") or f == "latest_is_D1": return "Demand history"
    return next((g for g, v in GROUPS.items() if f in v), "Other")

X_ALL = build_features(LOAD); Y_ALL = LOAD.copy()
FEATS = list(X_ALL.columns); FGROUP = pd.Series([group_of(f) for f in FEATS], index=FEATS)
print(len(FEATS), "features:", FGROUP.value_counts().to_dict())

# %%
# Leakage audit (original data)
issue_time = IDX.tz_convert(TZ).tz_localize(None).normalize() - pd.Timedelta(days=1) + pd.Timedelta(hours=ISSUE_HOUR)
def leakage_audit(load, builder, n):
    base = builder(load); rng = np.random.default_rng(0); viol, nch = [], 0
    okpos = np.flatnonzero(load.notna().values)
    for j in rng.choice(okpos[len(okpos) // 10:-24 * 30], n, replace=False):
        l2 = load.copy(); l2.iloc[j] += 1e5; X2 = builder(l2)
        ch = ~((X2.values == base.values) | (np.isnan(X2.values) & np.isnan(base.values))).all(axis=1); nch += ch.any()
        t_obs = IDX[j].tz_convert(TZ).tz_localize(None); bad = ch & (issue_time <= t_obs)
        if bad.any(): viol.append(f"demand at {t_obs} changes rows issued at {issue_time[bad][0]}")
    assert nch > 0
    if viol: raise RuntimeError("LEAKAGE:\n" + "\n".join(viol))
    return nch
print("leakage audit passed:", leakage_audit(LOAD, build_features, 6 if RUN_PLAN == "smoke" else 20), "perturbations")

# %% [markdown]
# ## 3. Injected ground truth (semi-synthetic real data)
#
# Three components are added to the **real** data; everything else stays real.
#
# * **Event**: on `n_event_days` randomly chosen non-holiday weekdays, demand is lowered by exactly `event_mw` between `event_hours`. The model sees an `event_flag` known in advance, like a holiday. True contribution: *C*_event.
# * **Driver**: an exogenous index *z* (smooth daily process, known at issue time like a forecast) with a nonlinear effect that grows with heat: *C*_z = `driver_mw` · tanh(1.2 *z*) · (1 + `driver_temp_interaction` · cooling degrees / 5). True contribution: *C*_z, credited to the driver.
# * **Decoy**: a variable correlated with *z* (correlation `decoy_corr`) with **no effect**. Any attribution to it is an error.
#
# The same injected series is used for every model and seed.
#
# The sizes in MW (`event_mw`, `driver_mw`) are the Spanish values multiplied by the ratio of mean demand, so the effects are as large relative to demand as in Spain.

# %%
def make_injection(cfg):
    rng = np.random.default_rng(cfg["seed"])
    loc = IDX.tz_convert(TZ); day = loc.tz_localize(None).normalize()
    days = pd.DatetimeIndex(sorted(set(day)))
    cand = [d for d in days[30:-2] if d.dayofweek < 5 and d.date() not in HOLS]
    ev_days = set(pd.DatetimeIndex(rng.choice(cand, cfg["n_event_days"], replace=False)))
    h0, h1 = cfg["event_hours"]
    flag = np.array([(d in ev_days) and (h0 <= hh <= h1) for d, hh in zip(day, loc.hour)], float)
    zd = np.zeros(len(days)); e = rng.normal(0, np.sqrt(1 - 0.9 ** 2), len(days))
    for i in range(1, len(days)): zd[i] = 0.9 * zd[i - 1] + e[i]
    di = pd.Series(np.arange(len(days)), index=days).reindex(day).values
    z = np.interp(di + loc.hour / 24, np.arange(len(days)), zd) + rng.normal(0, 0.05, len(IDX))
    decoy = cfg["decoy_corr"] * z + np.sqrt(1 - cfg["decoy_corr"] ** 2) * rng.normal(0, 1, len(IDX))
    cdd = np.clip(W["temp"].values - 18, 0, None)
    C_event = cfg["event_mw"] * flag
    C_z = cfg["driver_mw"] * np.tanh(1.2 * z) * (1 + cfg["driver_temp_interaction"] * np.nan_to_num(cdd) / 5)
    truth = pd.DataFrame({"C_event": C_event, "C_z": C_z}, index=IDX)
    extra = pd.DataFrame({"event_flag": flag, "driver_z": z, "decoy_d": decoy}, index=IDX)
    return LOAD + C_event + C_z, extra, truth, ev_days

INJECT_PT = {**INJECT, "event_mw": round(INJECT["event_mw"] * RATIO), "driver_mw": round(INJECT["driver_mw"] * RATIO)}   # scaled to Portugal
LOAD_INJ, EXTRA_INJ, TRUTH_INJ, EVENT_DAYS = make_injection(INJECT_PT)
def build_features_injected(load):
    X = build_features(load)
    for c in EXTRA_INJ.columns: X[c] = EXTRA_INJ[c].values
    return X
GROUPS.update({"Injected event": ["event_flag"], "Injected driver": ["driver_z"], "Injected decoy": ["decoy_d"]})
print(f"{len(EVENT_DAYS)} event days; driver effect sd {TRUTH_INJ['C_z'].std():.0f} MW; "
      f"corr(driver, decoy) = {np.corrcoef(EXTRA_INJ['driver_z'], EXTRA_INJ['decoy_d'])[0, 1]:.2f}")
print("leakage audit (injected):", leakage_audit(LOAD_INJ, build_features_injected, 4 if RUN_PLAN == "smoke" else 10), "perturbations")

# %% [markdown]
# ## 4. Datasets, split, windows

# %%
loc_day = IDX.tz_convert(TZ).tz_localize(None).normalize()
_X0 = build_features(LOAD)
_valid0 = _X0.notna().all(axis=1).values & LOAD.notna().values
_dv = pd.DatetimeIndex(sorted(set(loc_day[_valid0]))); _n = len(_dv)
VAL_START, TEST_START = _dv[int(SPLIT[0] * _n)], _dv[int((SPLIT[0] + SPLIT[1]) * _n)]
PART = np.where(loc_day < VAL_START, 0, np.where(loc_day < TEST_START, 1, 2))

def prepare(X, Y, extra_mask, seed=99):
    valid = X.notna().all(axis=1).values & Y.notna().values
    vr = pd.Series(valid.astype(int)).rolling(WINDOW).sum().fillna(0).values == WINDOW
    I = [np.flatnonzero(vr & (PART == k)) for k in range(3)]
    mu, sd = np.nanmean(X.values[I[0]], 0), np.nanstd(X.values[I[0]], 0) + 1e-9
    ym, ysd = np.nanmean(Y.values[I[0]]), np.nanstd(Y.values[I[0]])
    XS = np.nan_to_num(((X.values - mu) / sd).astype(np.float32)); YS = ((Y.values - ym) / ysd).astype(np.float32)
    rand = np.random.default_rng(seed).choice(I[2], min(N_EXPLAIN, len(I[2])), replace=False)
    extra = np.flatnonzero(extra_mask & vr & (PART == 2))
    sample = np.unique(np.r_[rand, extra])
    feats = list(X.columns)
    return dict(X=X, Y=Y, XS=XS, YS=YS, ym=ym, ysd=ysd, I_TR=I[0], I_VA=I[1], I_TE=I[2], SAMPLE=sample,
                feats=feats, group=np.array([group_of(f) for f in feats]), n_rand=len(rand))

loc_all = IDX.tz_convert(TZ)
hol_wd = (_X0["holiday"].values == 1) & (loc_all.dayofweek < 5)
DATA = {"real": prepare(_X0, LOAD, hol_wd)}
_XI = build_features_injected(LOAD_INJ)
DATA["injected"] = prepare(_XI, LOAD_INJ, EXTRA_INJ["event_flag"].values == 1)
for k, P in DATA.items():
    print(f"{k:9s}: {len(P['feats'])} features, windows {len(P['I_TR'])}/{len(P['I_VA'])}/{len(P['I_TE'])}, explained {len(P['SAMPLE'])}")
print(f"split: val from {VAL_START.date()}, test from {TEST_START.date()}")

# %% [markdown]
# ## 5. Models and explainers (unchanged)

# %%
import torch, torch.nn as nn
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", DEVICE, "| torch", torch.__version__, "| shap", shap.__version__ if HAVE_SHAP else "missing")
assert HAVE_SHAP, "pip install shap"

class LSTMNet(nn.Module):
    def __init__(self, F, hidden, layers, dropout):
        super().__init__()
        self.rnn = nn.LSTM(F, hidden, layers, batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(self, x):
        h, _ = self.rnn(x); return self.head(h[:, -1]).squeeze(-1)

class RETAINNet(nn.Module):
    def __init__(self, F, hidden, dropout):
        super().__init__()
        self.emb = nn.Linear(F, hidden)
        self.gru_a = nn.GRU(hidden, hidden, batch_first=True); self.gru_b = nn.GRU(hidden, hidden, batch_first=True)
        self.att_a = nn.Linear(hidden, 1); self.att_b = nn.Linear(hidden, hidden)
        self.drop = nn.Dropout(dropout); self.out = nn.Linear(hidden, 1)
    def attention(self, x):
        v = self.emb(x)
        rv = torch.flip(self.drop(v), [1])                       # reverse time
        g, _ = self.gru_a(rv); h, _ = self.gru_b(rv)
        g, h = torch.flip(g, [1]), torch.flip(h, [1])
        alpha = torch.softmax(self.att_a(g).squeeze(-1), dim=1)   # (n, T)
        beta = torch.tanh(self.att_b(h))                         # (n, T, H)
        return v, alpha, beta
    def forward(self, x):
        v, a, b = self.attention(x)
        return self.out((a.unsqueeze(-1) * b * v).sum(1)).squeeze(-1)
    def contributions(self, x):
        """Exact contribution (n, T, F) and the feature-independent bias (n,)."""
        v, a, b = self.attention(x)
        w, W = self.out.weight.squeeze(0), self.emb.weight          # (H,), (H, F)
        coef = torch.einsum("nt,nth,h,hf->ntf", a, b, w, W)
        bias = ((a.unsqueeze(-1) * b * self.emb.bias).sum(1) * w).sum(-1) + self.out.bias
        return x * coef, bias

class PosEnc(nn.Module):
    def __init__(self, d, T=512):
        super().__init__()
        pe = torch.zeros(T, d); pos = torch.arange(T).unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2) * (-np.log(10000.0) / d))
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer("pe", pe)
    def forward(self, x): return x + self.pe[: x.shape[1]]

class EncLayer(nn.Module):
    def __init__(self, d, heads, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.ReLU(), nn.Linear(4 * d, d))
        self.n1, self.n2, self.drop = nn.LayerNorm(d), nn.LayerNorm(d), nn.Dropout(dropout)
        self.last_w = None
    def forward(self, x, keep=False):
        a, w = self.attn(x, x, x, need_weights=keep, average_attn_weights=True)
        if keep: self.last_w = w                                  # (n, T, T)
        x = self.n1(x + self.drop(a)); return self.n2(x + self.drop(self.ff(x)))

class TransformerNet(nn.Module):
    def __init__(self, F, d_model, heads, layers, dropout):
        super().__init__()
        self.inp, self.pos = nn.Linear(F, d_model), PosEnc(d_model)
        self.layers = nn.ModuleList([EncLayer(d_model, heads, dropout) for _ in range(layers)])
        self.head = nn.Linear(d_model, 1)
    def forward(self, x, keep=False):
        z = self.pos(self.inp(x))
        for L in self.layers: z = L(z, keep)
        return self.head(z[:, -1]).squeeze(-1)
    def attention_score(self, x):
        """Section 4.2.2: attention of the last position on each step x |input|, summed over time (unsigned)."""
        self.forward(x, keep=True)
        w = self.layers[-1].last_w[:, -1, :]                      # (n, T)
        return (w.unsqueeze(-1) * x.abs()).sum(1)

class GLU(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__(); self.l = nn.Linear(d_in, 2 * d_out)
    def forward(self, x):
        a, b = self.l(x).chunk(2, dim=-1); return a * torch.sigmoid(b)

class GRN(nn.Module):
    """Gated residual network of the TFT."""
    def __init__(self, d_in, d_hid, d_out, dropout):
        super().__init__()
        self.skip = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()
        self.l1, self.l2 = nn.Linear(d_in, d_hid), nn.Linear(d_hid, d_hid)
        self.drop, self.glu, self.norm = nn.Dropout(dropout), GLU(d_hid, d_out), nn.LayerNorm(d_out)
    def forward(self, a):
        h = self.drop(self.l2(nn.functional.elu(self.l1(a))))
        return self.norm(self.skip(a) + self.glu(h))

class InterpretableMHA(nn.Module):
    """TFT interpretable multi-head attention: per-head queries/keys, shared values, heads averaged."""
    def __init__(self, d, heads, dropout):
        super().__init__()
        self.dk = d // heads
        self.q = nn.ModuleList([nn.Linear(d, self.dk) for _ in range(heads)])
        self.k = nn.ModuleList([nn.Linear(d, self.dk) for _ in range(heads)])
        self.v, self.o, self.drop = nn.Linear(d, self.dk), nn.Linear(self.dk, d), nn.Dropout(dropout)
    def forward(self, q, k):
        v, outs, atts = self.v(k), [], []
        for qi, ki in zip(self.q, self.k):
            a = torch.softmax(qi(q) @ ki(k).transpose(1, 2) / self.dk ** 0.5, dim=-1)
            atts.append(a); outs.append(self.drop(a) @ v)
        return self.o(torch.stack(outs).mean(0)), torch.stack(atts).mean(0)

class TFTNet(nn.Module):
    """Temporal Fusion Transformer (Lim et al., 2021), single-horizon version for one target hour:
    variable selection network -> LSTM encoder with gated skip -> enrichment GRN ->
    interpretable attention from the last step -> gated position-wise GRN -> output.
    Simplifications: no static covariates; one GRN shared by all variables (the original uses one per variable)."""
    def __init__(self, F, d_model, heads, dropout):
        super().__init__()
        d = d_model
        self.emb_w, self.emb_b = nn.Parameter(torch.randn(F, d) * 0.1), nn.Parameter(torch.zeros(F, d))
        self.vsn, self.var_grn = GRN(F * d, d, F, dropout), GRN(d, d, d, dropout)
        self.lstm = nn.LSTM(d, d, batch_first=True)
        self.gate1, self.norm1 = GLU(d, d), nn.LayerNorm(d)
        self.enrich = GRN(d, d, d, dropout)
        self.attn = InterpretableMHA(d, heads, dropout)
        self.gate2, self.norm2 = GLU(d, d), nn.LayerNorm(d)
        self.ff, self.gate3, self.norm3 = GRN(d, d, d, dropout), GLU(d, d), nn.LayerNorm(d)
        self.out = nn.Linear(d, 1)
        self.last_w = None
    def select(self, x):
        e = x.unsqueeze(-1) * self.emb_w + self.emb_b                         # (n, T, F, d)
        n, T, F_, d = e.shape
        w = torch.softmax(self.vsn(e.reshape(n, T, F_ * d)), dim=-1)           # (n, T, F) selection weights
        return (w.unsqueeze(-1) * self.var_grn(e)).sum(2), w
    def forward(self, x, keep=False):
        z, w = self.select(x)
        h, _ = self.lstm(z)
        h = self.enrich(self.norm1(z + self.gate1(h)))
        last = h[:, -1:]
        a, _ = self.attn(last, h)
        y = self.norm2(last + self.gate2(a))
        y = self.norm3(y + self.gate3(self.ff(y)))
        if keep: self.last_w = w
        return self.out(y[:, 0]).squeeze(-1)
    def selection_weights(self, x):
        """TFT's built-in importance: variable-selection weights averaged over the window (unsigned, sum to 1)."""
        self.forward(x, keep=True); return self.last_w.mean(1)

def build_net(kind, F):
    a = ARCH[kind]
    if kind == "LSTM": return LSTMNet(F, a["hidden"], a["layers"], a["dropout"])
    if kind == "RETAIN": return RETAINNet(F, a["hidden"], a["dropout"])
    if kind == "TFT": return TFTNet(F, a["d_model"], a["heads"], a["dropout"])
    return TransformerNet(F, a["d_model"], a["heads"], a["layers"], a["dropout"])

class Out2D(nn.Module):
    def __init__(self, net): super().__init__(); self.net = net
    def forward(self, x): return self.net(x).unsqueeze(-1)

class SeqModel:
    """Train / predict / explain one network on standardized windows (n, T, F)."""
    def __init__(self, kind, F, seed):
        torch.manual_seed(seed); np.random.seed(seed)
        self.kind, self.net = kind, build_net(kind, F).to(DEVICE)
    def _t(self, a): return torch.as_tensor(a, dtype=torch.float32, device=DEVICE)
    def fit(self, Xtr, ytr, Xva, yva):
        opt = torch.optim.AdamW(self.net.parameters(), lr=TRAIN["lr"], weight_decay=TRAIN["weight_decay"])
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)
        loss_fn = nn.HuberLoss(); best, best_state, bad = np.inf, None, 0
        Xtr_t, ytr_t = self._t(Xtr), self._t(ytr); n = len(Xtr)
        self.epochs = 0
        for ep in range(TRAIN["max_epochs"]):
            self.net.train(); perm = torch.randperm(n, device=DEVICE)
            for i in range(0, n, TRAIN["batch"]):
                idx = perm[i:i + TRAIN["batch"]]
                opt.zero_grad(); loss = loss_fn(self.net(Xtr_t[idx]), ytr_t[idx]); loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), TRAIN["clip"]); opt.step()
            val = float(np.mean(np.abs(self.predict(Xva) - yva)))
            sched.step(val); self.epochs = ep + 1
            if val < best - 1e-5:
                best, bad = val, 0; best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
            else:
                bad += 1
                if bad >= TRAIN["patience"]: break
        self.net.load_state_dict(best_state); self.net.eval(); return self
    @torch.no_grad()
    def predict(self, Xw, bs=2048):
        self.net.eval()
        return np.concatenate([self.net(self._t(Xw[i:i + bs])).cpu().numpy() for i in range(0, len(Xw), bs)])
    def shap(self, Xw, background, seed=0):
        self.net.eval()
        wrapped = Out2D(self.net).eval()                          # shap expects (n, 1) outputs
        with torch.backends.cudnn.flags(enabled=False):          # RNN backward in eval mode on GPU
            ex = shap.GradientExplainer(wrapped, self._t(background))
            sv = ex.shap_values(self._t(Xw), nsamples=SHAP_NSAMPLES, rseed=seed)
        sv = np.asarray(sv[0] if isinstance(sv, list) else sv)
        return sv.reshape(len(Xw), Xw.shape[1], Xw.shape[2])     # (n, T, F)
    @torch.no_grad()
    def intrinsic(self, Xw):
        self.net.eval(); x = self._t(Xw)
        if self.kind == "RETAIN":
            c, bias = self.net.contributions(x)
            err = (c.sum((1, 2)) + bias - self.net(x)).abs().max().item()
            assert err < 1e-3, f"RETAIN identity violated ({err:.2e})"
            return c.cpu().numpy(), err
        if self.kind == "Transformer":
            return self.net.attention_score(x).cpu().numpy(), None
        if self.kind == "TFT":
            return self.net.selection_weights(x).cpu().numpy(), None
        return None, None

# %%
def windows(Xs, idx, T=WINDOW):
    return np.stack([Xs[i - T + 1:i + 1] for i in idx]).astype(np.float32)

def seq_deletion_insertion(model, Xw, A, K=10):
    n, T, F = Xw.shape; order = np.argsort(-np.abs(A), axis=1)
    f0 = model.predict(Xw); scale = max(np.std(f0), 1e-9); dele, ins = [], []
    for m in range(1, min(K, F) + 1):
        M = Xw.copy()
        for i in range(n): M[i, :, order[i, :m]] = 0.0
        dele.append(np.mean(np.abs(f0 - model.predict(M))))
        M = np.zeros_like(Xw)
        for i in range(n): M[i, :, order[i, :m]] = Xw[i, :, order[i, :m]]
        ins.append(np.mean(np.abs(model.predict(M) - f0)))
    return float(np.mean(dele) / scale), float(1 - np.mean(ins) / scale)

def pair_agreement(a, b, k=5):
    """Agreement of two explanations of the same forecasts (Section 4.3)."""
    out = []
    for x, y in zip(a, b):
        ta, tb = np.argsort(-np.abs(x))[:k], np.argsort(-np.abs(y))[:k]
        out.append((stats.spearmanr(np.abs(x), np.abs(y)).statistic, stats.weightedtau(np.abs(x), np.abs(y)).statistic,
                    len(set(ta) & set(tb)) / len(set(ta) | set(tb)), np.mean(np.sign(x[tb]) == np.sign(y[tb]))))
    return np.nanmean(np.array(out), axis=0)

# %%
def _ig(self, Xw, steps=IG_STEPS, bs=64):
    """Integrated Gradients from the training mean (0 after standardization), per time step and feature."""
    self.net.eval(); out = []
    with torch.backends.cudnn.flags(enabled=False):
        for i in range(0, len(Xw), bs):
            x = self._t(Xw[i:i + bs]); total = torch.zeros_like(x)
            for a in torch.arange(1, steps + 1, device=DEVICE, dtype=torch.float32) / steps:
                xi = (a * x).detach().requires_grad_(True)
                g, = torch.autograd.grad(self.net(xi).sum(), xi); total += g
            out.append((total / steps * x).detach().cpu().numpy())
    return np.concatenate(out)
SeqModel.ig = _ig

class XGBRef:
    def __init__(self, seed, p):
        self.m = xgb.XGBRegressor(n_estimators=3000 if RUN_PLAN != "smoke" else 200, max_depth=p.get("max_depth", 6),
                                  learning_rate=p.get("lr", 0.03), min_child_weight=p.get("min_child_weight", 5),
                                  subsample=0.8, colsample_bytree=p.get("colsample", 0.8), early_stopping_rounds=100,
                                  n_jobs=-1, random_state=seed, verbosity=0)
    def fit(self, Xtr, ytr, Xva, yva):
        self.m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False); self.ex = shap.TreeExplainer(self.m); return self
    def predict(self, X): return self.m.predict(X)
    def shap(self, X): return np.asarray(self.ex.shap_values(X))

ABL_GROUPS = ["Demand history", "Hour of day", "Weekday", "Holiday and special days", "Season", "Temperature",
              "Other weather", "Renewable forecasts", "Gas and coal (consequences)", "Other generation"]
def group_ablation(predict, Xex, Xbg, group, seq):
    """Model-level effect of each feature group: f(x) minus the mean forecast when the group's inputs are replaced
    by values from random background (training) inputs. Returns (n, groups) in standardized target units."""
    f0 = predict(Xex); rng = np.random.default_rng(0); out = {}
    for g in ABL_GROUPS:
        cols = np.flatnonzero(group == g)
        if not len(cols): continue
        acc = np.zeros(len(Xex))
        for b in rng.choice(len(Xbg), ABLATION_K, replace=False):
            M = Xex.copy()
            if seq: M[:, :, cols] = Xbg[b][:, cols]
            else:   M[:, cols] = Xbg[b][cols]
            acc += predict(M)
        out[g] = f0 - acc / ABLATION_K
    return out

# %% [markdown]
# ## 6. Synthetic benchmark calibrated to Portugal
#
# The Spanish generator is used unchanged, with the Portuguese calendar and time zone. Its demand parameters are fitted to the Portuguese training data with one regression (daily profile as two waves, Saturday, Sunday and holiday drops, heating below 16 °C and cooling above 22 °C, with the Spanish thresholds and curvature), and its climate (mean, seasonal and daily range) to the Portuguese temperature. The sizes that cannot be fitted (hidden driver, shocks, noise, generation) are the Spanish values multiplied by the ratio of mean demand. Portugal has no nuclear power, so that input is constant. Scoring, features, split and explained hours are those of the paper.

# %%
BENCH_CODE = r'''
import itertools, json, re, time, warnings
import numpy as np, pandas as pd
from scipy import stats
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.preprocessing import StandardScaler
try:
    import xgboost as xgb
except ImportError:
    xgb = None
try:
    import shap
except ImportError:
    shap = None

from pathlib import Path

OUT_DIR = Path("results/benchmark_experiments")
TZ = "Europe/Madrid"
START, END = "2015-01-01 00:00", "2018-12-31 23:00"
SPLIT = (0.64, 0.16, 0.20)
LAG_L = 36                      # strict day-ahead (fixed-lag protocol, as in Chapter 8)
SEEDS = [0, 1, 2, 3, 4]
N_EXPLAIN = 1000                # explained test windows per run (as in the report)
SEED_SAMPLE = 0

# ---- conditions: variants (Section 5.3), hidden-driver sweep, interaction variant ----
CONDITIONS = [
    dict(name="default",       change="2016-07-01", latent=1200, interaction=0.0),
    dict(name="extrapolation", change="2017-07-01", latent=1200, interaction=0.0),
    dict(name="stationary",    change=None,         latent=1200, interaction=0.0),
    dict(name="latent_0",      change="2016-07-01", latent=0,    interaction=0.0),
    dict(name="latent_600",    change="2016-07-01", latent=600,  interaction=0.0),
    dict(name="latent_2400",   change="2016-07-01", latent=2400, interaction=0.0),
    dict(name="interaction",   change="2016-07-01", latent=1200, interaction=0.5),
]

# ---- generator parameters (Section 5.2). ASSUMED = not stated in the draft ----
GEN = dict(
    base=28000, hour_amp=(2600, 900), hour_phase=(9, 3),     # phases ASSUMED
    sat=-1800, sun=-3200, holiday=-2500,
    heat_thr=16, heat_slope=250, cool_thr=22, cool_coef=150, cool_exp=1.3, cool_factor_after=1.6,
    latent_phi=0.995, ar_phi=0.95, ar_sd=500,                 # ar_sd (stationary sd, MW) ASSUMED
    noise_sd=150, operator_sd=200, gas_share=0.55, gas_latent=400, gas_noise_sd=300,   # gas_latent, gas_noise_sd ASSUMED
)
USE_EXTERNAL_GENERATOR = None   # e.g. "data/synthetic/{name}_seed{seed}" with energy_dataset.csv, weather_features.csv, truth_components.csv
WRITE_FILES = False             # write each realization in Kaggle format for the main pipeline / LEAR notebook
FILES_DIR = Path("data/synthetic")

# ---- models and hooks ----
USE_XGB = True
EXTERNAL_ATTRIBUTIONS = {}      # {"SHAP on RETAIN": "ext/{name}_seed{seed}_retain_shap.csv"}  columns: time + feature columns
EXTERNAL_FEATURE_GROUPS = {}    # regex -> group for external feature names, e.g. {r"^hour_": "hour", r"^dow_|weekend": "week", ...}
EXTERNAL_MODELS = {}            # {"RETAIN": wrapper_instance} with .fit(X, y), .predict(X), .explain(X) -> (n, F) in MW

# ---- runtime ----
FAST_MODE = False               # smoke test: 2 conditions, 2 seeds, 1 year, no XGBoost tuning
RUN_CAUSAL = True
CAUSAL_SEEDS = 3                # PCMCI+ takes minutes per run; seeds used for causal discovery
CAUSAL_HOURS = 24 * 365         # hours of the training period used for causal discovery
def spanish_holidays(years):
    try:
        import holidays as _h
        return set(_h.Spain(years=years).keys())
    except ImportError:
        fixed = [(1, 1), (1, 6), (5, 1), (8, 15), (10, 12), (11, 1), (12, 6), (12, 8), (12, 25)]
        gf = {2015: (4, 3), 2016: (3, 25), 2017: (4, 14), 2018: (3, 30), 2019: (4, 19)}
        out = {pd.Timestamp(y, m, d).date() for y in years for m, d in fixed}
        return out | {pd.Timestamp(y, *gf[y]).date() for y in years if y in gf}

def _ar(n, phi, sd, rng):
    """AR(1) with stationary standard deviation sd."""
    e = rng.normal(0, sd * np.sqrt(1 - phi ** 2), n); x = np.empty(n); x[0] = rng.normal(0, sd)
    for i in range(1, n): x[i] = phi * x[i - 1] + e[i]
    return x

CITIES = {"Madrid": 0.0, " Barcelona": 1.5, "Valencia": 2.5, "Seville": 4.0, "Bilbao": -1.5}

def generate(cond, seed, g=GEN):
    rng = np.random.default_rng(1000 * seed + 17)
    idx = pd.date_range(pd.Timestamp(START, tz=TZ).tz_convert("UTC"),
                        pd.Timestamp(END, tz=TZ).tz_convert("UTC"), freq="h")
    loc = idx.tz_convert(TZ); n = len(idx)
    h, dow, doy = loc.hour.values, loc.dayofweek.values, loc.dayofyear.values
    day = loc.tz_localize(None).normalize()
    hols = spanish_holidays(range(loc.year.min(), loc.year.max() + 1))
    is_hol = np.array([d.date() in hols for d in day])

    # weather: five cities, seasonal + daily cycle + persistent national anomaly + local noise
    anom = _ar(n, 0.995, 2.5, rng)
    temps = {c: 15 + off - 9 * np.cos(2 * np.pi * (doy - 20) / 365.25) + (5 + 0.3 * off) * np.sin(2 * np.pi * (h - 9) / 24)
                + anom + _ar(n, 0.9, 0.8, rng) for c, off in CITIES.items()}
    T = np.mean(list(temps.values()), axis=0)
    clouds = 100 / (1 + np.exp(-_ar(n, 0.98, 1.5, rng)))
    humidity = np.clip(60 + 0.3 * (clouds - 50) - 1.2 * (T - 15) + rng.normal(0, 4, n), 10, 100)
    pressure = 1015 + _ar(n, 0.99, 6, rng)
    wind_speed = np.abs(4 + _ar(n, 0.97, 2.5, rng))

    # demand components
    C_hour = g["hour_amp"][0] * np.sin(2 * np.pi * (h - g["hour_phase"][0]) / 24) + \
             g["hour_amp"][1] * np.sin(4 * np.pi * (h - g["hour_phase"][1]) / 24)
    C_week = np.where(dow == 5, g["sat"], np.where(dow == 6, g["sun"], 0)).astype(float)
    C_hol = np.where(is_hol, g["holiday"], 0).astype(float)
    k = np.ones(n)
    if cond["change"]:
        k[loc.tz_localize(None) >= pd.Timestamp(cond["change"])] = g["cool_factor_after"]
    heat = g["heat_slope"] * np.clip(g["heat_thr"] - T, 0, None)
    cool = k * g["cool_coef"] * np.clip(T - g["cool_thr"], 0, None) ** g["cool_exp"]
    C_temp = heat + cool
    C_int = cond["interaction"] * cool * np.sin(2 * np.pi * (h - 10) / 24)
    days_u, day_pos = np.unique(day.values, return_inverse=True)
    z = _ar(len(days_u), g["latent_phi"], 1.0, rng)
    z_h = np.interp(day_pos + h / 24, np.arange(len(days_u)), z)
    C_lat = cond["latent"] * z_h
    C_AR = _ar(n, g["ar_phi"], g["ar_sd"], rng)
    eps = rng.normal(0, g["noise_sd"], n)
    D = g["base"] + C_hour + C_week + C_hol + C_temp + C_int + C_lat + C_AR + eps

    # generation: consequences of demand (gas, coal) and independent sources
    season = 0.6 + 0.4 * np.sin(2 * np.pi * (doy - 80) / 365.25)
    solar = 7000 * np.clip(np.sin(np.pi * (h - 7) / 13), 0, None) * season * (1 - 0.6 * clouds / 100)
    wind = np.minimum(15000, 60 * wind_speed ** 3)
    nuclear = np.full(n, 7000.0)
    for s in rng.choice(np.arange(0, n - 24 * 30), 3, replace=False):
        nuclear[s:s + 24 * 30] -= 1000
    hydro = 3000 + 1500 * np.sin(2 * np.pi * (doy - 60) / 365.25) + rng.normal(0, 150, n)
    resid = D - solar - wind - nuclear - hydro
    gas = g["gas_share"] * resid + g["gas_latent"] * z_h * (cond["latent"] > 0) + _ar(n, 0.9, g.get("gas_noise_sd", 0), rng)
    coal = np.clip(resid - gas, 0, None)
    ar_known = np.r_[np.full(LAG_L, 0.0), C_AR[:-LAG_L]] * g["ar_phi"] ** LAG_L
    operator = D - eps - C_AR + ar_known + rng.normal(0, g["operator_sd"], n)

    energy = pd.DataFrame({
        "generation solar": solar, "generation wind onshore": wind, "generation nuclear": nuclear,
        "generation hydro water reservoir": hydro, "generation fossil gas": gas, "generation fossil hard coal": coal,
        "forecast solar day ahead": solar + rng.normal(0, 200, n), "forecast wind onshore day ahead": wind + rng.normal(0, 800, n),
        "total load forecast": operator, "total load actual": D,
        "price day ahead": 30 + 0.002 * gas + rng.normal(0, 3, n)}, index=idx)
    weather = dict(temp=T, humidity=humidity, pressure=pressure, wind_speed=wind_speed, clouds_all=clouds, cities=temps)
    truth = pd.DataFrame(dict(C_hour=C_hour, C_week=C_week, C_holiday=C_hol, C_temp=C_temp, C_interaction=C_int,
                              C_latent=C_lat, C_AR=C_AR, eps=eps, ar_known=ar_known), index=idx)
    return energy, weather, truth, hols

def write_kaggle(energy, weather, truth, folder):
    folder.mkdir(parents=True, exist_ok=True)
    t = energy.index.tz_convert(TZ).strftime("%Y-%m-%d %H:%M:%S%z")
    t = [s[:-2] + ":" + s[-2:] for s in t]
    e = energy.copy(); e.insert(0, "time", t); e["price actual"] = e["price day ahead"]
    e.to_csv(folder / "energy_dataset.csv", index=False)
    rows = [pd.DataFrame({"dt_iso": t, "city_name": c, "temp": v + 273.15, "humidity": weather["humidity"],
                          "pressure": weather["pressure"], "wind_speed": weather["wind_speed"],
                          "clouds_all": weather["clouds_all"]}) for c, v in weather["cities"].items()]
    pd.concat(rows).to_csv(folder / "weather_features.csv", index=False)
    tr = truth.copy(); tr.insert(0, "time", t); tr.to_csv(folder / "truth_components.csv", index=False)


GROUPS = {"hour": ["hour_sin", "hour_cos"], "week": ["dow_sin", "dow_cos", "weekend"], "holiday": ["holiday"],
          "temperature": ["temp", "hdd", "cdd", "temp_day_mean", "temp_day_max", "temp_day_min"]}
TRUTH_COL = {"hour": "C_hour", "week": "C_week", "holiday": "C_holiday", "temperature": "C_temp"}

def build_features(energy, weather, hols, L=LAG_L):
    loc = energy.index.tz_convert(TZ); day = loc.tz_localize(None).normalize()
    X = pd.DataFrame(index=energy.index)
    X["hour_sin"], X["hour_cos"] = np.sin(2 * np.pi * loc.hour / 24), np.cos(2 * np.pi * loc.hour / 24)
    X["dow_sin"], X["dow_cos"] = np.sin(2 * np.pi * loc.dayofweek / 7), np.cos(2 * np.pi * loc.dayofweek / 7)
    X["doy_sin"], X["doy_cos"] = np.sin(2 * np.pi * loc.dayofyear / 365.25), np.cos(2 * np.pi * loc.dayofyear / 365.25)
    X["weekend"] = (loc.dayofweek >= 5).astype(float)
    inh = lambda dd: np.array([x.date() in hols for x in dd], float)
    X["holiday"] = inh(day)
    X["pre_holiday"], X["post_holiday"] = inh(day + pd.Timedelta(days=1)), inh(day - pd.Timedelta(days=1))
    X["bridge"] = (((loc.dayofweek == 0) & (X["pre_holiday"] == 1)) | ((loc.dayofweek == 4) & (X["post_holiday"] == 1))).astype(float)
    T = pd.Series(weather["temp"], index=energy.index)
    X["temp"] = T
    for c in ("humidity", "pressure", "wind_speed", "clouds_all"):
        X[c] = weather[c]
    X["hdd"], X["cdd"] = np.clip(18 - T, 0, None), np.clip(T - 18, 0, None)
    gb = T.groupby(day.values)
    X["temp_day_mean"], X["temp_day_max"], X["temp_day_min"] = (gb.transform(f).values for f in ("mean", "max", "min"))
    y = energy["total load actual"]
    for lag in sorted({L, 48, 168} | {l for l in (24, 48, 168) if l >= L}):
        X[f"load_lag{lag}"] = y.shift(lag)
    X["load_rollmean24"] = y.rolling(24).mean().shift(L)
    X["load_rollmean168"] = y.rolling(168).mean().shift(L)
    X["load_rollstd6"] = y.rolling(6).std().shift(L)
    X["load_diff1"] = y.diff(1).shift(L); X["load_diff24"] = y.diff(24).shift(L)
    gl = max(24, L)
    for c, nm in [("generation solar", "solar"), ("generation wind onshore", "wind_gen"), ("generation nuclear", "nuclear"),
                  ("generation hydro water reservoir", "hydro"), ("generation fossil gas", "gas"), ("generation fossil hard coal", "coal")]:
        X[f"{nm}_lag{gl}"] = energy[c].shift(gl)
    ok = X.notna().all(axis=1)
    return X[ok], y[ok]

def role(f):
    if any(f in v for v in GROUPS.values()): return "direct cause"
    if f.startswith("load_"): return "demand history"
    if f.startswith(("gas_", "coal_")): return "consequence"
    return "irrelevant"

def concept(f):
    for g, v in GROUPS.items():
        if f in v: return {"week": "day of week"}.get(g, g)
    if f.startswith("load_"): return "demand history"
    return re.sub(r"_lag\d+$", "", f)

TRUE_PARENTS = {"hour", "day of week", "holiday", "temperature", "demand history"}

class RidgeSHAP:
    name = "Ridge, linear SHAP"
    def fit(self, X, y, Xv=None, yv=None):
        self.sc = StandardScaler().fit(X); self.m = RidgeCV(alphas=np.logspace(-2, 4, 25)).fit(self.sc.transform(X), y)
        self.mu = self.sc.transform(X).mean(0); return self
    def predict(self, X): return self.m.predict(self.sc.transform(X))
    def explain(self, X): return (self.sc.transform(X) - self.mu) * self.m.coef_

class XGBTreeSHAP:
    name = "XGBoost, TreeSHAP"
    def fit(self, X, y, Xv=None, yv=None):
        params = dict(n_estimators=2000, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                      min_child_weight=5, early_stopping_rounds=50, n_jobs=-1, verbosity=0)
        if FAST_MODE: params.update(n_estimators=300, early_stopping_rounds=20)
        self.m = xgb.XGBRegressor(**params).fit(X, y, eval_set=[(Xv, yv)], verbose=False)
        self.ex = shap.TreeExplainer(self.m); return self
    def predict(self, X): return self.m.predict(X)
    def explain(self, X): return np.asarray(self.ex.shap_values(X))

def controls(base_attr, rng, feats=None):
    s = base_attr.std()
    out = {"control: random": rng.normal(0, s, base_attr.shape)}
    for k in (0.5, 1.0, 2.0):
        out[f"control: noise x{k}"] = base_attr + rng.normal(0, k * s, base_attr.shape)
    perm = rng.permutation(base_attr.shape[1])
    out["control: permuted labels"] = base_attr[:, perm]
    # magnitude controls: same ranking or same pattern, wrong size
    out["control: shrunk x0.5"] = 0.5 * base_attr
    out["control: inflated x1.5"] = 1.5 * base_attr
    if feats is not None:
        cal = [k for k, f in enumerate(feats) if f in sum(GROUPS.values(), [])]
        hist = [k for k, f in enumerate(feats) if f.startswith("load_")]
        if cal and hist:
            a = base_attr.copy(); moved = 0.5 * a[:, cal]
            a[:, cal] -= moved; a[:, hist[0]] += moved.sum(1)
            out["control: half of drivers moved to history"] = a
    return out

def split_idx(n):
    a, b = int(SPLIT[0] * n), int((SPLIT[0] + SPLIT[1]) * n)
    return slice(0, a), slice(a, b), slice(b, n)
def gt_scores(A, feats, truth_s, interaction=False):
    A = pd.DataFrame(A, columns=feats); out = {}
    tr = {g: truth_s[c].values.copy() for g, c in TRUTH_COL.items()}
    conv = {"": tr}
    if interaction:
        ci = truth_s["C_interaction"].values
        conv = {"": {**tr, "temperature": tr["temperature"] + ci},
                "_split": {**tr, "temperature": tr["temperature"] + ci / 2, "hour": tr["hour"] + ci / 2}}
    for suffix, tt in conv.items():
        bs = []
        for g, cols in GROUPS.items():
            a = A[[c for c in cols if c in A]].sum(1).values; c = tt[g] - tt[g].mean()
            if np.std(c) < 1e-9 or np.std(a) < 1e-12:
                out[f"r_{g}{suffix}"], out[f"b_{g}{suffix}"] = np.nan, 0.0 if np.std(c) > 1e-9 else np.nan
            else:
                out[f"r_{g}{suffix}"] = np.corrcoef(a, c)[0, 1]; out[f"b_{g}{suffix}"] = np.cov(a, c)[0, 1] / np.var(c, ddof=1)
            bs.append(out[f"b_{g}{suffix}"])
        out[f"magnitude_correctness{suffix}"] = 1 - np.nanmean(np.abs(1 - np.array(bs, float)))
        out[f"pattern_correctness{suffix}"] = np.nanmean([out[f"r_{g}{suffix}"] for g in GROUPS])
    tot = A.abs().sum().sum()
    for rl in ("direct cause", "demand history", "consequence", "irrelevant"):
        out[f"share_{rl}"] = A[[f for f in feats if role(f) == rl]].abs().sum().sum() / tot
    imp = A.abs().mean().groupby(lambda f: concept(f)).sum().sort_values(ascending=False)
    named = set(imp.index[:5]); tp = len(named & TRUE_PARENTS)
    out["parents_precision"], out["parents_recall"] = tp / 5, tp / len(TRUE_PARENTS)
    out["parents_F1"] = 0 if tp == 0 else 2 * tp / (5 + len(TRUE_PARENTS))
    out["false_parents_consequence"] = len(named & {"gas", "coal"})
    return out

def deletion_insertion(model, Xs, y, A, mu, K=10):
    """Most-relevant-first perturbation on the model output (per-forecast rankings by |A|).
    Deletion: replace the top-k features by their training mean and record |f(x) - f(x_-k)|;
    a faithful ranking moves the prediction the most for small k.
    Insertion: start from all features at the mean, restore the top-k, record |f(x_+k) - f(x)|;
    a faithful ranking recovers the prediction fastest. Both use k = 1..K and are scaled by the
    standard deviation of the predictions, so higher = more faithful for both."""
    n, F = Xs.shape; order = np.argsort(-np.abs(A), axis=1); rows = np.arange(n)[:, None]
    f = lambda M: model.predict(pd.DataFrame(M, columns=FEATS_CUR))
    f0 = f(Xs); scale = max(np.std(f0), 1e-9); dele, ins = [], []
    for m in range(1, min(K, F) + 1):
        M = Xs.copy(); M[rows, order[:, :m]] = mu[order[:, :m]]; dele.append(np.mean(np.abs(f0 - f(M))))
        M = np.tile(mu, (n, 1)); M[rows, order[:, :m]] = Xs[rows, order[:, :m]]; ins.append(np.mean(np.abs(f(M) - f0)))
    return float(np.mean(dele) / scale), float(1 - np.mean(ins) / scale)

def regime_stability(A, err):
    hi = np.abs(err) >= np.quantile(np.abs(err), 0.9)
    v1, v0 = A[hi].var(0), A[~hi].var(0); ok = v0 > 1e-12
    return float(np.median(v1[ok] / v0[ok]))

def eval_metrics(model, Xte, yte, A, mu, ridge_top10, feats):
    out = {}
    top10 = set(np.array(feats)[np.argsort(-np.abs(A).mean(0))[:10]])
    out["metric_recall@10"] = len(top10 & ridge_top10) / 10
    out["metric_deletion"], out["metric_insertion"] = deletion_insertion(model, Xte.values, yte.values, A, mu)
    out["metric_regime_stability"] = regime_stability(A, model.predict(Xte) - yte.values)
    return out
'''

# %%
# The Spanish generator (BENCH_CODE, unchanged above) with four small hooks so that its climate and its generation
# sizes can be set from the parameters. With the Spanish parameters the generator is exactly the original.
PATCHES = [
    ("temps = {c: 15 + off - 9 * np.cos(2 * np.pi * (doy - 20) / 365.25) + (5 + 0.3 * off) * np.sin(2 * np.pi * (h - 9) / 24)",
     "temps = {c: g.get('t_mean', 15) + off - g.get('t_season', 9) * np.cos(2 * np.pi * (doy - 20) / 365.25) + (g.get('t_daily', 5) + 0.3 * off) * np.sin(2 * np.pi * (h - 9) / 24)"),
    ("solar = 7000 *", "solar = g.get('gen_scale', 1.0) * 7000 *"),
    ("wind = np.minimum(15000, 60 * wind_speed ** 3)", "wind = g.get('gen_scale', 1.0) * np.minimum(15000, 60 * wind_speed ** 3)"),
    ("nuclear = np.full(n, 7000.0)", "nuclear = np.full(n, 7000.0 * g.get('nuclear_scale', 1.0))"),
    ("nuclear[s:s + 24 * 30] -= 1000", "nuclear[s:s + 24 * 30] -= 1000 * g.get('nuclear_scale', 1.0)"),
    ("hydro = 3000 + 1500 * np.sin(2 * np.pi * (doy - 60) / 365.25) + rng.normal(0, 150, n)",
     "hydro = g.get('gen_scale', 1.0) * (3000 + 1500 * np.sin(2 * np.pi * (doy - 60) / 365.25) + rng.normal(0, 150, n))"),
    ('"forecast solar day ahead": solar + rng.normal(0, 200, n), "forecast wind onshore day ahead": wind + rng.normal(0, 800, n)',
     '"forecast solar day ahead": solar + g.get("gen_scale", 1.0) * rng.normal(0, 200, n), '
     '"forecast wind onshore day ahead": wind + g.get("gen_scale", 1.0) * rng.normal(0, 800, n)'),
]
code = BENCH_CODE
for old, new in PATCHES:
    assert code.count(old) == 1, f"generator text not found: {old[:60]}"
    code = code.replace(old, new)
B = {}; exec(code, B)
B["TZ"] = TZ
B["spanish_holidays"] = country_holidays                      # the generator and its features use the Portuguese calendar
B["CITIES"] = {"Lisbon": 0.5, "Porto": -0.5, "Braga": -1.5, "Coimbra": 0.0, "Faro": 1.5}   # offsets average to 0
B["FAST_MODE"] = RUN_PLAN == "smoke"
if RUN_PLAN == "smoke": B["START"] = "2017-06-01 00:00"

# ---- calibration on the Portuguese training period (real demand and real temperature) ----
loc = IDX.tz_convert(TZ); h_ = loc.hour.values; dow_ = loc.dayofweek.values; doy_ = loc.dayofyear.values
T_ = W["temp"].values; y_ = LOAD.values
hol_ = np.array([d.date() in HOLS for d in loc.tz_localize(None).normalize()], float)
G0 = B["GEN"]
Dm = np.column_stack([np.ones(len(h_)), np.sin(2 * np.pi * h_ / 24), np.cos(2 * np.pi * h_ / 24),
                      np.sin(4 * np.pi * h_ / 24), np.cos(4 * np.pi * h_ / 24), dow_ == 5, dow_ == 6, hol_,
                      np.clip(G0["heat_thr"] - T_, 0, None), np.clip(T_ - G0["cool_thr"], 0, None) ** G0["cool_exp"]]).astype(float)
ok = (PART == 0) & ~np.isnan(y_) & ~np.isnan(T_)
cf = np.linalg.lstsq(Dm[ok], y_[ok], rcond=None)[0]
def amp_phase(a, b, w): return float(np.hypot(a, b)), float((np.arctan2(-b, a) / w) % (2 * np.pi / w))
a1, p1 = amp_phase(cf[1], cf[2], 2 * np.pi / 24); a2, p2 = amp_phase(cf[3], cf[4], 4 * np.pi / 24)
Dt = np.column_stack([np.ones(len(h_)), np.cos(2 * np.pi * (doy_ - 20) / 365.25), np.sin(2 * np.pi * (h_ - 9) / 24)])
okt = ~np.isnan(T_); ct = np.linalg.lstsq(Dt[okt], T_[okt], rcond=None)[0]
GEN_PT = dict(G0)
GEN_PT.update(base=float(cf[0]), hour_amp=(a1, a2), hour_phase=(p1, p2), sat=float(cf[5]), sun=float(cf[6]), holiday=float(cf[7]),
              heat_slope=max(float(cf[8]), 0.0), cool_coef=max(float(cf[9]), 0.0),
              t_mean=float(ct[0]), t_season=float(-ct[1]), t_daily=float(ct[2]),
              ar_sd=G0["ar_sd"] * RATIO, noise_sd=G0["noise_sd"] * RATIO, operator_sd=G0["operator_sd"] * RATIO,
              gas_latent=G0["gas_latent"] * RATIO, gas_noise_sd=G0["gas_noise_sd"] * RATIO,
              gen_scale=RATIO, nuclear_scale=0.0)                 # Portugal has no nuclear power
for k in ("heat_slope", "cool_coef"):
    if GEN_PT[k] == 0: print(f"note: fitted {k} was negative and is set to 0")
LAT_PT = {c: v * RATIO for c, v in SYN_CONDITIONS.items()}
cal = pd.DataFrame({"Spain (paper)": {k: G0.get(k, d) for k, d in [("base", 0), ("hour_amp", 0), ("hour_phase", 0), ("sat", 0), ("sun", 0),
                                         ("holiday", 0), ("heat_slope", 0), ("cool_coef", 0), ("t_mean", 15), ("t_season", 9), ("t_daily", 5),
                                         ("ar_sd", 0), ("noise_sd", 0)]}})
cal["Portugal (calibrated)"] = [GEN_PT[k] for k in cal.index]
cal.loc["hidden driver, default (MW per unit)"] = [1200, LAT_PT.get("default", 1200 * RATIO)]
show(cal)
print(f"ratio of mean demand Portugal/Spain generator: {RATIO:.3f}")
(RUN_DIR / "generator_portugal.json").write_text(json.dumps(dict(GEN_PT=GEN_PT, latent=LAT_PT, ratio=RATIO), indent=2, default=float))

# %%
def run_synth(cname, seed, kind):
    """One synthetic realization, one model: identical to the Spanish benchmark runs (fixed 36-hour lag, same
    features, split, explained hours and scoring), with the Portuguese generator."""
    cond = dict(name=cname, change="2016-07-01", latent=LAT_PT[cname], interaction=0.0)
    energy, weather, truth, hols = B["generate"](cond, seed, g=GEN_PT)
    X, y = B["build_features"](energy, weather, hols); feats = list(X.columns); B["FEATS_CUR"] = feats
    tr, va, te = B["split_idx"](len(X)); t0 = time.time()
    rng = np.random.default_rng(seed + 99)
    sample = np.sort(rng.choice(np.arange(te.start, te.stop), min(N_EXPLAIN, te.stop - te.start), replace=False))
    truth_s = truth.reindex(X.index[sample]); A, agree = {}, None
    if kind == "XGBoost":
        m = B["XGBTreeSHAP"]().fit(X.iloc[tr], y.iloc[tr], X.iloc[va], y.iloc[va])
        A["XGBoost, TreeSHAP"] = m.explain(X.iloc[sample])
        mae = float(np.mean(np.abs(m.predict(X.iloc[te]) - y.values[te])))
    else:
        mu, sd = X.iloc[tr].mean().values, X.iloc[tr].std().values + 1e-9
        ym, ysd = y.iloc[tr].mean(), y.iloc[tr].std()
        Xs = ((X.values - mu) / sd).astype(np.float32); ys = ((y.values - ym) / ysd).astype(np.float32)
        ARCH[kind] = dict(BENCH_ARCH[kind]); TRAIN["lr"] = 1e-3
        i_tr = np.arange(max(tr.start, WINDOW - 1), tr.stop); i_va = np.arange(va.start, va.stop); i_te = np.arange(te.start, te.stop)
        Wtr, Wva, Wte, Wex = windows(Xs, i_tr), windows(Xs, i_va), windows(Xs, i_te), windows(Xs, sample)
        bg = Wtr[np.random.default_rng(seed).choice(len(Wtr), SHAP_BACKGROUND, replace=False)]
        m = SeqModel(kind, len(feats), seed).fit(Wtr, ys[i_tr], Wva, ys[i_va])
        A[f"{kind}, SHAP"] = m.shap(Wex, bg, seed).sum(1) * ysd
        mae = float(np.mean(np.abs(m.predict(Wte) * ysd + ym - y.values[i_te])))
        if kind == "TFT":
            A["TFT, variable selection (unsigned)"] = m.intrinsic(Wex)[0]
            agree = pair_agreement(A["TFT, variable selection (unsigned)"], A["TFT, SHAP"])
    ymean = float(y.values[te].mean())
    for e, a in A.items():
        np.savez_compressed(RUN_DIR / "attributions" / f"synthetic_{cname}_{kind}_seed{seed}_{e.split(', ')[1].split(' ')[0]}.npz",
                            A=a.astype(np.float32), feats=np.array(feats), sample=sample)
        sc = B["gt_scores"](a, feats, truth_s)
        row = dict(stage="synthetic", condition=cname, latent_MW=LAT_PT[cname], model=kind, explainer=e, seed=seed,
                   signed=not e.endswith("(unsigned)"), MAE=mae, MAE_pct=100 * mae / ymean, seconds=round(time.time() - t0))
        row.update({k: float(v) for k, v in sc.items() if np.isscalar(v)})
        if agree is not None and e.endswith("(unsigned)"):
            row.update(agree_spearman=agree[0], agree_wtau=agree[1], agree_J5=agree[2], agree_S5=agree[3])
        log(row)
    main = A.get("XGBoost, TreeSHAP", A.get(f"{kind}, SHAP"))
    print(f"synthetic {cname:12s} seed {seed} {kind:11s}: correctness "
          f"{B['gt_scores'](main, feats, truth_s)['magnitude_correctness']:.2f}, MAE {mae:.0f} MW ({100 * mae / ymean:.1f}%), {time.time() - t0:.0f}s")

if RUN_SYNTHETIC:
    t_all = time.time()
    for cname in SYN_CONDITIONS:
        for seed in SEEDS:
            for kind in MODELS:
                if kind == "XGBoost" and not (HAVE_XGB and HAVE_SHAP): continue
                if is_done(stage="synthetic", condition=cname, model=kind, seed=seed): continue
                run_synth(cname, seed, kind)
    print(f"synthetic stage finished in {time.time() - t_all:.0f}s")

# %% [markdown]
# ## 7. Real data and injected truth: train, forecast and explain (resumable)

# %%
def run_one(dataset, kind, seed, ablation):
    P = DATA[dataset]; F = len(P["feats"]); t0 = time.time()
    XS, YS, ysd, ym = P["XS"], P["YS"], P["ysd"], P["ym"]
    I_TR, I_VA, I_TE, S = P["I_TR"], P["I_VA"], P["I_TE"], P["SAMPLE"]
    A, abl = {}, {}
    if kind == "XGBoost":
        m = XGBRef(seed, BEST["XGBoost"]).fit(XS[I_TR], YS[I_TR], XS[I_VA], YS[I_VA]); epochs = int(m.m.best_iteration)
        pred = m.predict(XS[I_TE]) * ysd + ym
        A["XGBoost, TreeSHAP"] = m.shap(XS[S]) * ysd
        if ablation:
            bg = XS[I_TR][np.random.default_rng(seed).choice(len(I_TR), 2000, replace=False)]
            abl = group_ablation(m.predict, XS[S], bg, P["group"], seq=False)
    else:
        ARCH[kind] = {k: v for k, v in BEST[kind].items() if k != "lr"}
        TRAIN["lr"] = BEST[kind].get("lr", 1e-3)
        Wtr, Wva, Wte, Wex = windows(XS, I_TR), windows(XS, I_VA), windows(XS, I_TE), windows(XS, S)
        bg = Wtr[np.random.default_rng(seed).choice(len(Wtr), SHAP_BACKGROUND, replace=False)]
        m = SeqModel(kind, F, seed).fit(Wtr, YS[I_TR], Wva, YS[I_VA]); epochs = m.epochs
        pred = m.predict(Wte) * ysd + ym
        A[f"{kind}, SHAP"] = m.shap(Wex, bg, seed).sum(1) * ysd
        if RUN_IG: A[f"{kind}, IG"] = m.ig(Wex).sum(1) * ysd
        if kind == "TFT": A["TFT, variable selection (unsigned)"] = m.intrinsic(Wex)[0]
        if ablation: abl = group_ablation(m.predict, Wex, bg, P["group"], seq=True)
    tag = f"{dataset}_{kind}_seed{seed}"
    pd.DataFrame({"row": I_TE, "forecast": pred}).to_csv(RUN_DIR / "forecasts" / f"{tag}.csv", index=False)
    np.savez_compressed(RUN_DIR / "attributions" / f"{tag}.npz", sample=S, feats=np.array(P["feats"]),
                        **{f"A::{k}": v.astype(np.float32) for k, v in A.items()},
                        **{f"ABL::{k}": (v * ysd).astype(np.float32) for k, v in abl.items()})
    mae = float(np.nanmean(np.abs(pred - P["Y"].values[I_TE])))
    log(dict(stage="main", dataset=dataset, model=kind, seed=seed, MAE=mae, epochs=epochs, seconds=round(time.time() - t0)))
    print(f"{dataset:9s} seed {seed} {kind:11s}: MAE {mae:6.0f} MW, {time.time() - t0:.0f}s")

t_all = time.time()
for dataset, flag in [("real", RUN_REAL), ("injected", RUN_INJECTED)]:
    if not flag: continue
    for seed in SEEDS:
        for kind in MODELS:
            if kind == "XGBoost" and not (HAVE_XGB and HAVE_SHAP): continue
            if is_done(stage="main", dataset=dataset, model=kind, seed=seed): continue
            run_one(dataset, kind, seed, ablation=(dataset == "real"))
print(f"real and injected stages finished in {time.time() - t_all:.0f}s")

# %% [markdown]
# ## 8. Results
#
# S: synthetic benchmark. A: accuracy. B: size check. C: injected truth. D: natural experiments. E: signatures. X: Spain vs Portugal, checks of every finding and the conclusion.

# %%
PACK = OUT_DIR / f"portugal_results_pack_{RUN_PLAN}"; PACK.mkdir(parents=True, exist_ok=True)
MDL = [f"# Portugal replication results ({pd.Timestamp.now():%Y-%m-%d %H:%M}), plan '{RUN_PLAN}'", ""]
def add(df, title, fname, note=None):
    df.to_csv(PACK / fname)
    try: txt = df.to_markdown(floatfmt=".3f")
    except Exception: txt = df.to_string()
    MDL.extend([f"## {title}", f"(file: {fname})", ""] + ([note, ""] if note else []) + [txt, ""]); show(df)
LOGDF = pd.DataFrame(logged())
def load_run(dataset, kind, seed):
    p = RUN_DIR / "attributions" / f"{dataset}_{kind}_seed{seed}.npz"
    if not p.exists(): return None
    z = np.load(p, allow_pickle=True)
    return dict(sample=z["sample"], feats=list(z["feats"]),
                A={k[3:]: z[k] for k in z.files if k.startswith("A::")}, ABL={k[5:]: z[k] for k in z.files if k.startswith("ABL::")})
RUNS = {(d, k, s): r for d in DATA for k in MODELS for s in SEEDS if (r := load_run(d, k, s)) is not None}
print(len(RUNS), "real or injected runs with explanations")

# ---------------- S. Synthetic benchmark calibrated to Portugal ----------------
SY = LOGDF[LOGDF["stage"] == "synthetic"].copy() if len(LOGDF) and "stage" in LOGDF else pd.DataFrame()
DRV = ["hour", "week", "holiday", "temperature"]
if len(SY):
    SY["signed"] = SY["signed"].astype(bool)
    S_sig = SY[SY.signed]
    cols = ["magnitude_correctness", "pattern_correctness"] + [f"b_{g}" for g in DRV] + [f"r_{g}" for g in DRV] + ["MAE", "MAE_pct"]
    add(S_sig[S_sig.condition == "default"].groupby("explainer")[cols].agg(["mean", "std"]).round(3),
        "S1. Correctness on the synthetic benchmark calibrated to Portugal (default version)", "S1_synthetic_correctness.csv",
        "b = 1: the full true effect is credited. r: pattern correlation with the true effect. Same scoring as Table 5 of the paper.")
    shc = ["share_direct cause", "share_demand history", "share_consequence", "share_irrelevant"]
    tabS2 = SY[SY.condition == "default"].groupby("explainer")[shc + [c for c in ["agree_spearman", "agree_wtau"] if c in SY]].mean().round(3)
    add(tabS2, "S2. Where the credit goes (synthetic, default) and agreement of TFT selection weights with SHAP", "S2_synthetic_shares.csv",
        "Spain: SHAP on the TFT 0.60 to true causes, XGBoost 0.50 to demand history, TFT selection weights 0.32 to irrelevant inputs and Spearman 0.25 with SHAP.")
    if S_sig.condition.nunique() > 1:
        tabS3 = S_sig.pivot_table(index="latent_MW", columns="explainer", values="magnitude_correctness", aggfunc="mean").round(3)
        add(tabS3, "S3. Hidden-driver sweep: magnitude correctness by size of the hidden driver (MW per unit, Portugal)", "S3_hidden_driver.csv",
            "Spain: TFT 0.85 without the hidden driver and 0.75 at the largest size, XGBoost 0.53 and 0.39.")
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
        for e in tabS3.columns:
            sd_ = S_sig[S_sig.explainer == e].groupby("latent_MW")["magnitude_correctness"].std().reindex(tabS3.index)
            ax[0].errorbar(tabS3.index / RATIO, tabS3[e], yerr=sd_, marker="o", capsize=3, label=e)
            hs = S_sig[S_sig.explainer == e].groupby("latent_MW")["share_demand history"].mean()
            ax[1].plot(hs.index / RATIO, hs.values, marker="o", label=e)
        for e, (v0, v1) in {"TFT, SHAP": (0.85, 0.75), "XGBoost, TreeSHAP": (0.53, 0.39)}.items():
            ax[0].plot([0, 2400], [v0, v1], ls=":", color="grey"); ax[0].annotate(f"Spain, {e.split(',')[0]}", (2400, v1), fontsize=7)
        ax[0].set(xlabel="hidden driver (Spanish-equivalent MW per unit)", ylabel="magnitude correctness", title="(a) Correctness")
        ax[1].set(xlabel="hidden driver (Spanish-equivalent MW per unit)", ylabel="share to demand history", title="(b) Credit to demand history")
        ax[0].legend(fontsize=8); fig.tight_layout(); fig.savefig(PACK / "fig_S3_hidden_driver.png"); plt.close(fig)
    pv = S_sig[S_sig.condition == "default"].pivot_table(index="seed", columns="explainer", values="magnitude_correctness")
    if {"TFT, SHAP", "XGBoost, TreeSHAP"} <= set(pv.columns) and len(pv) >= 2:
        d_ = pv["TFT, SHAP"] - pv["XGBoost, TreeSHAP"]
        p_ = stats.wilcoxon(d_).pvalue if len(d_) >= 5 else np.nan
        add(pd.DataFrame([dict(seeds=len(d_), mean_difference=d_.mean(), TFT_better_share=(d_ > 0).mean(), wilcoxon_p=p_)],
                         index=["TFT, SHAP vs XGBoost, TreeSHAP"]).round(4), "S4. Paired comparison by seed (synthetic, default)", "S4_synthetic_pairs.csv")

# %%
# ---------------- A. Accuracy on real data ----------------
P = DATA["real"]; y_te = P["Y"].values[P["I_TE"]]
def metr(f):
    e = f - y_te; ok = ~np.isnan(e); return dict(MAE=np.mean(np.abs(e[ok])), RMSE=np.sqrt(np.mean(e[ok] ** 2)), MAPE=100 * np.mean(np.abs(e[ok]) / y_te[ok]))
FC = {}
for k in MODELS:
    for s in SEEDS:
        p = RUN_DIR / "forecasts" / f"real_{k}_seed{s}.csv"
        if p.exists(): FC.setdefault(k, {})[s] = pd.read_csv(p)["forecast"].values
ENS = {k: np.mean(list(v.values()), axis=0) for k, v in FC.items()}
REF = {"TSO day-ahead forecast": E["total load forecast"].values[P["I_TE"]] if "total load forecast" in E else None,
       "Naive, one week earlier": LOAD.shift(168).values[P["I_TE"]], "Naive, two days earlier": LOAD.shift(48).values[P["I_TE"]]}
REF = {k: v for k, v in REF.items() if v is not None}
rows = []
for k, v in FC.items():
    ms = pd.DataFrame([metr(f) for f in v.values()])
    rows.append(dict(model=k, seeds=len(v), MAE=ms.MAE.mean(), MAE_sd=ms.MAE.std(), RMSE=ms.RMSE.mean(), MAPE=ms.MAPE.mean(), ensemble_MAE=metr(ENS[k])["MAE"]))
for k, v in REF.items():
    m_ = metr(v); rows.append(dict(model=k, MAE=m_["MAE"], RMSE=m_["RMSE"], MAPE=m_["MAPE"]))
ACC = pd.DataFrame(rows).set_index("model")
if "TSO day-ahead forecast" in REF: ACC["ratio_to_TSO"] = ACC["MAE"] / metr(REF["TSO day-ahead forecast"])["MAE"]
ACC["reduction_vs_weekly_naive_pct"] = 100 * (1 - ACC["MAE"] / metr(REF["Naive, one week earlier"])["MAE"])
add(ACC.round(2), "A1. Accuracy on the real Portuguese test period (MW). Spain: MAPE 5.75% TFT, 5.00% XGBoost; 6.45 and 5.62 times the TSO error", "A1_accuracy.csv")
add(pd.DataFrame({k: BEST[k] for k in MODELS if k in BEST}).T, "A2. Settings (selected on Spanish data, not retuned)", "A2_best_params.csv")
def dm(f1, f2, h=24):
    d = np.abs(f1 - y_te) - np.abs(f2 - y_te); d = d[~np.isnan(d)]; n_ = len(d); db = d.mean()
    g = [np.sum((d[k:] - db) * (d[:n_ - k] - db)) / n_ for k in range(h)]; var = (g[0] + 2 * sum(g[1:])) / n_
    var = var if var > 0 else g[0] / n_; s = db / np.sqrt(var) * np.sqrt((n_ + 1 - 2 * h + h * (h - 1) / n_) / n_)
    return s, 2 * stats.t.sf(abs(s), n_ - 1)
ALLF = {**{f"{k} (ensemble)": v for k, v in ENS.items()}, **REF}
rows = [dict(model_1=a, model_2=b, DM_stat=(r := dm(ALLF[a], ALLF[b]))[0], p_value=r[1],
             better=(a if r[0] < 0 else b) if r[1] < 0.05 else "no significant difference") for a, b in itertools.combinations(ALLF, 2)]
add(pd.DataFrame(rows).set_index(["model_1", "model_2"]).round(4), "A3. Diebold–Mariano tests (seed ensembles)", "A3_dm_tests.csv")
te_t = IDX[P["I_TE"]].tz_convert(TZ)
err = pd.DataFrame({k: np.abs(v - y_te) for k, v in {**ENS, **REF}.items()}, index=te_t)
add(err.groupby(err.index.month).mean().round(0), "A4. MAE by month (seed ensembles)", "A4_mae_by_month.csv")
add(err.groupby(err.index.hour).mean().round(0), "A5. MAE by hour (seed ensembles)", "A5_mae_by_hour.csv")

# %%
# A6. Quality of the TSO day-ahead forecast by year (all hours). In the Portuguese data its error rises sharply in 2018,
# the test year, so the weekly naive forecast is the accuracy reference used to compare with Spain.
if "total load forecast" in E:
    _f, _y = E["total load forecast"], LOAD
    _yr = _y.index.tz_convert(TZ).year
    tso = pd.DataFrame({"TSO forecast MAE (MW)": (_f - _y).abs().groupby(_yr).mean(),
                        "Weekly naive MAE (MW)": (_y.shift(168) - _y).abs().groupby(_yr).mean()}).round(0)
    tso["best time shift of TSO forecast (h)"] = int(np.argmin([np.nanmean(np.abs(_f.shift(s_) - _y)) for s_ in range(-3, 4)]) - 3)
    add(tso, "A6. TSO day-ahead forecast quality by year", "A6_tso_forecast_by_year.csv",
        "A best shift of 0 means the forecast is correctly aligned. A jump in one year points to a data-quality change, not to the models.")

# %%
# ---------------- B. Model-level size check (real data) ----------------
rows = []
for (d, k, s), R in RUNS.items():
    if d != "real" or not R["ABL"]: continue
    grp = np.array([group_of(f) for f in R["feats"]])
    for e, A in R["A"].items():
        if "unsigned" in e: continue
        for g, delta in R["ABL"].items():
            a = A[:, grp == g].sum(1)
            if np.std(delta) < 1e-6 or np.std(a) < 1e-9: continue
            rows.append(dict(explainer=e, seed=s, group=g, r=np.corrcoef(a, delta)[0, 1], b=np.cov(a, delta)[0, 1] / np.var(delta, ddof=1),
                             effect_sd_MW=np.std(delta)))
BB = pd.DataFrame(rows)
if len(BB):
    tab = BB.groupby(["explainer", "group"])[["r", "b", "effect_sd_MW"]].mean().round(3)
    add(tab, "B1. Model-level size check: group attribution vs the model's own effect of that group", "B1_size_check_by_group.csv",
        "b = 1: the explanation reports the full size of the model's own effect; r: pattern. Effects are interventional (group replaced by background inputs).")
    main = BB[BB.group.isin(["Demand history", "Hour of day", "Weekday", "Temperature", "Holiday and special days"])]
    summ = main.assign(size_err=lambda d_: (1 - d_.b).abs()).groupby(["explainer", "seed"]).agg(r=("r", "mean"), size_faithfulness=("size_err", lambda v: 1 - v.mean()))
    add(summ.groupby("explainer").agg(["mean", "std"]).round(3), "B2. Size faithfulness to the model (1 − mean |1 − b| over the five main groups)", "B2_size_faithfulness.csv")

# %%
# ---------------- C. Injected ground truth (real Portuguese data + known effects) ----------------
rows = []
for (d, k, s), R in RUNS.items():
    if d != "injected": continue
    S = R["sample"]; tr = TRUTH_INJ.iloc[S]; feats = R["feats"]
    for e, A in R["A"].items():
        row = dict(explainer=e, model=k, seed=s)
        for comp, feat, truth_col in [("event", "event_flag", "C_event"), ("driver", "driver_z", "C_z")]:
            a = A[:, feats.index(feat)]; c = tr[truth_col].values - tr[truth_col].values.mean()
            row[f"r_{comp}"] = np.corrcoef(a, c)[0, 1] if np.std(a) > 0 else np.nan
            row[f"b_{comp}"] = np.cov(a, c)[0, 1] / np.var(c, ddof=1) if "unsigned" not in e else np.nan
        dz, dd = np.abs(A[:, feats.index("driver_z")]).mean(), np.abs(A[:, feats.index("decoy_d")]).mean()
        row["decoy_credit_ratio"] = dd / max(dz, 1e-9)
        row["magnitude_correctness"] = 1 - np.mean([abs(1 - row["b_event"]), abs(1 - row["b_driver"])]) if "unsigned" not in e else np.nan
        rows.append(row)
INJ = pd.DataFrame(rows)
if len(INJ):
    tabC = INJ.groupby("explainer")[["r_event", "b_event", "r_driver", "b_driver", "decoy_credit_ratio", "magnitude_correctness"]].agg(["mean", "std"]).round(3)
    add(tabC, "C1. Injected ground truth in the real Portuguese data: recovery of a known event and a known nonlinear driver, and credit to a decoy",
        "C1_injected_truth.csv", f"Effects scaled to Portugal: event {INJECT_PT['event_mw']} MW, driver up to about {INJECT_PT['driver_mw']} MW. "
        "b = 1: the full injected effect is credited. decoy_credit_ratio: attribution to the decoy relative to the driver (0 is correct).")
    syn_mc = S_sig[S_sig.condition == "default"].groupby("explainer")["magnitude_correctness"].mean() if len(SY) else pd.Series(dtype=float)
    cmp = pd.DataFrame({"Portugal, injected real data": INJ.groupby("explainer")["magnitude_correctness"].mean(),
                        "Portugal, synthetic benchmark": syn_mc,
                        "Spain, injected real data": pd.Series({"TFT, SHAP": 0.78, "XGBoost, TreeSHAP": 0.42, "LSTM, SHAP": 0.75, "Transformer, SHAP": 0.72}),
                        "Spain, synthetic benchmark": pd.Series({"TFT, SHAP": 0.80, "XGBoost, TreeSHAP": 0.44, "LSTM, SHAP": 0.72, "Transformer, SHAP": 0.74})})
    cmp = cmp.loc[[e for e in cmp.index if e in set(INJ.explainer) and "unsigned" not in e]]
    cmp["Portugal, injected minus synthetic"] = cmp.iloc[:, 0] - cmp.iloc[:, 1]
    add(cmp.round(3), "C2. Does the synthetic result hold on real Portuguese data with injected truth?", "C2_synthetic_vs_injected.csv",
        "In Spain every injected value was within 0.03 of its benchmark value.")
    accI = LOGDF[(LOGDF.stage == "main") & (LOGDF.dataset == "injected")].groupby("model")["MAE"].agg(["mean", "std"]).round(0)
    add(accI, "C3. Accuracy on the injected data (MW)", "C3_accuracy_injected.csv")
    pv = INJ.pivot_table(index="seed", columns="explainer", values="magnitude_correctness")
    if {"TFT, SHAP", "XGBoost, TreeSHAP"} <= set(pv.columns) and len(pv) >= 2:
        d_ = pv["TFT, SHAP"] - pv["XGBoost, TreeSHAP"]
        add(pd.DataFrame([dict(seeds=len(d_), mean_difference=d_.mean(), TFT_better_share=(d_ > 0).mean())], index=["TFT, SHAP vs XGBoost, TreeSHAP"]).round(3),
            "C4. Paired comparison by seed (injected data)", "C4_injected_pairs.csv")

# %%
# ---------------- D. Natural experiments on real drivers ----------------
loc = IDX.tz_convert(TZ); Yr = LOAD.values; Xr = DATA["real"]["X"]
pos_of = pd.Series(np.arange(len(IDX)), index=IDX)
def shift_rows(rows_, hours): return rows_ + hours
# D1: holiday effect from holidays on weekdays: holiday hour minus the mean of the same weekday and hour one week before and after
isspecial = (Xr[["holiday", "bridge", "pre_holiday", "post_holiday", "christmas"]].sum(axis=1).values > 0)
hol_rows = np.flatnonzero((Xr["holiday"].values == 1) & (loc.dayofweek < 5) & (Xr["christmas"].values == 0))
nat = np.full(len(IDX), np.nan)
for r_ in hol_rows:
    a_, b_ = r_ - 168, r_ + 168
    if a_ < 0 or b_ >= len(IDX) or isspecial[a_] or isspecial[b_]: continue
    nat[r_] = Yr[r_] - 0.5 * (Yr[a_] + Yr[b_])
nat_tr = np.nanmean(nat[PART == 0]); nat_te_mask = ~np.isnan(nat) & (PART == 2)
rows = []
for (d, k, s), R in RUNS.items():
    if d != "real": continue
    S = R["sample"]; m_ = nat_te_mask[S]
    if m_.sum() < 10: continue
    for e, A in R["A"].items():
        if "unsigned" in e: continue
        a = A[m_, R["feats"].index("holiday")]; c = nat[S][m_]
        rows.append(dict(explainer=e, seed=s, hours=int(m_.sum()), natural_effect_MW=c.mean(), attribution_MW=a.mean(),
                         ratio=a.mean() / c.mean(), r=np.corrcoef(a, c)[0, 1] if np.std(a) > 0 else np.nan))
NH = pd.DataFrame(rows)
if len(NH):
    add(NH.groupby("explainer")[["hours", "natural_effect_MW", "attribution_MW", "ratio", "r"]].mean().round(3),
        "D1. Holiday effect: natural estimate (holiday vs same weekday one week before and after) vs attribution to the holiday indicator, test holidays",
        "D1_holiday_natural.csv", f"Natural holiday effect over the training period: {nat_tr:.0f} MW (mean over weekday-holiday hours). ratio = 1: the attribution has the natural size.")
# D2: Sunday drop: mean demand on Sundays minus the mean on Tuesday-Thursday of the same week, same hour (no special days)
wk = pd.DataFrame({"y": Yr, "week": loc.isocalendar().week.values, "year": loc.isocalendar().year.values, "dow": loc.dayofweek,
                   "hour": loc.hour, "special": isspecial, "part": PART})
wk = wk[~wk.special]
gsun = wk[wk.dow == 6].groupby(["year", "week", "hour"]).y.mean(); gmid = wk[wk.dow.isin([1, 2, 3])].groupby(["year", "week", "hour"]).y.mean()
gap = (gsun - gmid).dropna()
test_keys = set(map(tuple, wk[wk.part == 2][["year", "week"]].drop_duplicates().values))
nat_sun = gap[[(y_, w_) in test_keys for y_, w_, h_ in gap.index]].mean()
rows = []
for (d, k, s), R in RUNS.items():
    if d != "real": continue
    S = R["sample"]; dw = loc.dayofweek.values[S]; grp = np.array([group_of(f) for f in R["feats"]])
    for e, A in R["A"].items():
        if "unsigned" in e: continue
        aw = A[:, grp == "Weekday"].sum(1)
        if (dw == 6).sum() < 10: continue
        mg = aw[dw == 6].mean() - aw[np.isin(dw, [1, 2, 3])].mean()
        rows.append(dict(explainer=e, seed=s, natural_sunday_gap_MW=nat_sun, attribution_gap_MW=mg, ratio=mg / nat_sun))
NS = pd.DataFrame(rows)
if len(NS):
    add(NS.groupby("explainer")[["natural_sunday_gap_MW", "attribution_gap_MW", "ratio"]].agg(["mean", "std"]).round(3),
        "D2. Sunday drop: natural estimate vs gap in weekday attribution (Sunday minus Tuesday–Thursday), test period", "D2_sunday_natural.csv",
        "The natural gap also contains other differences between days (e.g. temperature), so it is an approximate reference.")
# D3: temperature sensitivity (MW per degree) from a calendar-controlled regression vs slope implied by the temperature attribution
def cal_design(rows_):
    lc = loc[rows_]; D = pd.DataFrame(index=range(len(rows_)))
    for d_ in range(1, 7): D[f"dow{d_}"] = (lc.dayofweek == d_).astype(float)
    for hh in range(1, 24): D[f"h{hh}"] = (lc.hour == hh).astype(float)
    for mm in range(2, 13): D[f"m{mm}"] = (lc.month == mm).astype(float)
    for c in ["holiday", "pre_holiday", "post_holiday", "bridge", "christmas"]: D[c] = Xr[c].values[rows_]
    D["hdd"], D["cdd"] = Xr["hdd"].values[rows_], Xr["cdd"].values[rows_]; return D
trr = DATA["real"]["I_TR"]; reg = Ridge(alpha=1.0).fit(cal_design(trr), Yr[trr]); cf = pd.Series(reg.coef_, index=cal_design(trr[:3]).columns)
rows = []
for (d, k, s), R in RUNS.items():
    if d != "real": continue
    S = R["sample"]; grp = np.array([group_of(f) for f in R["feats"]]); cdd, hdd = Xr["cdd"].values[S], Xr["hdd"].values[S]
    for e, A in R["A"].items():
        if "unsigned" in e: continue
        at = A[:, grp == "Temperature"].sum(1); row = dict(explainer=e, seed=s)
        for nm, v, mask in [("cooling", cdd, cdd > 0), ("heating", hdd, hdd > 0)]:
            if mask.sum() > 30: row[f"{nm}_MW_per_degree"] = np.polyfit(v[mask], at[mask], 1)[0]
        rows.append(row)
NT = pd.DataFrame(rows)
if len(NT):
    tabT = NT.groupby("explainer").mean(numeric_only=True).drop(columns="seed").round(1)
    tabT.loc["calendar regression (training period)"] = [cf["cdd"], cf["hdd"]] if tabT.shape[1] == 2 else np.nan
    add(tabT, "D3. Temperature sensitivity: implied by the temperature attribution vs a calendar-controlled regression (MW per degree)",
        "D3_temperature_sensitivity.csv", "Heating sensitivity is reported per heating degree (demand rises as temperature falls).")

# %%
# ---------------- E. Signatures and robustness (real data) ----------------
gnames = ["Demand history"] + [g for g in GROUPS if not g.startswith("Injected")]
def gsum(A, grp): return np.stack([A[:, grp == g].sum(1) for g in gnames], 1)
rows = []
for (d, k, s), R in RUNS.items():
    if d != "real": continue
    grp = np.array([group_of(f) for f in R["feats"]])
    for e, A in R["A"].items():
        G = np.abs(gsum(A, grp)); rows.append(dict(explainer=e, seed=s, **dict(zip(gnames, G.sum(0) / G.sum()))))
SH = pd.DataFrame(rows)
if len(SH):
    add(SH.groupby("explainer")[gnames].mean().round(3), "E1. Share of absolute attribution by feature group (real Portuguese data)", "E1_group_shares.csv")
    ES_HIST = {"XGBoost, TreeSHAP": (0.44, 0.50), "TFT, SHAP": (np.nan, 0.19), "LSTM, SHAP": (np.nan, 0.26), "Transformer, SHAP": (np.nan, 0.24)}
    syn_h = SY[SY.condition == "default"].groupby("explainer")["share_demand history"].mean() if len(SY) else pd.Series(dtype=float)
    sig = pd.DataFrame({e: {"Portugal, real": SH[SH.explainer == e]["Demand history"].mean(), "Portugal, synthetic": syn_h.get(e, np.nan),
                            "Spain, real": ES_HIST.get(e, (np.nan, np.nan))[0], "Spain, synthetic": ES_HIST.get(e, (np.nan, np.nan))[1]}
                        for e in SH.explainer.unique() if "unsigned" not in e}).T.round(3)
    add(sig, "E2. Signature check: share of credit to demand history", "E2_signatures.csv",
        "Spain, real data: 0.44 for XGBoost and 0.18 to 0.23 for the neural models.")
    fig, ax = plt.subplots(figsize=(10, 3.4))
    SH.groupby("explainer")[gnames].mean().plot(kind="barh", stacked=True, ax=ax, colormap="tab20", width=0.7)
    ax.set_xlabel("share of absolute attribution"); ax.legend(fontsize=7, bbox_to_anchor=(1.01, 1), loc="upper left")
    fig.tight_layout(); fig.savefig(PACK / "fig_E1_group_shares.png"); plt.close(fig)
rows = []
real_runs = {key: R for key, R in RUNS.items() if key[0] == "real"}
for (d, k, s), R in real_runs.items():
    if "TFT, variable selection (unsigned)" in R["A"]:
        r_ = pair_agreement(R["A"]["TFT, variable selection (unsigned)"], R["A"]["TFT, SHAP"], k=5)
        rows.append(dict(comparison="TFT: selection weights vs SHAP (features)", spearman=r_[0], wtau=r_[1], J=r_[2], S=r_[3]))
for k in MODELS:
    ss = sorted(s for (d, kk, s) in real_runs if kk == k)
    e = "XGBoost, TreeSHAP" if k == "XGBoost" else f"{k}, SHAP"
    for s1, s2 in itertools.combinations(ss, 2):
        R1, R2 = real_runs[("real", k, s1)], real_runs[("real", k, s2)]; grp = np.array([group_of(f) for f in R1["feats"]])
        r_ = pair_agreement(gsum(R1["A"][e], grp), gsum(R2["A"][e], grp), k=3); rows.append(dict(comparison=f"seed ceiling: {e}", spearman=r_[0], wtau=r_[1], J=r_[2], S=r_[3]))
for k1, k2 in itertools.combinations(MODELS, 2):
    for s in SEEDS:
        if ("real", k1, s) in real_runs and ("real", k2, s) in real_runs:
            e1 = "XGBoost, TreeSHAP" if k1 == "XGBoost" else f"{k1}, SHAP"; e2 = "XGBoost, TreeSHAP" if k2 == "XGBoost" else f"{k2}, SHAP"
            R1, R2 = real_runs[("real", k1, s)], real_runs[("real", k2, s)]; grp = np.array([group_of(f) for f in R1["feats"]])
            r_ = pair_agreement(gsum(R1["A"][e1], grp), gsum(R2["A"][e2], grp), k=3); rows.append(dict(comparison=f"{e1} vs {e2}", spearman=r_[0], wtau=r_[1], J=r_[2], S=r_[3]))
AGR = pd.DataFrame(rows).groupby("comparison").mean().round(3) if rows else pd.DataFrame()
if len(AGR):
    add(AGR, "E3. Agreement: TFT selection weights vs SHAP, between seeds and between models (group level unless stated)",
        "E3_agreement.csv", "Spain: TFT selection weights vs SHAP, Spearman 0.24 on real data and 0.25 on the benchmark.")

# %%
# ---------------- X. Spain vs Portugal, and conclusions ----------------
EX = ["TFT, SHAP", "XGBoost, TreeSHAP"]
MODEL_OF = {"TFT, SHAP": "TFT", "XGBoost, TreeSHAP": "XGBoost"}
def gv(df, idx, col):
    try:
        v = df.loc[idx, col]; v = v.iloc[0] if isinstance(v, pd.Series) else v
        return float(v)
    except Exception: return np.nan
def mean_of(df, e, col, cond=None):
    if df is None or not len(df) or col not in df: return np.nan
    d_ = df[df.explainer == e]
    if cond is not None: d_ = d_[d_.condition == cond]
    return float(d_[col].mean()) if len(d_) else np.nan

# Spanish values from the paper (Tables 5 to 7, Figures 1 and 3, Section 6.6)
ES = {
 ("Synthetic", "Magnitude correctness"): (0.80, 0.44), ("Synthetic", "Lowest pattern r over the four drivers"): (0.92, 0.89),
 ("Synthetic", "b hour"): (0.70, 0.14), ("Synthetic", "b weekday"): (1.07, 0.52), ("Synthetic", "b holiday"): (1.01, 0.78),
 ("Synthetic", "b temperature"): (0.69, 0.30), ("Synthetic", "Share to true causes"): (0.60, 0.40),
 ("Synthetic", "Share to demand history"): (0.19, 0.50), ("Synthetic", "Correctness, no hidden driver"): (0.85, 0.53),
 ("Synthetic", "Correctness, largest hidden driver"): (0.75, 0.39),
 ("Injected", "Magnitude correctness"): (0.78, 0.42), ("Injected", "b event"): (0.86, 0.35), ("Injected", "b driver"): (0.80, 0.49),
 ("Injected", "r event"): (0.79, 0.94), ("Injected", "Decoy credit"): (0.09, 0.03),
 ("Real", "MAPE (%)"): (5.75, 5.00), ("Real", "Gain over weekly naive (%)"): (34.3, 42.7),
 ("Real", "MAE / TSO forecast MAE (not comparable, see A6)"): (6.45, 5.62),
 ("Real", "Size faithfulness to the model"): (0.72, 0.68), ("Real", "Sunday drop, attribution / estimate"): (0.89, 0.57),
 ("Real", "Holiday effect, attribution / estimate"): (2.29, 0.78), ("Real", "Cooling sensitivity / regression"): (185 / 117, 65 / 117),
 ("Real", "Heating sensitivity / regression"): (63 / 105, 50 / 105), ("Real", "Share to demand history"): (np.nan, 0.44),
}
S_def = S_sig[S_sig.condition == "default"] if len(SY) else None
lat_lo = min(LAT_PT.values()); lat_hi = max(LAT_PT.values())
PT = {}
for i, e in enumerate(EX):
    m_ = MODEL_OF[e]
    PT[("Synthetic", "Magnitude correctness", e)] = mean_of(S_def, e, "magnitude_correctness")
    PT[("Synthetic", "Lowest pattern r over the four drivers", e)] = np.nanmin([mean_of(S_def, e, f"r_{g}") for g in DRV]) if S_def is not None else np.nan
    for g, nm in zip(DRV, ["hour", "weekday", "holiday", "temperature"]):
        PT[("Synthetic", f"b {nm}", e)] = mean_of(S_def, e, f"b_{g}")
    PT[("Synthetic", "Share to true causes", e)] = mean_of(S_def, e, "share_direct cause")
    PT[("Synthetic", "Share to demand history", e)] = mean_of(S_def, e, "share_demand history")
    if len(SY) and S_sig.condition.nunique() > 1:
        PT[("Synthetic", "Correctness, no hidden driver", e)] = float(S_sig[(S_sig.explainer == e) & (S_sig.latent_MW == lat_lo)]["magnitude_correctness"].mean())
        PT[("Synthetic", "Correctness, largest hidden driver", e)] = float(S_sig[(S_sig.explainer == e) & (S_sig.latent_MW == lat_hi)]["magnitude_correctness"].mean())
    PT[("Injected", "Magnitude correctness", e)] = mean_of(INJ, e, "magnitude_correctness")
    PT[("Injected", "b event", e)] = mean_of(INJ, e, "b_event"); PT[("Injected", "b driver", e)] = mean_of(INJ, e, "b_driver")
    PT[("Injected", "r event", e)] = mean_of(INJ, e, "r_event"); PT[("Injected", "Decoy credit", e)] = mean_of(INJ, e, "decoy_credit_ratio")
    PT[("Real", "MAPE (%)", e)] = gv(ACC, m_, "MAPE")
    PT[("Real", "Gain over weekly naive (%)", e)] = gv(ACC, m_, "reduction_vs_weekly_naive_pct")
    PT[("Real", "MAE / TSO forecast MAE (not comparable, see A6)", e)] = gv(ACC, m_, "ratio_to_TSO") if "ratio_to_TSO" in ACC else np.nan
    PT[("Real", "Size faithfulness to the model", e)] = float(summ.loc[e, "size_faithfulness"].mean()) if "summ" in globals() and e in summ.index.get_level_values(0) else np.nan
    PT[("Real", "Sunday drop, attribution / estimate", e)] = mean_of(NS, e, "ratio") if "NS" in globals() else np.nan
    PT[("Real", "Holiday effect, attribution / estimate", e)] = mean_of(NH, e, "ratio") if "NH" in globals() else np.nan
    if "tabT" in globals() and e in tabT.index:
        reg_ = tabT.loc["calendar regression (training period)"]
        PT[("Real", "Cooling sensitivity / regression", e)] = gv(tabT, e, "cooling_MW_per_degree") / reg_.get("cooling_MW_per_degree", np.nan)
        PT[("Real", "Heating sensitivity / regression", e)] = gv(tabT, e, "heating_MW_per_degree") / reg_.get("heating_MW_per_degree", np.nan)
    PT[("Real", "Share to demand history", e)] = mean_of(SH, e, "Demand history") if len(SH) else np.nan
rows = []
for (stage, metric), (es_t, es_x) in ES.items():
    for e, es_v in zip(EX, (es_t, es_x)):
        pt_v = PT.get((stage, metric, e), np.nan)
        rows.append(dict(data=stage, metric=metric, explainer=e, Spain=es_v, Portugal=pt_v, difference=pt_v - es_v))
CMPX = pd.DataFrame(rows).set_index(["data", "metric", "explainer"])
add(CMPX.round(3), "X1. Spain vs Portugal: every headline number", "X1_spain_vs_portugal.csv",
    "Spain: values of the paper. Ratios are used where sizes in MW are not comparable between countries.")

# Figure: correctness in both countries
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
lab = ["Synthetic", "Injected"]; xx = np.arange(len(lab)); w_ = 0.2
for j, (e, col) in enumerate(zip(EX, ["#D9622B", "#8A919B"])):
    es_ = [ES[(s_, "Magnitude correctness")][j] for s_ in lab]; pt_ = [PT.get((s_, "Magnitude correctness", e), np.nan) for s_ in lab]
    ax[0].bar(xx + (2 * j - 1.5) * w_, es_, w_, color=col, alpha=0.45, label=f"{e}, Spain")
    ax[0].bar(xx + (2 * j - 0.5) * w_, pt_, w_, color=col, label=f"{e}, Portugal")
ax[0].set_xticks(xx); ax[0].set_xticklabels(["Synthetic benchmark", "Real data with injected truth"]); ax[0].set_ylim(0, 1)
ax[0].set_ylabel("magnitude correctness (1 = perfect)"); ax[0].set_title("(a) Correctness"); ax[0].legend(fontsize=7)
names = ["hour", "weekday", "holiday", "temperature"]; xx = np.arange(4)
for j, (e, col) in enumerate(zip(EX, ["#D9622B", "#8A919B"])):
    ax[1].bar(xx + (2 * j - 1.5) * w_, [ES[("Synthetic", f"b {n}")][j] for n in names], w_, color=col, alpha=0.45)
    ax[1].bar(xx + (2 * j - 0.5) * w_, [PT.get(("Synthetic", f"b {n}", e), np.nan) for n in names], w_, color=col)
ax[1].axhline(1, ls=":", color="k"); ax[1].set_xticks(xx); ax[1].set_xticklabels(names)
ax[1].set_ylabel("slope b (1 = full effect)"); ax[1].set_title("(b) Share of each effect credited, synthetic (light: Spain)")
fig.tight_layout(); fig.savefig(PACK / "fig_X1_spain_vs_portugal.png"); plt.close(fig)

# Automatic checks of each finding of the paper
def P_(s, m, e): return PT.get((s, m, e), np.nan)
T_, X_ = "TFT, SHAP", "XGBoost, TreeSHAP"
checks = [
 ("TFT + SHAP is more correct than XGBoost + TreeSHAP on the synthetic benchmark",
  P_("Synthetic", "Magnitude correctness", T_) > P_("Synthetic", "Magnitude correctness", X_),
  f"{P_('Synthetic', 'Magnitude correctness', T_):.2f} vs {P_('Synthetic', 'Magnitude correctness', X_):.2f} (Spain 0.80 vs 0.44)"),
 ("TFT + SHAP is more correct than XGBoost + TreeSHAP on real data with injected truth",
  P_("Injected", "Magnitude correctness", T_) > P_("Injected", "Magnitude correctness", X_),
  f"{P_('Injected', 'Magnitude correctness', T_):.2f} vs {P_('Injected', 'Magnitude correctness', X_):.2f} (Spain 0.78 vs 0.42)"),
 ("Explanations are right about when (pattern r above 0.85) but not about how much (correctness below 0.9)",
  all(P_("Synthetic", "Lowest pattern r over the four drivers", e) > 0.85 and P_("Synthetic", "Magnitude correctness", e) < 0.9 for e in EX),
  f"lowest r {P_('Synthetic', 'Lowest pattern r over the four drivers', T_):.2f} and {P_('Synthetic', 'Lowest pattern r over the four drivers', X_):.2f}"),
 ("The synthetic correctness holds on real data with injected truth (within 0.05)",
  all(abs(P_("Injected", "Magnitude correctness", e) - P_("Synthetic", "Magnitude correctness", e)) <= 0.05 for e in EX),
  ", ".join(f"{e.split(',')[0]} {P_('Injected', 'Magnitude correctness', e) - P_('Synthetic', 'Magnitude correctness', e):+.2f}" for e in EX) + " (Spain within 0.03)"),
 ("The correctness of TFT + SHAP is similar to Spain (within 0.05)",
  abs(P_("Synthetic", "Magnitude correctness", T_) - 0.80) <= 0.05 and abs(P_("Injected", "Magnitude correctness", T_) - 0.78) <= 0.05,
  f"synthetic {P_('Synthetic', 'Magnitude correctness', T_) - 0.80:+.2f}, injected {P_('Injected', 'Magnitude correctness', T_) - 0.78:+.2f} vs Spain"),
 ("XGBoost leans on demand history more than the TFT",
  P_("Synthetic", "Share to demand history", X_) > P_("Synthetic", "Share to demand history", T_)
  and (np.isnan(P_("Real", "Share to demand history", X_)) or P_("Real", "Share to demand history", X_) > P_("Real", "Share to demand history", T_)),
  f"synthetic {P_('Synthetic', 'Share to demand history', X_):.2f} vs {P_('Synthetic', 'Share to demand history', T_):.2f}, "
  f"real {P_('Real', 'Share to demand history', X_):.2f} vs {P_('Real', 'Share to demand history', T_):.2f}"),
 ("A hidden driver lowers correctness for both explanations",
  all(P_("Synthetic", "Correctness, largest hidden driver", e) < P_("Synthetic", "Correctness, no hidden driver", e) for e in EX)
  if not np.isnan(P_("Synthetic", "Correctness, no hidden driver", T_)) else None,
  ", ".join(f"{e.split(',')[0]} {P_('Synthetic', 'Correctness, no hidden driver', e):.2f} to {P_('Synthetic', 'Correctness, largest hidden driver', e):.2f}" for e in EX)),
 ("On real data, the TFT credits the Sunday drop closer to its estimated size than XGBoost",
  abs(1 - P_("Real", "Sunday drop, attribution / estimate", T_)) < abs(1 - P_("Real", "Sunday drop, attribution / estimate", X_)),
  f"{P_('Real', 'Sunday drop, attribution / estimate', T_):.2f} vs {P_('Real', 'Sunday drop, attribution / estimate', X_):.2f} (Spain 0.89 vs 0.57)"),
 ("Forecast accuracy does not reveal which explanation is more correct",
  P_("Real", "MAPE (%)", T_) >= P_("Real", "MAPE (%)", X_) - 0.25 and P_("Synthetic", "Magnitude correctness", T_) > P_("Synthetic", "Magnitude correctness", X_),
  f"MAPE {P_('Real', 'MAPE (%)', T_):.2f}% vs {P_('Real', 'MAPE (%)', X_):.2f}% (Spain 5.75% vs 5.00%)"),
]
if len(SY) and "agree_spearman" in SY:
    sw = SY[(SY.condition == "default") & (SY.explainer == "TFT, variable selection (unsigned)")]
    checks.append(("TFT selection weights are not effect sizes (weak agreement with SHAP, more credit to irrelevant inputs)",
                   sw["agree_spearman"].mean() < 0.5 and sw["share_irrelevant"].mean() > mean_of(S_def, T_, "share_irrelevant"),
                   f"Spearman {sw['agree_spearman'].mean():.2f} (Spain 0.25), irrelevant share {sw['share_irrelevant'].mean():.2f} (Spain 0.32)"))
CHK = pd.DataFrame([dict(finding=f, holds_in_Portugal={True: "yes", False: "no", None: "not run"}[None if h is None else bool(h)], evidence=ev)
                    for f, h, ev in checks]).set_index("finding")
add(CHK, "X2. Does each finding of the Spanish study hold in Portugal?", "X2_findings_check.csv")

n_yes = (CHK.holds_in_Portugal == "yes").sum(); n_tested = (CHK.holds_in_Portugal != "not run").sum()
core = CHK.holds_in_Portugal.iloc[[0, 1, 2, 3]].eq("yes").all()
lines = [f"**Conclusion.** {n_yes} of {n_tested} findings of the Spanish study hold in Portugal."]
lines.append("The core result replicates: TFT + SHAP is the more correct explanation on the synthetic benchmark and on real data with injected truth, "
             "explanations get the timing of each driver right but not its size, and the synthetic ranking carries over to real data."
             if core else "The core result does not fully replicate. The failed checks above show where Portugal differs from Spain.")
lines.append(f"SHAP on the TFT reaches {P_('Synthetic', 'Magnitude correctness', T_):.2f} on the Portuguese benchmark and "
             f"{P_('Injected', 'Magnitude correctness', T_):.2f} with injected truth (Spain 0.80 and 0.78). "
             f"TreeSHAP on XGBoost reaches {P_('Synthetic', 'Magnitude correctness', X_):.2f} and {P_('Injected', 'Magnitude correctness', X_):.2f} (Spain 0.44 and 0.42).")
fails = [f for f, h in CHK.holds_in_Portugal.items() if h == "no"]
if fails: lines.append("Findings that do not hold: " + "; ".join(fails) + ".")
CONCLUSION = "\n\n".join(lines); print(CONCLUSION)
MDL.extend(["## Conclusion", "", CONCLUSION, ""])

# %%
json.dump(dict(country=COUNTRY, run_plan=RUN_PLAN, models=MODELS, seeds=SEEDS, best_params={k: BEST[k] for k in MODELS if k in BEST},
               bench_arch=BENCH_ARCH, train=TRAIN, window=WINDOW, n_explain=N_EXPLAIN, shap_nsamples=SHAP_NSAMPLES,
               inject=INJECT_PT, ratio=RATIO, generator=GEN_PT, latent=LAT_PT, energy_source=(DATA_DIR / "SOURCE.txt").read_text() if (DATA_DIR / "SOURCE.txt").exists() else "",
               split=dict(val_start=str(VAL_START.date()), test_start=str(TEST_START.date())), runs=len(LOGDF)),
          open(PACK / "settings.json", "w"), indent=2, default=str)
LOGDF.to_csv(PACK / "log.csv", index=False)
(PACK / "RESULTS_PORTUGAL.md").write_text("\n".join(MDL), encoding="utf8")
zp = PACK.with_suffix(".zip")
with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
    for f in PACK.iterdir(): z.write(f, f.name)
print("Results pack:", zp)


# ======================================================================================================================
# ==== PART 6: Part 6. How SHAP is combined, LIME, faithfulness and robustness indicators, correlated inputs
# ======================================================================================================================

# %% [markdown]
# # Part 6. How SHAP is combined, LIME, faithfulness and robustness indicators, correlated inputs
#
# **In the paper:** Sections 4.3, 4.4 and 6.8 and Appendix B. Results are written to `results/extensions/`. No data files are needed.
#
# Every test runs on the synthetic benchmark of the paper, where the true contribution of every driver is recorded each hour. So every choice and every indicator is checked against the truth instead of argued for. Models and settings are those of the paper (TFT + SHAP, XGBoost + TreeSHAP), on the default benchmark version with five seeds.
#
# | Part | Question | What is computed |
# |---|---|---|
# | A | **How should SHAP be combined over the sliding 24-hour input window?** | Per forecast: sum over lags (paper), last lag only, mean over lags, and the exact target of the question, the **grouped Shapley value** (each feature's whole 24-hour series is one player), estimated by permutation sampling. Also grouped at the level of the four drivers. The gap between sum and grouped value is the cross-feature interaction term of Appendix B |
# | B | **How should explanations be combined over many forecasts or test periods?** | Mean signed, mean absolute, root mean square and rank-based (mean rank) global importance, each compared with the global importance of the true components |
# | C | **Add LIME** | LIME (Ribeiro et al., default kernel) and, with the same samples, KernelSHAP (Shapley kernel). The two differ only in the weighting kernel, so the comparison isolates that choice |
# | D | **New quality indicators: robustness and faithfulness** | Max-sensitivity (Yeh et al., 2019), retraining stability, faithfulness correlation (Bhatt et al., 2020), infidelity (Yeh et al., 2019) and deletion. Each indicator is then tested against the true correctness, as in RQ1, including the halved and inflated copies |
# | E | **Correlated (covariant) inputs** | XGBoost: TreeSHAP in its path-dependent form (used in the paper, closer to conditional) against the interventional form; TreeSHAP interaction values to measure how much of each prediction comes from interactions between features. TFT: grouping correlated features of a driver into one player |
#
# **Settings.** The plan (`smoke` or `deep`), the number of seeds and the training length come from the command line (`--plan`, `--seeds`, `--max-epochs`, ...). Everything else is set in the first lines of this part. Finished runs are saved and skipped when the part is run again.

# %%
# The launcher passes the shared settings as GLOBAL_SETTINGS (plan, seeds, training length, explained hours, search size).
import json
from pathlib import Path
print(GLOBAL_SETTINGS)

# %%
# ============================== CONFIGURATION ==============================
from pathlib import Path
OUT_DIR = Path("results/extensions")
GENERATOR = "spain"                    # "spain" (paper) or the path of generator_portugal.json from the Portugal notebook
TZ = "Europe/Madrid"                   # "Europe/Lisbon" with the Portuguese generator
RUN_PLAN = "smoke" if GLOBAL_SETTINGS["plan"] == "smoke" else "full"
MODELS = ["TFT", "XGBoost"]
SEEDS = list(range(GLOBAL_SETTINGS["seeds"]))
CONDITIONS = {"default": 1200}         # hidden-driver size in MW per unit (Spanish scale); add "latent_2400": 2400 for a stress test
WINDOW = 24
ARCH = dict(LSTM=dict(hidden=128, layers=2, dropout=0.1), Transformer=dict(d_model=64, heads=4, layers=2, dropout=0.1),
            TFT=dict(d_model=32, heads=4, dropout=0.1), RETAIN=dict(hidden=128, dropout=0.2))
TRAIN = dict(lr=1e-3, weight_decay=1e-4, batch=256, max_epochs=GLOBAL_SETTINGS["max_epochs"], patience=GLOBAL_SETTINGS["patience"], clip=1.0)
SHAP_BACKGROUND, SHAP_NSAMPLES = 300, 200        # as in the paper

N_EXPLAIN = GLOBAL_SETTINGS["n_explain"]   # forecasts explained with SHAP (parts A sum/last/mean, B)
N_GROUPED = 100           # forecasts explained with the grouped Shapley value, LIME and KernelSHAP (subset of the above)
PERM_SAMPLES = 8          # orderings per forecast for the grouped Shapley value (plus their reverses, so 16)
GROUPED_BG = 16           # background instances averaged in every coalition (grouped Shapley), one fixed set per run
LIME_SAMPLES = 1000       # coalitions per forecast for LIME and KernelSHAP
LIME_BG = 8               # background instances averaged in every coalition (LIME, KernelSHAP), one fixed set per run
N_ROBUST, ROBUST_K, ROBUST_R = 50, 5, 0.1        # max-sensitivity: forecasts, perturbations, radius in standard deviations
N_FAITH, FAITH_K, FAITH_FRAC = 200, 30, 0.2       # faithfulness: forecasts, random subsets, subset size as share of features
RETRAIN_STABILITY = True  # train a second model on the same data with another seed and compare explanations
N_INTERACT = 200          # forecasts for TreeSHAP interaction values (XGBoost)

if RUN_PLAN == "smoke":
    SEEDS = [0]; N_EXPLAIN, N_GROUPED, PERM_SAMPLES, LIME_SAMPLES, GROUPED_BG, LIME_BG = 80, 12, 2, 120, 4, 2
    N_ROBUST, ROBUST_K, N_FAITH, FAITH_K, N_INTERACT, SHAP_NSAMPLES = 6, 2, 20, 8, 20, 30
    TRAIN.update(max_epochs=2, patience=1)
RUN_DIR = OUT_DIR / RUN_PLAN

# %%
import itertools, json, time, warnings, zipfile, math
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import Ridge, LinearRegression
warnings.filterwarnings("ignore")
(RUN_DIR / "attributions").mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 200, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
try:
    show = display
except NameError:
    show = print
import xgboost as xgb, shap
HAVE_XGB = HAVE_SHAP = True
LOG = RUN_DIR / "log.jsonl"
def logged():
    return [json.loads(l) for l in LOG.read_text().splitlines() if l.strip()] if LOG.exists() else []
def log(row):
    with open(LOG, "a") as f: f.write(json.dumps(row, default=float) + "\n")
def is_done(**key):
    return any(all(r.get(k) == v for k, v in key.items()) for r in logged())

# %% [markdown]
# ## 1. Benchmark generator and scoring (the paper's code, unchanged)

# %%
BENCH_CODE = r'''
import itertools, json, re, time, warnings
import numpy as np, pandas as pd
from scipy import stats
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.preprocessing import StandardScaler
try:
    import xgboost as xgb
except ImportError:
    xgb = None
try:
    import shap
except ImportError:
    shap = None

from pathlib import Path

OUT_DIR = Path("results/benchmark_experiments")
TZ = "Europe/Madrid"
START, END = "2015-01-01 00:00", "2018-12-31 23:00"
SPLIT = (0.64, 0.16, 0.20)
LAG_L = 36                      # strict day-ahead (fixed-lag protocol, as in Chapter 8)
SEEDS = [0, 1, 2, 3, 4]
N_EXPLAIN = 1000                # explained test windows per run (as in the report)
SEED_SAMPLE = 0

# ---- conditions: variants (Section 5.3), hidden-driver sweep, interaction variant ----
CONDITIONS = [
    dict(name="default",       change="2016-07-01", latent=1200, interaction=0.0),
    dict(name="extrapolation", change="2017-07-01", latent=1200, interaction=0.0),
    dict(name="stationary",    change=None,         latent=1200, interaction=0.0),
    dict(name="latent_0",      change="2016-07-01", latent=0,    interaction=0.0),
    dict(name="latent_600",    change="2016-07-01", latent=600,  interaction=0.0),
    dict(name="latent_2400",   change="2016-07-01", latent=2400, interaction=0.0),
    dict(name="interaction",   change="2016-07-01", latent=1200, interaction=0.5),
]

# ---- generator parameters (Section 5.2). ASSUMED = not stated in the draft ----
GEN = dict(
    base=28000, hour_amp=(2600, 900), hour_phase=(9, 3),     # phases ASSUMED
    sat=-1800, sun=-3200, holiday=-2500,
    heat_thr=16, heat_slope=250, cool_thr=22, cool_coef=150, cool_exp=1.3, cool_factor_after=1.6,
    latent_phi=0.995, ar_phi=0.95, ar_sd=500,                 # ar_sd (stationary sd, MW) ASSUMED
    noise_sd=150, operator_sd=200, gas_share=0.55, gas_latent=400, gas_noise_sd=300,   # gas_latent, gas_noise_sd ASSUMED
)
USE_EXTERNAL_GENERATOR = None   # e.g. "data/synthetic/{name}_seed{seed}" with energy_dataset.csv, weather_features.csv, truth_components.csv
WRITE_FILES = False             # write each realization in Kaggle format for the main pipeline / LEAR notebook
FILES_DIR = Path("data/synthetic")

# ---- models and hooks ----
USE_XGB = True
EXTERNAL_ATTRIBUTIONS = {}      # {"SHAP on RETAIN": "ext/{name}_seed{seed}_retain_shap.csv"}  columns: time + feature columns
EXTERNAL_FEATURE_GROUPS = {}    # regex -> group for external feature names, e.g. {r"^hour_": "hour", r"^dow_|weekend": "week", ...}
EXTERNAL_MODELS = {}            # {"RETAIN": wrapper_instance} with .fit(X, y), .predict(X), .explain(X) -> (n, F) in MW

# ---- runtime ----
FAST_MODE = False               # smoke test: 2 conditions, 2 seeds, 1 year, no XGBoost tuning
RUN_CAUSAL = True
CAUSAL_SEEDS = 3                # PCMCI+ takes minutes per run; seeds used for causal discovery
CAUSAL_HOURS = 24 * 365         # hours of the training period used for causal discovery
def spanish_holidays(years):
    try:
        import holidays as _h
        return set(_h.Spain(years=years).keys())
    except ImportError:
        fixed = [(1, 1), (1, 6), (5, 1), (8, 15), (10, 12), (11, 1), (12, 6), (12, 8), (12, 25)]
        gf = {2015: (4, 3), 2016: (3, 25), 2017: (4, 14), 2018: (3, 30), 2019: (4, 19)}
        out = {pd.Timestamp(y, m, d).date() for y in years for m, d in fixed}
        return out | {pd.Timestamp(y, *gf[y]).date() for y in years if y in gf}

def _ar(n, phi, sd, rng):
    """AR(1) with stationary standard deviation sd."""
    e = rng.normal(0, sd * np.sqrt(1 - phi ** 2), n); x = np.empty(n); x[0] = rng.normal(0, sd)
    for i in range(1, n): x[i] = phi * x[i - 1] + e[i]
    return x

CITIES = {"Madrid": 0.0, " Barcelona": 1.5, "Valencia": 2.5, "Seville": 4.0, "Bilbao": -1.5}

def generate(cond, seed, g=GEN):
    rng = np.random.default_rng(1000 * seed + 17)
    idx = pd.date_range(pd.Timestamp(START, tz=TZ).tz_convert("UTC"),
                        pd.Timestamp(END, tz=TZ).tz_convert("UTC"), freq="h")
    loc = idx.tz_convert(TZ); n = len(idx)
    h, dow, doy = loc.hour.values, loc.dayofweek.values, loc.dayofyear.values
    day = loc.tz_localize(None).normalize()
    hols = spanish_holidays(range(loc.year.min(), loc.year.max() + 1))
    is_hol = np.array([d.date() in hols for d in day])

    # weather: five cities, seasonal + daily cycle + persistent national anomaly + local noise
    anom = _ar(n, 0.995, 2.5, rng)
    temps = {c: 15 + off - 9 * np.cos(2 * np.pi * (doy - 20) / 365.25) + (5 + 0.3 * off) * np.sin(2 * np.pi * (h - 9) / 24)
                + anom + _ar(n, 0.9, 0.8, rng) for c, off in CITIES.items()}
    T = np.mean(list(temps.values()), axis=0)
    clouds = 100 / (1 + np.exp(-_ar(n, 0.98, 1.5, rng)))
    humidity = np.clip(60 + 0.3 * (clouds - 50) - 1.2 * (T - 15) + rng.normal(0, 4, n), 10, 100)
    pressure = 1015 + _ar(n, 0.99, 6, rng)
    wind_speed = np.abs(4 + _ar(n, 0.97, 2.5, rng))

    # demand components
    C_hour = g["hour_amp"][0] * np.sin(2 * np.pi * (h - g["hour_phase"][0]) / 24) + \
             g["hour_amp"][1] * np.sin(4 * np.pi * (h - g["hour_phase"][1]) / 24)
    C_week = np.where(dow == 5, g["sat"], np.where(dow == 6, g["sun"], 0)).astype(float)
    C_hol = np.where(is_hol, g["holiday"], 0).astype(float)
    k = np.ones(n)
    if cond["change"]:
        k[loc.tz_localize(None) >= pd.Timestamp(cond["change"])] = g["cool_factor_after"]
    heat = g["heat_slope"] * np.clip(g["heat_thr"] - T, 0, None)
    cool = k * g["cool_coef"] * np.clip(T - g["cool_thr"], 0, None) ** g["cool_exp"]
    C_temp = heat + cool
    C_int = cond["interaction"] * cool * np.sin(2 * np.pi * (h - 10) / 24)
    days_u, day_pos = np.unique(day.values, return_inverse=True)
    z = _ar(len(days_u), g["latent_phi"], 1.0, rng)
    z_h = np.interp(day_pos + h / 24, np.arange(len(days_u)), z)
    C_lat = cond["latent"] * z_h
    C_AR = _ar(n, g["ar_phi"], g["ar_sd"], rng)
    eps = rng.normal(0, g["noise_sd"], n)
    D = g["base"] + C_hour + C_week + C_hol + C_temp + C_int + C_lat + C_AR + eps

    # generation: consequences of demand (gas, coal) and independent sources
    season = 0.6 + 0.4 * np.sin(2 * np.pi * (doy - 80) / 365.25)
    solar = 7000 * np.clip(np.sin(np.pi * (h - 7) / 13), 0, None) * season * (1 - 0.6 * clouds / 100)
    wind = np.minimum(15000, 60 * wind_speed ** 3)
    nuclear = np.full(n, 7000.0)
    for s in rng.choice(np.arange(0, n - 24 * 30), 3, replace=False):
        nuclear[s:s + 24 * 30] -= 1000
    hydro = 3000 + 1500 * np.sin(2 * np.pi * (doy - 60) / 365.25) + rng.normal(0, 150, n)
    resid = D - solar - wind - nuclear - hydro
    gas = g["gas_share"] * resid + g["gas_latent"] * z_h * (cond["latent"] > 0) + _ar(n, 0.9, g.get("gas_noise_sd", 0), rng)
    coal = np.clip(resid - gas, 0, None)
    ar_known = np.r_[np.full(LAG_L, 0.0), C_AR[:-LAG_L]] * g["ar_phi"] ** LAG_L
    operator = D - eps - C_AR + ar_known + rng.normal(0, g["operator_sd"], n)

    energy = pd.DataFrame({
        "generation solar": solar, "generation wind onshore": wind, "generation nuclear": nuclear,
        "generation hydro water reservoir": hydro, "generation fossil gas": gas, "generation fossil hard coal": coal,
        "forecast solar day ahead": solar + rng.normal(0, 200, n), "forecast wind onshore day ahead": wind + rng.normal(0, 800, n),
        "total load forecast": operator, "total load actual": D,
        "price day ahead": 30 + 0.002 * gas + rng.normal(0, 3, n)}, index=idx)
    weather = dict(temp=T, humidity=humidity, pressure=pressure, wind_speed=wind_speed, clouds_all=clouds, cities=temps)
    truth = pd.DataFrame(dict(C_hour=C_hour, C_week=C_week, C_holiday=C_hol, C_temp=C_temp, C_interaction=C_int,
                              C_latent=C_lat, C_AR=C_AR, eps=eps, ar_known=ar_known), index=idx)
    return energy, weather, truth, hols

def write_kaggle(energy, weather, truth, folder):
    folder.mkdir(parents=True, exist_ok=True)
    t = energy.index.tz_convert(TZ).strftime("%Y-%m-%d %H:%M:%S%z")
    t = [s[:-2] + ":" + s[-2:] for s in t]
    e = energy.copy(); e.insert(0, "time", t); e["price actual"] = e["price day ahead"]
    e.to_csv(folder / "energy_dataset.csv", index=False)
    rows = [pd.DataFrame({"dt_iso": t, "city_name": c, "temp": v + 273.15, "humidity": weather["humidity"],
                          "pressure": weather["pressure"], "wind_speed": weather["wind_speed"],
                          "clouds_all": weather["clouds_all"]}) for c, v in weather["cities"].items()]
    pd.concat(rows).to_csv(folder / "weather_features.csv", index=False)
    tr = truth.copy(); tr.insert(0, "time", t); tr.to_csv(folder / "truth_components.csv", index=False)


GROUPS = {"hour": ["hour_sin", "hour_cos"], "week": ["dow_sin", "dow_cos", "weekend"], "holiday": ["holiday"],
          "temperature": ["temp", "hdd", "cdd", "temp_day_mean", "temp_day_max", "temp_day_min"]}
TRUTH_COL = {"hour": "C_hour", "week": "C_week", "holiday": "C_holiday", "temperature": "C_temp"}

def build_features(energy, weather, hols, L=LAG_L):
    loc = energy.index.tz_convert(TZ); day = loc.tz_localize(None).normalize()
    X = pd.DataFrame(index=energy.index)
    X["hour_sin"], X["hour_cos"] = np.sin(2 * np.pi * loc.hour / 24), np.cos(2 * np.pi * loc.hour / 24)
    X["dow_sin"], X["dow_cos"] = np.sin(2 * np.pi * loc.dayofweek / 7), np.cos(2 * np.pi * loc.dayofweek / 7)
    X["doy_sin"], X["doy_cos"] = np.sin(2 * np.pi * loc.dayofyear / 365.25), np.cos(2 * np.pi * loc.dayofyear / 365.25)
    X["weekend"] = (loc.dayofweek >= 5).astype(float)
    inh = lambda dd: np.array([x.date() in hols for x in dd], float)
    X["holiday"] = inh(day)
    X["pre_holiday"], X["post_holiday"] = inh(day + pd.Timedelta(days=1)), inh(day - pd.Timedelta(days=1))
    X["bridge"] = (((loc.dayofweek == 0) & (X["pre_holiday"] == 1)) | ((loc.dayofweek == 4) & (X["post_holiday"] == 1))).astype(float)
    T = pd.Series(weather["temp"], index=energy.index)
    X["temp"] = T
    for c in ("humidity", "pressure", "wind_speed", "clouds_all"):
        X[c] = weather[c]
    X["hdd"], X["cdd"] = np.clip(18 - T, 0, None), np.clip(T - 18, 0, None)
    gb = T.groupby(day.values)
    X["temp_day_mean"], X["temp_day_max"], X["temp_day_min"] = (gb.transform(f).values for f in ("mean", "max", "min"))
    y = energy["total load actual"]
    for lag in sorted({L, 48, 168} | {l for l in (24, 48, 168) if l >= L}):
        X[f"load_lag{lag}"] = y.shift(lag)
    X["load_rollmean24"] = y.rolling(24).mean().shift(L)
    X["load_rollmean168"] = y.rolling(168).mean().shift(L)
    X["load_rollstd6"] = y.rolling(6).std().shift(L)
    X["load_diff1"] = y.diff(1).shift(L); X["load_diff24"] = y.diff(24).shift(L)
    gl = max(24, L)
    for c, nm in [("generation solar", "solar"), ("generation wind onshore", "wind_gen"), ("generation nuclear", "nuclear"),
                  ("generation hydro water reservoir", "hydro"), ("generation fossil gas", "gas"), ("generation fossil hard coal", "coal")]:
        X[f"{nm}_lag{gl}"] = energy[c].shift(gl)
    ok = X.notna().all(axis=1)
    return X[ok], y[ok]

def role(f):
    if any(f in v for v in GROUPS.values()): return "direct cause"
    if f.startswith("load_"): return "demand history"
    if f.startswith(("gas_", "coal_")): return "consequence"
    return "irrelevant"

def concept(f):
    for g, v in GROUPS.items():
        if f in v: return {"week": "day of week"}.get(g, g)
    if f.startswith("load_"): return "demand history"
    return re.sub(r"_lag\d+$", "", f)

TRUE_PARENTS = {"hour", "day of week", "holiday", "temperature", "demand history"}

class RidgeSHAP:
    name = "Ridge, linear SHAP"
    def fit(self, X, y, Xv=None, yv=None):
        self.sc = StandardScaler().fit(X); self.m = RidgeCV(alphas=np.logspace(-2, 4, 25)).fit(self.sc.transform(X), y)
        self.mu = self.sc.transform(X).mean(0); return self
    def predict(self, X): return self.m.predict(self.sc.transform(X))
    def explain(self, X): return (self.sc.transform(X) - self.mu) * self.m.coef_

class XGBTreeSHAP:
    name = "XGBoost, TreeSHAP"
    def fit(self, X, y, Xv=None, yv=None):
        params = dict(n_estimators=2000, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                      min_child_weight=5, early_stopping_rounds=50, n_jobs=-1, verbosity=0)
        if FAST_MODE: params.update(n_estimators=300, early_stopping_rounds=20)
        self.m = xgb.XGBRegressor(**params).fit(X, y, eval_set=[(Xv, yv)], verbose=False)
        self.ex = shap.TreeExplainer(self.m); return self
    def predict(self, X): return self.m.predict(X)
    def explain(self, X): return np.asarray(self.ex.shap_values(X))

def controls(base_attr, rng, feats=None):
    s = base_attr.std()
    out = {"control: random": rng.normal(0, s, base_attr.shape)}
    for k in (0.5, 1.0, 2.0):
        out[f"control: noise x{k}"] = base_attr + rng.normal(0, k * s, base_attr.shape)
    perm = rng.permutation(base_attr.shape[1])
    out["control: permuted labels"] = base_attr[:, perm]
    # magnitude controls: same ranking or same pattern, wrong size
    out["control: shrunk x0.5"] = 0.5 * base_attr
    out["control: inflated x1.5"] = 1.5 * base_attr
    if feats is not None:
        cal = [k for k, f in enumerate(feats) if f in sum(GROUPS.values(), [])]
        hist = [k for k, f in enumerate(feats) if f.startswith("load_")]
        if cal and hist:
            a = base_attr.copy(); moved = 0.5 * a[:, cal]
            a[:, cal] -= moved; a[:, hist[0]] += moved.sum(1)
            out["control: half of drivers moved to history"] = a
    return out

def split_idx(n):
    a, b = int(SPLIT[0] * n), int((SPLIT[0] + SPLIT[1]) * n)
    return slice(0, a), slice(a, b), slice(b, n)
def gt_scores(A, feats, truth_s, interaction=False):
    A = pd.DataFrame(A, columns=feats); out = {}
    tr = {g: truth_s[c].values.copy() for g, c in TRUTH_COL.items()}
    conv = {"": tr}
    if interaction:
        ci = truth_s["C_interaction"].values
        conv = {"": {**tr, "temperature": tr["temperature"] + ci},
                "_split": {**tr, "temperature": tr["temperature"] + ci / 2, "hour": tr["hour"] + ci / 2}}
    for suffix, tt in conv.items():
        bs = []
        for g, cols in GROUPS.items():
            a = A[[c for c in cols if c in A]].sum(1).values; c = tt[g] - tt[g].mean()
            if np.std(c) < 1e-9 or np.std(a) < 1e-12:
                out[f"r_{g}{suffix}"], out[f"b_{g}{suffix}"] = np.nan, 0.0 if np.std(c) > 1e-9 else np.nan
            else:
                out[f"r_{g}{suffix}"] = np.corrcoef(a, c)[0, 1]; out[f"b_{g}{suffix}"] = np.cov(a, c)[0, 1] / np.var(c, ddof=1)
            bs.append(out[f"b_{g}{suffix}"])
        out[f"magnitude_correctness{suffix}"] = 1 - np.nanmean(np.abs(1 - np.array(bs, float)))
        out[f"pattern_correctness{suffix}"] = np.nanmean([out[f"r_{g}{suffix}"] for g in GROUPS])
    tot = A.abs().sum().sum()
    for rl in ("direct cause", "demand history", "consequence", "irrelevant"):
        out[f"share_{rl}"] = A[[f for f in feats if role(f) == rl]].abs().sum().sum() / tot
    imp = A.abs().mean().groupby(lambda f: concept(f)).sum().sort_values(ascending=False)
    named = set(imp.index[:5]); tp = len(named & TRUE_PARENTS)
    out["parents_precision"], out["parents_recall"] = tp / 5, tp / len(TRUE_PARENTS)
    out["parents_F1"] = 0 if tp == 0 else 2 * tp / (5 + len(TRUE_PARENTS))
    out["false_parents_consequence"] = len(named & {"gas", "coal"})
    return out

def deletion_insertion(model, Xs, y, A, mu, K=10):
    """Most-relevant-first perturbation on the model output (per-forecast rankings by |A|).
    Deletion: replace the top-k features by their training mean and record |f(x) - f(x_-k)|;
    a faithful ranking moves the prediction the most for small k.
    Insertion: start from all features at the mean, restore the top-k, record |f(x_+k) - f(x)|;
    a faithful ranking recovers the prediction fastest. Both use k = 1..K and are scaled by the
    standard deviation of the predictions, so higher = more faithful for both."""
    n, F = Xs.shape; order = np.argsort(-np.abs(A), axis=1); rows = np.arange(n)[:, None]
    f = lambda M: model.predict(pd.DataFrame(M, columns=FEATS_CUR))
    f0 = f(Xs); scale = max(np.std(f0), 1e-9); dele, ins = [], []
    for m in range(1, min(K, F) + 1):
        M = Xs.copy(); M[rows, order[:, :m]] = mu[order[:, :m]]; dele.append(np.mean(np.abs(f0 - f(M))))
        M = np.tile(mu, (n, 1)); M[rows, order[:, :m]] = Xs[rows, order[:, :m]]; ins.append(np.mean(np.abs(f(M) - f0)))
    return float(np.mean(dele) / scale), float(1 - np.mean(ins) / scale)

def regime_stability(A, err):
    hi = np.abs(err) >= np.quantile(np.abs(err), 0.9)
    v1, v0 = A[hi].var(0), A[~hi].var(0); ok = v0 > 1e-12
    return float(np.median(v1[ok] / v0[ok]))

def eval_metrics(model, Xte, yte, A, mu, ridge_top10, feats):
    out = {}
    top10 = set(np.array(feats)[np.argsort(-np.abs(A).mean(0))[:10]])
    out["metric_recall@10"] = len(top10 & ridge_top10) / 10
    out["metric_deletion"], out["metric_insertion"] = deletion_insertion(model, Xte.values, yte.values, A, mu)
    out["metric_regime_stability"] = regime_stability(A, model.predict(Xte) - yte.values)
    return out
'''

# ---- generator: the paper's benchmark code, with the hooks of the Portugal notebook (identical output for Spain) ----
PATCHES = [
    ("temps = {c: 15 + off - 9 * np.cos(2 * np.pi * (doy - 20) / 365.25) + (5 + 0.3 * off) * np.sin(2 * np.pi * (h - 9) / 24)",
     "temps = {c: g.get('t_mean', 15) + off - g.get('t_season', 9) * np.cos(2 * np.pi * (doy - 20) / 365.25) + (g.get('t_daily', 5) + 0.3 * off) * np.sin(2 * np.pi * (h - 9) / 24)"),
    ("solar = 7000 *", "solar = g.get('gen_scale', 1.0) * 7000 *"),
    ("wind = np.minimum(15000, 60 * wind_speed ** 3)", "wind = g.get('gen_scale', 1.0) * np.minimum(15000, 60 * wind_speed ** 3)"),
    ("nuclear = np.full(n, 7000.0)", "nuclear = np.full(n, 7000.0 * g.get('nuclear_scale', 1.0))"),
    ("nuclear[s:s + 24 * 30] -= 1000", "nuclear[s:s + 24 * 30] -= 1000 * g.get('nuclear_scale', 1.0)"),
    ("hydro = 3000 + 1500 * np.sin(2 * np.pi * (doy - 60) / 365.25) + rng.normal(0, 150, n)",
     "hydro = g.get('gen_scale', 1.0) * (3000 + 1500 * np.sin(2 * np.pi * (doy - 60) / 365.25) + rng.normal(0, 150, n))"),
    ('"forecast solar day ahead": solar + rng.normal(0, 200, n), "forecast wind onshore day ahead": wind + rng.normal(0, 800, n)',
     '"forecast solar day ahead": solar + g.get("gen_scale", 1.0) * rng.normal(0, 200, n), '
     '"forecast wind onshore day ahead": wind + g.get("gen_scale", 1.0) * rng.normal(0, 800, n)'),
]
code = BENCH_CODE
for old, new in PATCHES:
    assert code.count(old) == 1, f"generator text not found: {old[:60]}"
    code = code.replace(old, new)
B = {}; exec(code, B)
B["FAST_MODE"] = RUN_PLAN == "smoke"
if RUN_PLAN == "smoke": B["START"] = "2017-06-01 00:00"
if GENERATOR == "spain":
    GEN_USE, LAT_SCALE = dict(B["GEN"]), 1.0
else:
    gj = json.loads(Path(GENERATOR).read_text()); GEN_USE, LAT_SCALE = gj["GEN_PT"], gj["ratio"]
    GEN_USE["hour_amp"], GEN_USE["hour_phase"] = tuple(GEN_USE["hour_amp"]), tuple(GEN_USE["hour_phase"])
    def pt_holidays(years):
        try:
            import holidays as _h
            return set(_h.country_holidays("PT", years=list(years)).keys())
        except ImportError:
            fixed = [(1, 1), (4, 25), (5, 1), (6, 10), (8, 15), (12, 8), (12, 25)]
            gf = {2015: (4, 3), 2016: (3, 25), 2017: (4, 14), 2018: (3, 30)}
            out = {pd.Timestamp(y, m, d).date() for y in years for m, d in fixed}
            out |= {pd.Timestamp(y, m, d).date() for y in years if y >= 2016 for m, d in [(10, 5), (11, 1), (12, 1)]}
            return out | {pd.Timestamp(y, *gf[y]).date() for y in years if y in gf}
    B["spanish_holidays"] = pt_holidays
    B["CITIES"] = {"Lisbon": 0.5, "Porto": -0.5, "Braga": -1.5, "Coimbra": 0.0, "Faro": 1.5}
B["TZ"] = TZ
DRIVERS = list(B["GROUPS"])                       # hour, week, holiday, temperature
print("generator:", GENERATOR, "| drivers:", DRIVERS)

# %% [markdown]
# ## 2. Models and SHAP (the paper's code, unchanged)

# %%
import torch, torch.nn as nn
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", DEVICE, "| torch", torch.__version__, "| shap", shap.__version__ if HAVE_SHAP else "missing")
assert HAVE_SHAP, "pip install shap"

class LSTMNet(nn.Module):
    def __init__(self, F, hidden, layers, dropout):
        super().__init__()
        self.rnn = nn.LSTM(F, hidden, layers, batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(self, x):
        h, _ = self.rnn(x); return self.head(h[:, -1]).squeeze(-1)

class RETAINNet(nn.Module):
    def __init__(self, F, hidden, dropout):
        super().__init__()
        self.emb = nn.Linear(F, hidden)
        self.gru_a = nn.GRU(hidden, hidden, batch_first=True); self.gru_b = nn.GRU(hidden, hidden, batch_first=True)
        self.att_a = nn.Linear(hidden, 1); self.att_b = nn.Linear(hidden, hidden)
        self.drop = nn.Dropout(dropout); self.out = nn.Linear(hidden, 1)
    def attention(self, x):
        v = self.emb(x)
        rv = torch.flip(self.drop(v), [1])                       # reverse time
        g, _ = self.gru_a(rv); h, _ = self.gru_b(rv)
        g, h = torch.flip(g, [1]), torch.flip(h, [1])
        alpha = torch.softmax(self.att_a(g).squeeze(-1), dim=1)   # (n, T)
        beta = torch.tanh(self.att_b(h))                         # (n, T, H)
        return v, alpha, beta
    def forward(self, x):
        v, a, b = self.attention(x)
        return self.out((a.unsqueeze(-1) * b * v).sum(1)).squeeze(-1)
    def contributions(self, x):
        """Exact contribution (n, T, F) and the feature-independent bias (n,)."""
        v, a, b = self.attention(x)
        w, W = self.out.weight.squeeze(0), self.emb.weight          # (H,), (H, F)
        coef = torch.einsum("nt,nth,h,hf->ntf", a, b, w, W)
        bias = ((a.unsqueeze(-1) * b * self.emb.bias).sum(1) * w).sum(-1) + self.out.bias
        return x * coef, bias

class PosEnc(nn.Module):
    def __init__(self, d, T=512):
        super().__init__()
        pe = torch.zeros(T, d); pos = torch.arange(T).unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2) * (-np.log(10000.0) / d))
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer("pe", pe)
    def forward(self, x): return x + self.pe[: x.shape[1]]

class EncLayer(nn.Module):
    def __init__(self, d, heads, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.ReLU(), nn.Linear(4 * d, d))
        self.n1, self.n2, self.drop = nn.LayerNorm(d), nn.LayerNorm(d), nn.Dropout(dropout)
        self.last_w = None
    def forward(self, x, keep=False):
        a, w = self.attn(x, x, x, need_weights=keep, average_attn_weights=True)
        if keep: self.last_w = w                                  # (n, T, T)
        x = self.n1(x + self.drop(a)); return self.n2(x + self.drop(self.ff(x)))

class TransformerNet(nn.Module):
    def __init__(self, F, d_model, heads, layers, dropout):
        super().__init__()
        self.inp, self.pos = nn.Linear(F, d_model), PosEnc(d_model)
        self.layers = nn.ModuleList([EncLayer(d_model, heads, dropout) for _ in range(layers)])
        self.head = nn.Linear(d_model, 1)
    def forward(self, x, keep=False):
        z = self.pos(self.inp(x))
        for L in self.layers: z = L(z, keep)
        return self.head(z[:, -1]).squeeze(-1)
    def attention_score(self, x):
        """Section 4.2.2: attention of the last position on each step x |input|, summed over time (unsigned)."""
        self.forward(x, keep=True)
        w = self.layers[-1].last_w[:, -1, :]                      # (n, T)
        return (w.unsqueeze(-1) * x.abs()).sum(1)

class GLU(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__(); self.l = nn.Linear(d_in, 2 * d_out)
    def forward(self, x):
        a, b = self.l(x).chunk(2, dim=-1); return a * torch.sigmoid(b)

class GRN(nn.Module):
    """Gated residual network of the TFT."""
    def __init__(self, d_in, d_hid, d_out, dropout):
        super().__init__()
        self.skip = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()
        self.l1, self.l2 = nn.Linear(d_in, d_hid), nn.Linear(d_hid, d_hid)
        self.drop, self.glu, self.norm = nn.Dropout(dropout), GLU(d_hid, d_out), nn.LayerNorm(d_out)
    def forward(self, a):
        h = self.drop(self.l2(nn.functional.elu(self.l1(a))))
        return self.norm(self.skip(a) + self.glu(h))

class InterpretableMHA(nn.Module):
    """TFT interpretable multi-head attention: per-head queries/keys, shared values, heads averaged."""
    def __init__(self, d, heads, dropout):
        super().__init__()
        self.dk = d // heads
        self.q = nn.ModuleList([nn.Linear(d, self.dk) for _ in range(heads)])
        self.k = nn.ModuleList([nn.Linear(d, self.dk) for _ in range(heads)])
        self.v, self.o, self.drop = nn.Linear(d, self.dk), nn.Linear(self.dk, d), nn.Dropout(dropout)
    def forward(self, q, k):
        v, outs, atts = self.v(k), [], []
        for qi, ki in zip(self.q, self.k):
            a = torch.softmax(qi(q) @ ki(k).transpose(1, 2) / self.dk ** 0.5, dim=-1)
            atts.append(a); outs.append(self.drop(a) @ v)
        return self.o(torch.stack(outs).mean(0)), torch.stack(atts).mean(0)

class TFTNet(nn.Module):
    """Temporal Fusion Transformer (Lim et al., 2021), single-horizon version for one target hour:
    variable selection network -> LSTM encoder with gated skip -> enrichment GRN ->
    interpretable attention from the last step -> gated position-wise GRN -> output.
    Simplifications: no static covariates; one GRN shared by all variables (the original uses one per variable)."""
    def __init__(self, F, d_model, heads, dropout):
        super().__init__()
        d = d_model
        self.emb_w, self.emb_b = nn.Parameter(torch.randn(F, d) * 0.1), nn.Parameter(torch.zeros(F, d))
        self.vsn, self.var_grn = GRN(F * d, d, F, dropout), GRN(d, d, d, dropout)
        self.lstm = nn.LSTM(d, d, batch_first=True)
        self.gate1, self.norm1 = GLU(d, d), nn.LayerNorm(d)
        self.enrich = GRN(d, d, d, dropout)
        self.attn = InterpretableMHA(d, heads, dropout)
        self.gate2, self.norm2 = GLU(d, d), nn.LayerNorm(d)
        self.ff, self.gate3, self.norm3 = GRN(d, d, d, dropout), GLU(d, d), nn.LayerNorm(d)
        self.out = nn.Linear(d, 1)
        self.last_w = None
    def select(self, x):
        e = x.unsqueeze(-1) * self.emb_w + self.emb_b                         # (n, T, F, d)
        n, T, F_, d = e.shape
        w = torch.softmax(self.vsn(e.reshape(n, T, F_ * d)), dim=-1)           # (n, T, F) selection weights
        return (w.unsqueeze(-1) * self.var_grn(e)).sum(2), w
    def forward(self, x, keep=False):
        z, w = self.select(x)
        h, _ = self.lstm(z)
        h = self.enrich(self.norm1(z + self.gate1(h)))
        last = h[:, -1:]
        a, _ = self.attn(last, h)
        y = self.norm2(last + self.gate2(a))
        y = self.norm3(y + self.gate3(self.ff(y)))
        if keep: self.last_w = w
        return self.out(y[:, 0]).squeeze(-1)
    def selection_weights(self, x):
        """TFT's built-in importance: variable-selection weights averaged over the window (unsigned, sum to 1)."""
        self.forward(x, keep=True); return self.last_w.mean(1)

def build_net(kind, F):
    a = ARCH[kind]
    if kind == "LSTM": return LSTMNet(F, a["hidden"], a["layers"], a["dropout"])
    if kind == "RETAIN": return RETAINNet(F, a["hidden"], a["dropout"])
    if kind == "TFT": return TFTNet(F, a["d_model"], a["heads"], a["dropout"])
    return TransformerNet(F, a["d_model"], a["heads"], a["layers"], a["dropout"])

class Out2D(nn.Module):
    def __init__(self, net): super().__init__(); self.net = net
    def forward(self, x): return self.net(x).unsqueeze(-1)

class SeqModel:
    """Train / predict / explain one network on standardized windows (n, T, F)."""
    def __init__(self, kind, F, seed):
        torch.manual_seed(seed); np.random.seed(seed)
        self.kind, self.net = kind, build_net(kind, F).to(DEVICE)
    def _t(self, a): return torch.as_tensor(a, dtype=torch.float32, device=DEVICE)
    def fit(self, Xtr, ytr, Xva, yva):
        opt = torch.optim.AdamW(self.net.parameters(), lr=TRAIN["lr"], weight_decay=TRAIN["weight_decay"])
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)
        loss_fn = nn.HuberLoss(); best, best_state, bad = np.inf, None, 0
        Xtr_t, ytr_t = self._t(Xtr), self._t(ytr); n = len(Xtr)
        self.epochs = 0
        for ep in range(TRAIN["max_epochs"]):
            self.net.train(); perm = torch.randperm(n, device=DEVICE)
            for i in range(0, n, TRAIN["batch"]):
                idx = perm[i:i + TRAIN["batch"]]
                opt.zero_grad(); loss = loss_fn(self.net(Xtr_t[idx]), ytr_t[idx]); loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), TRAIN["clip"]); opt.step()
            val = float(np.mean(np.abs(self.predict(Xva) - yva)))
            sched.step(val); self.epochs = ep + 1
            if val < best - 1e-5:
                best, bad = val, 0; best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
            else:
                bad += 1
                if bad >= TRAIN["patience"]: break
        self.net.load_state_dict(best_state); self.net.eval(); return self
    @torch.no_grad()
    def predict(self, Xw, bs=2048):
        self.net.eval()
        return np.concatenate([self.net(self._t(Xw[i:i + bs])).cpu().numpy() for i in range(0, len(Xw), bs)])
    def shap(self, Xw, background, seed=0):
        self.net.eval()
        wrapped = Out2D(self.net).eval()                          # shap expects (n, 1) outputs
        with torch.backends.cudnn.flags(enabled=False):          # RNN backward in eval mode on GPU
            ex = shap.GradientExplainer(wrapped, self._t(background))
            sv = ex.shap_values(self._t(Xw), nsamples=SHAP_NSAMPLES, rseed=seed)
        sv = np.asarray(sv[0] if isinstance(sv, list) else sv)
        return sv.reshape(len(Xw), Xw.shape[1], Xw.shape[2])     # (n, T, F)
    @torch.no_grad()
    def intrinsic(self, Xw):
        self.net.eval(); x = self._t(Xw)
        if self.kind == "RETAIN":
            c, bias = self.net.contributions(x)
            err = (c.sum((1, 2)) + bias - self.net(x)).abs().max().item()
            assert err < 1e-3, f"RETAIN identity violated ({err:.2e})"
            return c.cpu().numpy(), err
        if self.kind == "Transformer":
            return self.net.attention_score(x).cpu().numpy(), None
        if self.kind == "TFT":
            return self.net.selection_weights(x).cpu().numpy(), None
        return None, None

# %%
def windows(Xs, idx, T=WINDOW):
    return np.stack([Xs[i - T + 1:i + 1] for i in idx]).astype(np.float32)

def seq_deletion_insertion(model, Xw, A, K=10):
    n, T, F = Xw.shape; order = np.argsort(-np.abs(A), axis=1)
    f0 = model.predict(Xw); scale = max(np.std(f0), 1e-9); dele, ins = [], []
    for m in range(1, min(K, F) + 1):
        M = Xw.copy()
        for i in range(n): M[i, :, order[i, :m]] = 0.0
        dele.append(np.mean(np.abs(f0 - model.predict(M))))
        M = np.zeros_like(Xw)
        for i in range(n): M[i, :, order[i, :m]] = Xw[i, :, order[i, :m]]
        ins.append(np.mean(np.abs(model.predict(M) - f0)))
    return float(np.mean(dele) / scale), float(1 - np.mean(ins) / scale)

def pair_agreement(a, b, k=5):
    """Agreement of two explanations of the same forecasts (Section 4.3)."""
    out = []
    for x, y in zip(a, b):
        ta, tb = np.argsort(-np.abs(x))[:k], np.argsort(-np.abs(y))[:k]
        out.append((stats.spearmanr(np.abs(x), np.abs(y)).statistic, stats.weightedtau(np.abs(x), np.abs(y)).statistic,
                    len(set(ta) & set(tb)) / len(set(ta) | set(tb)), np.mean(np.sign(x[tb]) == np.sign(y[tb]))))
    return np.nanmean(np.array(out), axis=0)

# %% [markdown]
# ## 3. Estimators and quality indicators
#
# Grouped Shapley value, LIME and KernelSHAP on the same samples, max-sensitivity, faithfulness correlation, infidelity and global aggregation rules. Definitions are in the docstrings and in the mathematical note.

# %%
# ======================= Estimators and quality indicators =======================
# Every function works for both model types. An instance is either a window (T, F) for the TFT or a row (F,) for
# XGBoost. A "player" is a list of column indices on the last axis: for the TFT one column is a feature's whole
# 24-hour series, so masking a player replaces that feature at every lag. f maps a batch of instances to MW.

def coalition_values(f, x, Bset, players, Z):
    """v(z) = mean over the background set of f(instance with the players in z taken from x and the rest from the
    background). This is the interventional value function of SHAP. Z is (n_coalitions, P) with 0/1 entries."""
    nZ, nB = len(Z), len(Bset)
    inst = np.repeat(Bset[None], nZ, axis=0).reshape((nZ * nB,) + Bset.shape[1:]).copy()
    Zr = np.repeat(Z, nB, axis=0).astype(bool)
    for j, cols in enumerate(players):
        on = Zr[:, j]
        if on.any():
            sub = inst[on]; sub[..., cols] = x[..., cols]; inst[on] = sub
    return f(inst).reshape(nZ, nB).mean(1)

def grouped_shapley(f, X, BG, players, M, seed=0, n_bg=GROUPED_BG):
    """Interventional Shapley value with the given players, estimated by sampling M orderings and their reverses
    (antithetic) and averaging every coalition over n_bg background instances. With players = features, each player
    is a feature's whole 24-hour series. The values add up exactly to f(x) minus the mean forecast on the background
    set. Returns (n, P) in MW."""
    rng = np.random.default_rng(seed); P = len(players); out = np.zeros((len(X), P))
    Bset = BG[rng.choice(len(BG), min(n_bg, len(BG)), replace=False)]      # one fixed set: any reference shift is the same for all forecasts
    for i, x in enumerate(X):
        perms = [rng.permutation(P) for _ in range(M)]; perms += [p[::-1] for p in perms]
        Z = np.zeros((len(perms) * (P + 1), P))
        for k, p in enumerate(perms):
            for j in range(1, P + 1): Z[k * (P + 1) + j, p[:j]] = 1
        v = coalition_values(f, x, Bset, players, Z).reshape(len(perms), P + 1)
        for k, p in enumerate(perms): out[i, p] += np.diff(v[k])
        out[i] /= len(perms)
    return out

def lime_kernelshap(f, X, BG, players, S, seed=0, n_bg=LIME_BG):
    """LIME and KernelSHAP fitted on the same sampled coalitions. A coalition z switches players off (background
    values), with the number switched off uniform on 1..P as in LIME's own sampler, and each coalition is averaged
    over n_bg background instances. LIME: default kernel of the lime package, sqrt(exp(-d^2 / w^2)) with d the
    Euclidean distance of z from the instance and w = 0.75 sqrt(P), and a weighted ridge regression with alpha = 1.
    KernelSHAP: Shapley kernel corrected for this sampling, proportional to 1 / (|z| (P - |z|)), with the empty and
    full coalitions forced, and unpenalised least squares. Coefficients are contributions in MW of each player being
    present. Returns two (n, P) arrays."""
    rng = np.random.default_rng(seed); P = len(players)
    L, K = np.zeros((len(X), P)), np.zeros((len(X), P)); width = 0.75 * np.sqrt(P)
    Bset = BG[rng.choice(len(BG), min(n_bg, len(BG)), replace=False)]
    for i, x in enumerate(X):
        Z = np.ones((S + 1, P)); Z[-1] = 0
        for s in range(1, S):
            Z[s, rng.choice(P, rng.integers(1, P + 1), replace=False)] = 0
        y = coalition_values(f, x, Bset, players, Z)
        d = np.sqrt(P - Z.sum(1)); wl = np.sqrt(np.exp(-d ** 2 / width ** 2))
        L[i] = Ridge(alpha=1.0).fit(Z[:-1], y[:-1], sample_weight=wl[:-1]).coef_
        size = Z.sum(1); wk = np.where((size == 0) | (size == P), 1e6, 1.0 / np.maximum(size * (P - size), 1))
        K[i] = LinearRegression().fit(Z, y, sample_weight=wk).coef_
    return L, K

def to_feature_matrix(A_players, players, F):
    """Spread player values onto features (all of a player's value on its first column), so the paper's scoring,
    which sums features within each driver, can be applied unchanged when each player sits inside one driver."""
    A = np.zeros((len(A_players), F))
    for j, cols in enumerate(players): A[:, cols[0]] += A_players[:, j]
    return A

def max_sensitivity(explain, X, cols, scale, K, r, seed=0):
    """Yeh et al. (2019): largest relative change of the explanation under small input perturbations (uniform in
    +-r standard deviations, applied only to measured inputs: weather and demand history). Median over forecasts."""
    rng = np.random.default_rng(seed); e0 = explain(X); out = []
    for k in range(K):
        Xp = X.copy(); d = rng.uniform(-r, r, size=Xp[..., cols].shape) * scale[cols]
        Xp[..., cols] = Xp[..., cols] + d
        ek = explain(Xp)
        out.append(np.linalg.norm(ek - e0, axis=1) / np.maximum(np.linalg.norm(e0, axis=1), 1e-9))
    return float(np.median(np.max(np.stack(out), axis=0)))

def faithfulness(f, X, A, baseline, frac, K, seed=0):
    """Remove a random subset S of features (set to the baseline, the training mean) and compare the attribution of
    S with the change in the forecast. Faithfulness correlation (Bhatt et al., 2020): correlation over subsets,
    averaged over forecasts. Relative infidelity (Yeh et al., 2019): mean squared difference divided by the mean
    squared change, so 0 is perfect and 1 is no better than an explanation of zeros. Correlation ignores the size of
    the attributions; infidelity does not."""
    rng = np.random.default_rng(seed); F = A.shape[1]; s = max(1, int(round(frac * F)))
    f0 = f(X); corr, num, den = [], 0.0, 0.0
    subsets = [rng.choice(F, s, replace=False) for _ in range(K)]
    D = np.zeros((len(X), K)); Sa = np.zeros((len(X), K))
    for k, S_ in enumerate(subsets):
        Xm = X.copy(); Xm[..., S_] = baseline[..., S_]
        D[:, k] = f0 - f(Xm); Sa[:, k] = A[:, S_].sum(1)
    for i in range(len(X)):
        if np.std(D[i]) > 1e-9 and np.std(Sa[i]) > 1e-9: corr.append(np.corrcoef(Sa[i], D[i])[0, 1])
    return float(np.mean(corr)) if corr else np.nan, float(np.mean((Sa - D) ** 2) / max(np.mean(D ** 2), 1e-12))

def global_scores(A, feats, truth_s, truth_bg):
    """Part B: global importance of each driver over many forecasts with four aggregation rules, against the truth.
    Players scored: the four drivers (true importance > 0), consequences of demand and irrelevant inputs
    (true importance 0). Demand history is left out because its true importance is not defined by the generator."""
    Ad = pd.DataFrame(A, columns=feats); out = {}
    G = {g: Ad[[c for c in cols if c in Ad]].sum(1).values for g, cols in B["GROUPS"].items()}
    G["consequence"] = Ad[[c for c in feats if B["role"](c) == "consequence"]].abs().sum(1).values
    G["irrelevant"] = Ad[[c for c in feats if B["role"](c) == "irrelevant"]].abs().sum(1).values
    T = {g: truth_s[c].values - truth_bg[c].mean() for g, c in B["TRUTH_COL"].items()}
    T["consequence"] = T["irrelevant"] = np.zeros(len(truth_s))
    names = list(G); Gm = np.column_stack([G[g] for g in names]); Tm = np.column_stack([T[g] for g in names])
    ranks = stats.rankdata(-np.abs(Gm), axis=1)
    rules = {"mean_signed": (Gm.mean(0), Tm.mean(0)), "mean_abs": (np.abs(Gm).mean(0), np.abs(Tm).mean(0)),
             "rms": (np.sqrt((Gm ** 2).mean(0)), np.sqrt((Tm ** 2).mean(0))), "mean_rank": (-ranks.mean(0), np.abs(Tm).mean(0))}
    for rule, (est, tru) in rules.items():
        out[f"tau_{rule}"] = stats.kendalltau(est, np.abs(tru)).statistic
        if rule != "mean_rank":
            ratio = [est[k] / tru[k] for k, g in enumerate(names) if g in DRIVERS and abs(tru[k]) > 1e-9]
            out[f"size_error_{rule}"] = float(np.mean(np.abs(1 - np.array(ratio)))) if ratio else np.nan
    return out

# %% [markdown]
# ## 4. Runs (resumable)

# %%
WEATHER_RAW = ["temp", "humidity", "pressure", "wind_speed", "clouds_all"]

def interventional_xgb(mx, BG, feats, f):
    """Interventional SHAP for XGBoost. First try TreeSHAP (exact, fast). Some shap and xgboost version pairs refuse it
    ("Categorical split is not yet supported") although the model has no categorical features. Then the same quantity
    is estimated with shap's model-agnostic permutation explainer and an independent masker over the same background,
    which is the interventional value function by definition. Returns (function rows -> (n, F) in MW, route name)."""
    try:
        ex = shap.TreeExplainer(mx.m, data=BG, feature_perturbation="interventional")
        fn = lambda R: np.asarray(ex.shap_values(pd.DataFrame(np.asarray(R), columns=feats)))
        fn(BG[:2])
        return fn, "TreeSHAP"
    except Exception as e:
        print(f"interventional TreeSHAP not available ({type(e).__name__}); using the permutation estimator")
        pe = shap.explainers.Permutation(lambda R: f(np.asarray(R)), shap.maskers.Independent(BG, max_samples=100))
        return (lambda R: np.asarray(pe(np.asarray(R), max_evals=10 * len(feats), silent=True).values)), "permutation estimator"

def run(cname, seed, kind):
    cond = dict(name=cname, change="2016-07-01", latent=CONDITIONS[cname] * LAT_SCALE, interaction=0.0)
    energy, weather, truth, hols = B["generate"](cond, seed, g=GEN_USE)
    X, y = B["build_features"](energy, weather, hols); feats = list(X.columns); F = len(feats); B["FEATS_CUR"] = feats
    tr, va, te = B["split_idx"](len(X)); t0 = time.time()
    rng = np.random.default_rng(seed + 99)                                    # same explained hours as the paper
    sample = np.sort(rng.choice(np.arange(te.start, te.stop), min(N_EXPLAIN, te.stop - te.start), replace=False))
    sub = np.sort(np.random.default_rng(seed + 7).choice(len(sample), min(N_GROUPED, len(sample)), replace=False))
    truth_bg = truth.reindex(X.index[tr])
    players_feat = [[j] for j in range(F)]
    in_driver = {f_: g for g, cols in B["GROUPS"].items() for f_ in cols}
    players_drv = [[feats.index(c) for c in cols if c in feats] for cols in B["GROUPS"].values()] + \
                  [[j] for j, f_ in enumerate(feats) if f_ not in in_driver]
    meas = [j for j, f_ in enumerate(feats) if f_.startswith("load_") or f_ in WEATHER_RAW]
    A_full, A_sub, info = {}, {}, {}

    if kind == "TFT":
        mu, sd = X.iloc[tr].mean().values, X.iloc[tr].std().values + 1e-9
        ym, ysd = y.iloc[tr].mean(), y.iloc[tr].std()
        Xs = ((X.values - mu) / sd).astype(np.float32); ys = ((y.values - ym) / ysd).astype(np.float32)
        ARCH["TFT"] = dict(ARCH["TFT"]); TRAIN["lr"] = 1e-3
        i_tr = np.arange(max(tr.start, WINDOW - 1), tr.stop); i_va = np.arange(va.start, va.stop); i_te = np.arange(te.start, te.stop)
        Wtr, Wva, Wte, Wex = windows(Xs, i_tr), windows(Xs, i_va), windows(Xs, i_te), windows(Xs, sample)
        BG = Wtr[np.random.default_rng(seed).choice(len(Wtr), SHAP_BACKGROUND, replace=False)]
        m = SeqModel("TFT", F, seed).fit(Wtr, ys[i_tr], Wva, ys[i_va])
        f = lambda W: m.predict(W.astype(np.float32)) * ysd + ym
        mae = float(np.mean(np.abs(f(Wte) - y.values[i_te])))
        cell = m.shap(Wex, BG, seed) * ysd                                        # (n, T, F): one value per lag and feature
        A_full["TFT, SHAP sum over lags (paper)"] = cell.sum(1)
        A_full["TFT, SHAP last lag only"] = cell[:, -1, :]
        A_full["TFT, SHAP mean over lags"] = cell.mean(1)
        info["lag_profile"] = (np.abs(cell).mean(0) / np.abs(cell).mean(0).sum(0, keepdims=True)).mean(1).tolist()   # share of credit by lag
        Wsub = Wex[sub]
        A_sub["TFT, grouped Shapley, players = features"] = grouped_shapley(f, Wsub, BG, players_feat, PERM_SAMPLES, seed)
        A_sub["TFT, grouped Shapley, players = drivers"] = to_feature_matrix(grouped_shapley(f, Wsub, BG, players_drv, PERM_SAMPLES, seed), players_drv, F)
        L, K = lime_kernelshap(f, Wsub, BG, players_feat, LIME_SAMPLES, seed)
        A_sub["TFT, LIME"], A_sub["TFT, KernelSHAP (same samples)"] = L, K
        base_name, base_full, Xex_sub, baseline = "TFT, SHAP sum over lags (paper)", A_full["TFT, SHAP sum over lags (paper)"], Wsub, np.zeros(Wex.shape[1:], np.float32)
        scale = np.ones(F)
        explainers_sens = {"TFT, SHAP sum over lags (paper)": lambda W: m.shap(W.astype(np.float32), BG, seed).sum(1) * ysd,
                           "TFT, LIME": lambda W: lime_kernelshap(f, W, BG, players_feat, LIME_SAMPLES, seed)[0]}
        deletion = lambda A_: seq_deletion_insertion(m, Xex_sub[:N_FAITH], A_[:N_FAITH])[0]
        if RETRAIN_STABILITY:
            m2 = SeqModel("TFT", F, seed + 100).fit(Wtr, ys[i_tr], Wva, ys[i_va])
            A2 = m2.shap(Wsub, BG, seed).sum(1) * ysd
            info["retrain_stability"] = {"TFT, SHAP sum over lags (paper)": float(np.mean(
                [stats.spearmanr(np.abs(a), np.abs(b)).statistic for a, b in zip(A_full[base_name][sub], A2)]))}
    else:
        mx = B["XGBTreeSHAP"]().fit(X.iloc[tr], y.iloc[tr], X.iloc[va], y.iloc[va])
        f = lambda R: mx.predict(pd.DataFrame(R, columns=feats))
        mae = float(np.mean(np.abs(f(X.values[te]) - y.values[te])))
        BG = X.values[tr][np.random.default_rng(seed).choice(tr.stop - tr.start, SHAP_BACKGROUND, replace=False)]
        Xex = X.values[sample]
        A_full["XGBoost, TreeSHAP path-dependent (paper)"] = mx.explain(X.iloc[sample])
        Xsub = Xex[sub]
        int_fn, int_route = interventional_xgb(mx, BG, feats, f)
        if int_route == "TreeSHAP":            # fast exact route: all explained forecasts
            A_full["XGBoost, TreeSHAP interventional"] = int_fn(Xex)
        else:                                  # permutation estimator: the subset only, to keep the run time down
            A_sub["XGBoost, TreeSHAP interventional"] = int_fn(Xsub)
        info["interventional_route"] = int_route
        A_sub["XGBoost, grouped Shapley, players = drivers"] = to_feature_matrix(grouped_shapley(f, Xsub, BG, players_drv, PERM_SAMPLES, seed), players_drv, F)
        L, K = lime_kernelshap(f, Xsub, BG, players_feat, LIME_SAMPLES, seed)
        A_sub["XGBoost, LIME"], A_sub["XGBoost, KernelSHAP (same samples)"] = L, K
        base_name, base_full, Xex_sub, baseline = "XGBoost, TreeSHAP path-dependent (paper)", A_full["XGBoost, TreeSHAP path-dependent (paper)"], Xsub, X.values[tr].mean(0)
        scale = X.values[tr].std(0) + 1e-9
        explainers_sens = {"XGBoost, TreeSHAP path-dependent (paper)": lambda R: mx.explain(pd.DataFrame(R, columns=feats)),
                           "XGBoost, TreeSHAP interventional": int_fn,
                           "XGBoost, LIME": lambda R: lime_kernelshap(f, R, BG, players_feat, LIME_SAMPLES, seed)[0]}
        mu_x = X.values[tr].mean(0)
        deletion = lambda A_: B["deletion_insertion"](mx, Xex_sub[:N_FAITH], None, A_[:N_FAITH], mu_x)[0]
        try:                                                                       # interactions between features
            Iv = np.asarray(shap.TreeExplainer(mx.m).shap_interaction_values(X.iloc[sample[sub[:N_INTERACT]]]))
            gid = np.array([list(B["GROUPS"]).index(in_driver[f_]) if f_ in in_driver else 100 + j for j, f_ in enumerate(feats)])
            cross = gid[:, None] != gid[None, :]
            info["interaction_share_between_features"] = float(np.abs(Iv * cross).sum() / np.abs(Iv).sum())
        except Exception as e:
            info["interaction_share_between_features"] = np.nan; print("interaction values skipped:", e)
        if RETRAIN_STABILITY:
            mx2 = B["XGBTreeSHAP"](); p_ = dict(n_estimators=2000, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                                              min_child_weight=5, early_stopping_rounds=50, n_jobs=-1, verbosity=0, random_state=seed + 100)
            if B["FAST_MODE"]: p_.update(n_estimators=300, early_stopping_rounds=20)
            mx2.m = xgb.XGBRegressor(**p_).fit(X.iloc[tr], y.iloc[tr], eval_set=[(X.iloc[va], y.iloc[va])], verbose=False)
            mx2.ex = shap.TreeExplainer(mx2.m); A2 = mx2.explain(X.iloc[sample[sub]])
            info["retrain_stability"] = {base_name: float(np.mean(
                [stats.spearmanr(np.abs(a), np.abs(b)).statistic for a, b in zip(A_full[base_name][sub], A2)]))}

    # ---- scoring against the truth ----
    for k_, v in A_full.items(): A_sub[k_] = v[sub]
    ctl = B["controls"](base_full[sub], np.random.default_rng(seed), feats)
    ctl = {f"control: {k_.split(': ')[1]}": v for k_, v in ctl.items()}
    truth_full, truth_sub = truth.reindex(X.index[sample]), truth.reindex(X.index[sample[sub]])
    sens = {e: max_sensitivity(fn, Xex_sub[:N_ROBUST], meas, scale, ROBUST_K, ROBUST_R, seed) for e, fn in explainers_sens.items()}
    for e, A in {**A_sub, **ctl}.items():
        row = dict(condition=cname, seed=seed, model=kind, explainer=e, is_control=e.startswith("control"), MAE=mae,
                   n_scored=len(A))
        sc = B["gt_scores"](A, feats, truth_sub); row.update({k_: float(v) for k_, v in sc.items() if np.isscalar(v)})
        if e in A_full:
            scf = B["gt_scores"](A_full[e], feats, truth_full)
            row.update({f"full_{k_}": float(scf[k_]) for k_ in ["magnitude_correctness", "pattern_correctness"] + [f"b_{g}" for g in DRIVERS]})
            row.update(global_scores(A_full[e], feats, truth_full, truth_bg))
        else:
            row.update(global_scores(A, feats, truth_sub, truth_bg))
        row["faithfulness_correlation"], row["relative_infidelity"] = faithfulness(f, Xex_sub[:N_FAITH], A[:N_FAITH], baseline, FAITH_FRAC, FAITH_K, seed)
        row["deletion"] = deletion(A)
        row["max_sensitivity"] = sens.get(e, np.nan)
        row["retrain_stability"] = info.get("retrain_stability", {}).get(e, np.nan)
        if e.endswith("players = features") or e.endswith("players = drivers"):
            ref = A_sub[base_name]; G1 = pd.DataFrame(ref, columns=feats); G2 = pd.DataFrame(A, columns=feats)
            gaps = [np.mean(np.abs((G1[cols].sum(1) - G1[cols].sum(1).mean()) - (G2[cols].sum(1) - G2[cols].sum(1).mean())))
                    / max(np.mean(np.abs(G2[cols].sum(1) - G2[cols].sum(1).mean())), 1e-9)
                    for cols in B["GROUPS"].values()]
            row["relative_gap_to_paper_explanation"] = float(np.mean(gaps))
        log(row)
    np.savez_compressed(RUN_DIR / "attributions" / f"{cname}_{kind}_seed{seed}.npz", feats=np.array(feats), sample=sample, sub=sub,
                        **{k_: v.astype(np.float32) for k_, v in A_sub.items()})
    log(dict(stage="run_info", condition=cname, seed=seed, model=kind, MAE=mae, seconds=round(time.time() - t0),
             **{k_: v for k_, v in info.items() if k_ != "retrain_stability"}))
    print(f"{cname} seed {seed} {kind:8s}: MAE {mae:.0f} MW, paper explanation correctness "
          f"{B['gt_scores'](base_full, feats, truth_full)['magnitude_correctness']:.2f}, {time.time() - t0:.0f}s")

t_all = time.time()
for cname in CONDITIONS:
    for seed in SEEDS:
        for kind in MODELS:
            if is_done(stage="run_info", condition=cname, seed=seed, model=kind): continue
            run(cname, seed, kind)
print(f"finished in {time.time() - t_all:.0f}s")

# %% [markdown]
# ## 5. Results

# %%
PACK = OUT_DIR / f"extensions_results_pack_{RUN_PLAN}"; PACK.mkdir(parents=True, exist_ok=True)
MDL = [f"# Extensions results ({pd.Timestamp.now():%Y-%m-%d %H:%M}), plan '{RUN_PLAN}', generator '{GENERATOR}'", ""]
def add(df, title, fname, note=None):
    df.to_csv(PACK / fname)
    try: txt = df.to_markdown(floatfmt=".3f")
    except Exception: txt = df.to_string()
    MDL.extend([f"## {title}", f"(file: {fname})", ""] + ([note, ""] if note else []) + [txt, ""]); show(df)
LG = pd.DataFrame(logged()); R = LG[LG.get("stage").isna()] if "stage" in LG else LG
INFO = LG[LG.get("stage") == "run_info"] if "stage" in LG else pd.DataFrame()
R = R[R.condition == "default"] if "default" in set(R.condition) else R
EXPL = [e for e in R.explainer.unique() if not e.startswith("control")]
bcols = [f"b_{g}" for g in DRIVERS]

# ---- A. Aggregation over the input window (same forecasts for every explainer) ----
cols = ["magnitude_correctness", "pattern_correctness"] + bcols + ["relative_gap_to_paper_explanation"]
tabA = R[R.explainer.isin(EXPL)].groupby("explainer")[[c for c in cols if c in R]].agg(["mean", "std"]).round(3)
add(tabA, "A1. Per forecast: sum over lags, last lag, mean over lags, grouped Shapley value, LIME, KernelSHAP (scored on the same forecasts)",
    "A1_window_aggregation.csv", "b = 1: full true effect. relative_gap: mean |sum of lags - grouped Shapley| per driver, relative to the grouped value. "
    "The mean over lags is the sum divided by 24, so its b is 1/24 of the sum's.")
full = R[R.explainer.isin(EXPL)].dropna(subset=["full_magnitude_correctness"]) if "full_magnitude_correctness" in R else pd.DataFrame()
if len(full):
    add(full.groupby("explainer")[["full_magnitude_correctness", "full_pattern_correctness"] + [f"full_{c}" for c in bcols]].agg(["mean", "std"]).round(3),
        "A2. The same on all explained forecasts (paper sample)", "A2_full_sample.csv")
lp = INFO.dropna(subset=["lag_profile"]) if "lag_profile" in INFO else pd.DataFrame()
if len(lp):
    prof = np.mean(np.stack(lp.lag_profile.values), axis=0)
    fig, ax = plt.subplots(figsize=(6, 3)); ax.bar(np.arange(-len(prof) + 1, 1), prof, color="#2F6690")
    ax.set(xlabel="lag (hours before the target hour; 0 = target hour)", ylabel="share of |SHAP|", title="Where SHAP puts credit within the 24-hour window (TFT)")
    fig.tight_layout(); fig.savefig(PACK / "fig_A_lag_profile.png"); plt.close(fig)
    add(pd.DataFrame({"lag": np.arange(-len(prof) + 1, 1), "share_of_abs_shap": prof}).set_index("lag").round(4),
        "A3. Share of the TFT's SHAP credit at each lag", "A3_lag_profile.csv",
        "If most credit sat at lag 0, the last lag alone would be enough. Spread credit is why the sum over lags is needed.")

# ---- B. Aggregation over many forecasts ----
gcols = [c for c in R.columns if c.startswith("tau_") or c.startswith("size_error_")]
add(R[R.explainer.isin(EXPL)].groupby("explainer")[gcols].mean().round(3),
    "B1. Global importance over the explained forecasts: four aggregation rules against the true global importance", "B1_global_aggregation.csv",
    "tau: Kendall correlation between the global ranking and the true ranking (drivers, consequences, irrelevant inputs). "
    "size_error: mean |1 - estimated / true| over the four drivers (0 is perfect). The mean rank has no size.")

# ---- C/D. Quality indicators and whether they track the truth ----
ind = ["faithfulness_correlation", "relative_infidelity", "deletion", "max_sensitivity", "retrain_stability"]
add(R.groupby("explainer")[["magnitude_correctness"] + [c for c in ind if c in R]].mean().round(3),
    "D1. Quality indicators for every explainer and for the controls", "D1_indicators.csv",
    "max_sensitivity: lower is more robust. relative_infidelity: lower is more faithful (0 perfect, 1 = zero explanation). "
    "faithfulness_correlation, deletion, retrain_stability: higher is better.")
rows = []
for (c_, s_, m_), g in R.groupby(["condition", "seed", "model"]):
    for i_ in ["faithfulness_correlation", "relative_infidelity", "deletion"]:
        gg = g.dropna(subset=[i_, "magnitude_correctness"])
        if len(gg) >= 4:
            sign = -1 if i_ == "relative_infidelity" else 1
            rows.append(dict(model=m_, seed=s_, indicator=i_, spearman_with_correctness=sign * stats.spearmanr(gg[i_], gg.magnitude_correctness).statistic))
    for nm, (lo, hi) in {"halved vs inflated copy": ("control: shrunk x0.5", "control: inflated x1.5")}.items():
        a_, b_ = g[g.explainer == lo], g[g.explainer == hi]
        for i_ in ["faithfulness_correlation", "relative_infidelity", "deletion"]:
            if len(a_) and len(b_):
                rows.append(dict(model=m_, seed=s_, indicator=i_, separates_halved_from_inflated=float(abs(a_[i_].values[0] - b_[i_].values[0]) > 1e-3)))
MT = pd.DataFrame(rows)
if len(MT):
    add(MT.groupby(["model", "indicator"]).mean(numeric_only=True).drop(columns="seed").round(3),
        "D2. Does each indicator track the true correctness? (rank correlation over explainers and controls, oriented so that higher = agrees)",
        "D2_indicator_vs_truth.csv", "separates_halved_from_inflated: share of runs in which the indicator gives the halved and the inflated copies "
        "different values. Their true correctness differs, so an indicator that cannot separate them is blind to size.")
rb = R[R.explainer.isin(EXPL)][["model", "explainer", "magnitude_correctness", "max_sensitivity", "retrain_stability"]].groupby(["model", "explainer"]).mean().dropna(how="all", subset=["max_sensitivity", "retrain_stability"])
if len(rb): add(rb.round(3), "D3. Robustness against correctness", "D3_robustness_vs_correctness.csv",
                "A robust explanation (low sensitivity, high retraining stability) is not necessarily a correct one.")

# ---- E. Correlated inputs ----
ecols = ["magnitude_correctness", "share_direct cause", "share_demand history", "share_consequence", "share_irrelevant"]
add(R[R.explainer.isin(EXPL)].groupby("explainer")[ecols].mean().round(3), "E1. Where the credit goes under each estimator", "E1_credit_shares.csv",
    "Consequences (gas, coal) are correlated with demand but have no effect. Path-dependent TreeSHAP follows the data's correlations, "
    "interventional TreeSHAP follows the model.")
if "interaction_share_between_features" in INFO:
    add(INFO.groupby("model")[["interaction_share_between_features"]].agg(["mean", "std"]).round(3),
        "E2. Share of XGBoost's TreeSHAP interaction values that lies between different features or drivers", "E2_interactions.csv",
        "This is the term that makes the sum of attributions differ from the grouped Shapley value (mathematical note, Proposition 2).")

# ---- figure: aggregation choices ----
show_e = [e for e in EXPL if not e.startswith("XGBoost, TreeSHAP interventional")]
mc = R[R.explainer.isin(show_e)].groupby("explainer")["magnitude_correctness"].agg(["mean", "std"]).sort_values("mean")
fig, ax = plt.subplots(figsize=(7.5, 0.45 * len(mc) + 1))
ax.barh(mc.index, mc["mean"], xerr=mc["std"], color=["#D9622B" if "TFT" in e else "#8A919B" for e in mc.index])
ax.set_xlabel("magnitude correctness (1 = perfect)"); ax.axvline(0, color="k", lw=0.6)
fig.tight_layout(); fig.savefig(PACK / "fig_A_estimators.png"); plt.close(fig)

json.dump(dict(run_plan=RUN_PLAN, generator=GENERATOR, models=MODELS, seeds=SEEDS, conditions=CONDITIONS, n_explain=N_EXPLAIN, n_grouped=N_GROUPED,
               perm_samples=PERM_SAMPLES, lime_samples=LIME_SAMPLES, robust=(N_ROBUST, ROBUST_K, ROBUST_R), faith=(N_FAITH, FAITH_K, FAITH_FRAC),
               arch=ARCH, train=TRAIN, shap=(SHAP_BACKGROUND, SHAP_NSAMPLES)), open(PACK / "settings.json", "w"), indent=2, default=str)
LG.to_csv(PACK / "log.csv", index=False)
(PACK / "RESULTS_EXTENSIONS.md").write_text("\n".join(MDL), encoding="utf8")
zp = PACK.with_suffix(".zip")
with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
    for f_ in PACK.iterdir(): z.write(f_, f_.name)
print("Results pack:", zp)


# ======================================================================================================================
# ==== PART 7: Part 7. A causal reference for the explanations: causal models against SHAP
# ======================================================================================================================

# %% [markdown]
# # Part 7. A causal reference for the explanations: causal models against SHAP
#
# **In the paper:** Sections 4.5 and 6.9. Needs `tigramite`. No neural model is trained here, so only the number of seeds changes with the plan (two seeds in the smoke plan). Every (dataset, seed) is cached in `results/causal_reference/`. The reference values labelled as published TFT-SHAP and XGBoost-TreeSHAP means are typed in by hand: update them from the tables of Part 1 once the deep run has finished.
#
# Same protocol as the paper's TFT / SHAP evaluation:
#
# * **Synthetic benchmark**, 5 seeds; score each driver (hour, weekday, holiday, temperature) by the pattern correlation **r** and the magnitude slope **b**, and **magnitude correctness = 1 − mean |1 − b|**.
# * **Real Spanish data with injected effects**, 5 seeds (Kaggle, 2015–2018, cleaned: impossible weather readings removed before averaging cities, missing hours interpolated, never dropped): an event (−1,500 MW, 08:00–21:00, 60 random weekdays, flagged), a nonlinear heat-dependent driver (up to ~900 MW), and a decoy correlated 0.8 with the driver but with no effect. Score r and b for event and driver, and **decoy credit** (|credit to decoy| / |credit to driver|, 0 is correct). About 30% of the driver's effect depends on temperature alone (its average grows with heat), so every method shares that part with temperature; the extra column *driver (driver-only part)* scores the slope against the rest. Magnitude correctness uses the paper's definition (full effect) for comparability with TFT-SHAP; *magnitude correctness (identifiable)* uses the event and the driver-only part, the two quantities an additive causal estimator can identify. How the temperature-only part is split between driver and temperature is a convention (each estimator and each Shapley variant splits it differently), so the full-driver slope is not a test of correctness for causal estimators.
#
# Compared:
#
# | Step | Methods |
# |---|---|
# | Causal discovery (which variables cause demand) | PCMCI+, LPCMCI (daily), NOTEARS, target-centred two-test selection, exogenous HAC screen, and PCMCI+ dynamics + HAC exogenous |
# | Causal effect estimation (how much), using the PCMCI+ ∪ HAC-screen parents | graph + additive regression with hour-of-week fixed effects and splines (GLS with AR-prewhitened errors, and OLS with month/year fixed effects); graph + g-computation (boosted trees); double machine learning (cross-fitted, prewhitened final stage); "shortcut" regression with demand history (negative control) |
# | Explanations of a forecasting model | marginal group Shapley (≈ TreeSHAP); causal asymmetric Shapley using the graph |
# | Paper reference | TFT-SHAP and XGBoost-TreeSHAP, published five-seed means |
#
# The notebook ends with the causal graph of the best model, with its estimated effect sizes on the edges.
#
# **Settings.** The plan (`smoke` or `deep`), the number of seeds and the training length come from the command line (`--plan`, `--seeds`, `--max-epochs`, ...). Everything else is set in the first lines of this part. Finished runs are saved and skipped when the part is run again.

# %%
# The launcher passes the shared settings as GLOBAL_SETTINGS (plan, seeds, training length, explained hours, search size).
import json
from pathlib import Path
print(GLOBAL_SETTINGS)

# %% [markdown]
# ## 0. Setup
# Needs `tigramite` in addition to the scientific Python stack (PyTorch is not needed here).

# %%
import os, time, pickle, itertools, warnings
from math import factorial
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats, optimize
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold
from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.lpcmci import LPCMCI
from tigramite.independence_tests.parcorr import ParCorr

warnings.filterwarnings("ignore")
pd.set_option("display.width", 170); pd.set_option("display.max_columns", 30)

# %% [markdown]
# ## 1. Configuration

# %%
CFG = dict(
    SEEDS=[0, 1] if GLOBAL_SETTINGS["plan"] == "smoke" else list(range(GLOBAL_SETTINGS["seeds"])),
    RUN_SYNTHETIC=True,
    RUN_INJECTED=True,                    # needs the two Kaggle files below
    REAL_ENERGY_CSV="energy_dataset.csv",        # Kaggle: nicholasjhana / energy-consumption-generation-prices-and-weather
                                                 # (found automatically next to the notebook or in Data/; or give a full path)
    REAL_WEATHER_CSV="weather_features.csv",
    CACHE_DIR="results/causal_reference/cache_" + GLOBAL_SETTINGS["plan"],
    PC_ALPHA=0.01,
    N_EXPLAIN=200, N_BG=30,               # forecast-model Shapley (as in the paper's grouped Shapley runs)
    N_GCOMP=2000, N_GCOMP_BG=50,          # g-computation evaluation points / background rows
)
os.makedirs(CFG["CACHE_DIR"], exist_ok=True)
# Cache files carry this tag: results from an older notebook version (different cleaning or estimators) are never reused.
CACHE_TAG = "v20261029"
def _find(name, folders=(".", "Data", "data", os.path.join("..", "Data"), os.path.join("..", "data"))):
    """Return the file if it exists as given, else look for it in the usual data folders."""
    if os.path.exists(name):
        return name
    for f in folders:
        cand = os.path.join(f, os.path.basename(name))
        if os.path.exists(cand):
            return cand
    return None

if CFG["RUN_INJECTED"]:
    e_path, w_path = _find(CFG["REAL_ENERGY_CSV"]), _find(CFG["REAL_WEATHER_CSV"])
    if e_path and w_path:
        CFG["REAL_ENERGY_CSV"], CFG["REAL_WEATHER_CSV"] = e_path, w_path
        print("real data:", os.path.abspath(e_path), "|", os.path.abspath(w_path))
    else:
        raise FileNotFoundError(
            "Kaggle files not found. Looked for energy_dataset.csv and weather_features.csv in "
            f"{os.path.abspath('.')} and its Data/ or data/ subfolders (and ../Data). Either copy them there, or set the full "
            'paths in CFG, e.g. REAL_ENERGY_CSV="path/to/energy_dataset.csv". '
            "To run only the synthetic benchmark on purpose, set RUN_INJECTED=False.")

# Published five-seed means from the paper (Table 5 benchmark, Table 6 injected) for comparison
PAPER = {
    "synthetic": pd.DataFrame({"hour": [0.70, 0.14], "week": [1.07, 0.52], "holiday": [1.01, 0.78], "temp": [0.69, 0.30],
                               "magnitude correctness": [0.80, 0.44]},
                              index=["TFT, SHAP (paper)", "XGBoost, TreeSHAP (paper)"]),
    "injected": pd.DataFrame({"event": [0.86, 0.35], "driver": [0.80, 0.49], "decoy credit": [0.09, 0.03],
                              "magnitude correctness": [0.78, 0.42]},
                             index=["TFT, SHAP (paper)", "XGBoost, TreeSHAP (paper)"]),
}

# %% [markdown]
# ## 2. Data: synthetic benchmark, real data, injection
# The benchmark generator is the one of Section 5 of the paper. `inject_effects` reproduces the paper's injection protocol; if your original injection code differs (for example the exact driver series), replace this one function and everything downstream stays the same.

# %%
def spanish_holidays(years):
    """Spanish national holidays incl. Good Friday."""
    out = []
    for y in years:
        a = y % 19; b, c = divmod(y, 100); d, e = divmod(b, 4)
        f = (b + 8) // 25; g = (b - f + 1) // 3
        h = (19 * a + b - d - g + 15) % 30; i, k = divmod(c, 4)
        l = (32 + 2 * e + 2 * i - h - k) % 7; m = (a + 11 * h + 22 * l) // 451
        month = (h + l - 7 * m + 114) // 31; day = ((h + l - 7 * m + 114) % 31) + 1
        easter = pd.Timestamp(y, month, day)
        out += [easter - pd.Timedelta(days=2)]
        out += [pd.Timestamp(y, mo, dd) for mo, dd in
                [(1, 1), (1, 6), (5, 1), (8, 15), (10, 12), (11, 1), (12, 6), (12, 8), (12, 25)]]
    return pd.DatetimeIndex(out).normalize()


def ar1(n, phi, sd_stationary, rng):
    e = rng.normal(0, sd_stationary * np.sqrt(1 - phi ** 2), n)
    x = np.empty(n); x[0] = rng.normal(0, sd_stationary)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e[t]
    return x


def generate_benchmark(seed=0, variant="default", latent_strength=1.0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2015-01-01", "2018-12-31 23:00", freq="h")
    n = len(idx)
    hour = idx.hour.values; dow = idx.dayofweek.values; doy = idx.dayofyear.values
    tt = np.arange(n)

    # calendar components
    C_hour = 2600 * np.cos(2 * np.pi * (hour - 13) / 24) + 900 * np.cos(4 * np.pi * (hour - 20) / 24)
    C_week = np.where(dow == 5, -1800.0, np.where(dow == 6, -3200.0, 0.0))
    hol = np.isin(idx.normalize(), spanish_holidays(range(2015, 2019))).astype(float)
    C_holiday = -2500.0 * hol

    # weather: 5 cities, seasonal + daily + persistent anomaly, averaged
    common_anom = ar1(n, 0.998, 2.5, rng)
    temps = []
    for base, amp, damp in [(15, 10, 6), (16.5, 8, 4), (18, 8, 5), (19, 10, 7), (14.5, 7, 4)]:
        city = (base + amp * np.cos(2 * np.pi * (doy - 200) / 365.25)
                + damp * np.cos(2 * np.pi * (hour - 15) / 24)
                + common_anom + ar1(n, 0.99, 1.0, rng))
        temps.append(city)
    T = np.mean(temps, axis=0)
    clouds = 100 / (1 + np.exp(-ar1(n, 0.98, 1.5, rng)))
    humidity = np.clip(65 + 0.25 * (clouds - 50) - 1.2 * (T - 16) + rng.normal(0, 5, n), 5, 100)
    pressure = 1015 + ar1(n, 0.995, 6, rng)
    wind_speed = np.abs(3 + ar1(n, 0.97, 2.0, rng))

    # temperature response with regime change in cooling
    change = {"default": "2016-07-01", "extrapolation": "2017-07-01", "stationary": None}[variant]
    cool_mult = np.ones(n) if change is None else np.where(idx >= pd.Timestamp(change), 1.6, 1.0)
    C_temp = 250 * np.clip(16 - T, 0, None) + cool_mult * 150 * np.clip(T - 22, 0, None) ** 1.3

    # hidden driver (economic activity): daily AR(0.995) interpolated to hours
    n_days = n // 24 + 2
    lat_d = ar1(n_days, 0.995, 1.0, rng)
    latent = np.interp(tt / 24.0, np.arange(n_days), lat_d)
    C_latent = 1200 * latent_strength * latent
    C_AR = ar1(n, 0.95, 500, rng)
    eps = rng.normal(0, 150, n)
    D = 28000 + C_hour + C_week + C_holiday + C_temp + C_latent + C_AR + eps

    # generation, produced after demand
    sun = np.clip(np.cos(2 * np.pi * (hour - 13.5) / 24), 0, None) ** 1.5
    solar = 4500 * sun * (1 + 0.3 * np.cos(2 * np.pi * (doy - 172) / 365.25)) * (1 - 0.7 * clouds / 100)
    wind = 18000 * np.minimum(1, (wind_speed / 11) ** 3)
    nuclear = np.full(n, 7000.0)
    for _ in range(12):
        s = rng.integers(0, n - 720); nuclear[s:s + 720] -= 1000
    hydro = 3000 + 1500 * np.cos(2 * np.pi * (doy - 60) / 365.25) + ar1(n, 0.99, 300, rng)
    residual = D - solar - wind - nuclear - hydro
    gas = np.clip(0.55 * residual + 600 * latent_strength * latent + rng.normal(0, 200, n), 0, None)
    coal = np.clip(residual - gas, 0, None)

    df = pd.DataFrame(dict(
        time=idx, demand=D, temp=T, humidity=humidity, pressure=pressure, wind_speed=wind_speed,
        clouds=clouds, holiday=hol, gen_solar=solar, gen_wind=wind, gen_nuclear=nuclear,
        gen_hydro=hydro, gen_gas=gas, gen_coal=coal,
        C_hour=C_hour, C_week=C_week, C_holiday=C_holiday, C_temp=C_temp,
        C_latent=C_latent, C_AR=C_AR))
    return df.set_index("time")


def load_kaggle(energy_csv, weather_csv):
    """Kaggle 'Hourly energy demand generation and weather' (Spain 2015-2018), cleaned:
    impossible weather readings set to missing per city BEFORE averaging over the 5 cities, local Madrid time,
    every hour present (short gaps interpolated, never dropped: dropping a row would shift all lags after it)."""
    en = pd.read_csv(energy_csv)
    en["time"] = pd.to_datetime(en["time"], utc=True).dt.tz_convert("Europe/Madrid").dt.tz_localize(None)
    en = en.drop_duplicates("time").set_index("time")           # autumn DST: repeated local hour dropped
    w = pd.read_csv(weather_csv)
    w["time"] = pd.to_datetime(w["dt_iso"], utc=True).dt.tz_convert("Europe/Madrid").dt.tz_localize(None)
    w = w.drop_duplicates(["time", "city_name"])
    w.loc[~w["pressure"].between(900, 1100), "pressure"] = np.nan
    w.loc[w["wind_speed"] > 50, "wind_speed"] = np.nan
    w.loc[w["humidity"] <= 0, "humidity"] = np.nan
    wm = w.groupby("time")[["temp", "humidity", "pressure", "wind_speed", "clouds_all"]].mean()
    wm["temp"] = wm["temp"] - 273.15
    df = pd.DataFrame(index=en.index)
    df["demand"] = en["total load actual"]
    m = {"gen_solar": "generation solar", "gen_wind": "generation wind onshore", "gen_nuclear": "generation nuclear",
         "gen_hydro": "generation hydro water reservoir", "gen_gas": "generation fossil gas",
         "gen_coal": "generation fossil hard coal"}
    for k, c in m.items():
        df[k] = en[c]
    df.loc[(df["gen_nuclear"] == 0) | (df["gen_gas"] == 0), ["gen_nuclear", "gen_gas", "gen_solar"]] = np.nan  # 4 rows coded 0
    df = df.join(wm.rename(columns={"clouds_all": "clouds"}), how="left")
    df = df.asfreq("h")                                          # spring DST: missing local hour inserted
    n_missing = int(df["demand"].isna().sum())
    df = df.interpolate(method="time", limit=12, limit_direction="both")
    assert df.notna().all().all(), "gap longer than 12 h in the real data"
    assert (np.diff(df.index.values).astype("timedelta64[h]").astype(int) == 1).all()
    df["holiday"] = np.isin(df.index.normalize(), spanish_holidays(range(2015, 2019))).astype(float)
    print(f"real data: {len(df)} contiguous hours, {n_missing} missing demand hours interpolated")
    return df


DRIVER_PERSISTENCE = 0.95    # hourly AR coefficient of the injected driver (slow: memory of ~20 hours).
# At 0.99 the driver is as slow as the drift of real demand and no hourly method separates it from that drift.


def inject_effects(df, seed):
    """Event (-1,500 MW, 08-21h, 60 random weekdays, flagged), heat-dependent nonlinear driver (<= ~900 MW),
    decoy correlated 0.8 with the driver and without effect. Returns the modified frame and the true components."""
    rng = np.random.default_rng(1000 + seed)
    out = df.copy(); idx = out.index; n = len(out)
    days = pd.Index(idx.normalize().unique())
    weekdays = days[(days.dayofweek < 5) & ~days.isin(spanish_holidays(range(days.year.min(), days.year.max() + 1)))]
    ev_days = pd.DatetimeIndex(rng.choice(weekdays, 60, replace=False))
    flag = (idx.normalize().isin(ev_days) & (idx.hour >= 8) & (idx.hour <= 21)).astype(float)
    x = ar1(n, DRIVER_PERSISTENCE, 1.0, rng)                        # slow driver, unit variance
    heat = np.clip((out["temp"].values - 10) / 20, 0, 1)            # 0 below 10 °C, 1 above 30 °C
    drv_eff = 900.0 / (1 + np.exp(-1.5 * x)) * (0.3 + 0.7 * heat)   # nonlinear, grows with heat, 0 to 900 MW
    decoy = 0.8 * x + 0.6 * ar1(n, DRIVER_PERSISTENCE, 1.0, rng)    # corr ~0.8, same timescale, no effect
    out["event_flag"] = flag; out["driver"] = x; out["decoy"] = decoy
    out["C_event"] = -1500.0 * flag; out["C_driver"] = drv_eff - drv_eff.mean()
    # Part of the driver effect that does not depend on temperature alone: the driver's average effect grows with heat,
    # and that temperature-only part (about 30% of its variance) is shared with temperature by any method.
    e_logistic = np.mean(900.0 / (1 + np.exp(-1.5 * np.random.default_rng(0).normal(size=200000))))
    only = drv_eff - e_logistic * (0.3 + 0.7 * heat)
    out["C_driver_only"] = only - only.mean()
    out["demand"] = out["demand"] - 1500.0 * flag + drv_eff
    return out

# %% [markdown]
# ## 3. Variables, groups and roles
# Discovery runs on raw hourly series. Each variable belongs to a **group** (one driver) and has a **role** (true cause, history, consequence, irrelevant). On real data the true role of calendar and weather is unknown, so only the injected event, driver and decoy are scored.

# %%
DEMAND_LAGS = [1, 2, 24, 48, 168]
GEN_LAGS = [1, 24]
CALENDAR = ["h_s1", "h_c1", "h_s2", "h_c2", "sat", "sun", "holiday"]
WEATHER = ["hdd", "cdd", "humidity", "pressure", "wind_speed", "clouds"]
GEN = ["gen_solar", "gen_gas"]
INJ = ["event_flag", "driver", "decoy"]
GROUP = {"demand": "history", "h_s1": "hour", "h_c1": "hour", "h_s2": "hour", "h_c2": "hour",
         "sat": "week", "sun": "week", "holiday": "holiday", "hdd": "temp", "cdd": "temp",
         "doy_s": "season", "doy_c": "season", "humidity": "other weather", "pressure": "other weather",
         "wind_speed": "other weather", "clouds": "other weather", "gen_solar": "solar", "gen_gas": "gas",
         "event_flag": "event", "driver": "driver", "decoy": "decoy"}
TRUTH = {
    "synthetic": dict(causes={"hour", "week", "holiday", "temp", "history"},
                      scored={"hour": "C_hour", "week": "C_week", "holiday": "C_holiday", "temp": "C_temp"},
                      consequences={"gas"}, irrelevant={"season", "other weather", "solar"}),
    "injected": dict(causes={"event", "driver"}, scored={"event": "C_event", "driver": "C_driver"},
                     consequences=set(), irrelevant={"decoy"}),
}


def build_vars(df, injected=False):
    h = df.index.hour.values; dw = df.index.dayofweek.values; dy = df.index.dayofyear.values
    V = pd.DataFrame(index=df.index)
    V["demand"] = df["demand"].values
    V["h_s1"] = np.sin(2 * np.pi * h / 24); V["h_c1"] = np.cos(2 * np.pi * h / 24)
    V["h_s2"] = np.sin(4 * np.pi * h / 24); V["h_c2"] = np.cos(4 * np.pi * h / 24)
    V["sat"] = (dw == 5).astype(float); V["sun"] = (dw == 6).astype(float); V["holiday"] = df["holiday"].values
    V["hdd"] = np.clip(18 - df["temp"].values, 0, None); V["cdd"] = np.clip(df["temp"].values - 18, 0, None)
    V["doy_s"] = np.sin(2 * np.pi * dy / 365.25); V["doy_c"] = np.cos(2 * np.pi * dy / 365.25)
    for c in ["humidity", "pressure", "wind_speed", "clouds"]:
        V[c] = df[c].values
    if injected:
        for c in INJ:
            V[c] = df[c].values
    for c in GEN:
        V[c] = df[c].values
    return V


def exog_of(names):
    return [n for n in names if n not in ("demand",) and n not in GEN]

# %% [markdown]
# ## 4. Causal discovery methods
# **Why a second, exogenous screen.** Two reasons. (1) PCMCI+ tests each candidate given demand's own past. A *slow* exogenous cause (the injected driver, or economic activity) changes little from one hour to the next, so demand's past already contains its effect and the test has almost no power: the cause is hidden by the history that it created. For exogenous candidates, demand history is a descendant, not a confounder, so the effect can be tested without it; the price is strong autocorrelation, handled with Newey–West HAC standard errors (two-week Bartlett window). (2) PCMCI+ uses a *linear* partial-correlation test (nonlinear tests are too slow for 35,000 hours): when a cause acts nonlinearly (the injected driver), conditioning on it linearly leaves part of its effect, and a correlated variable without effect (the decoy) can pick that up and be named a cause. The screen tests each group in a flexible basis, conditional on all other groups. The final graph therefore takes the dynamic structure and orientation (history, generation) from PCMCI+, and the exogenous causes from the screen.
#
# Background knowledge in PCMCI+/LPCMCI: calendar, weather and injected series are exogenous (nothing points into them; weather and the injected driver/decoy keep their own past as a parent); the direction between demand and generation is left open. Gas and coal are not both included: generation adds up to demand, and conditioning on both makes demand nearly deterministic (a faithfulness violation that hides real causes).

# %%
def link_assumptions(names):
    idx = {n: i for i, n in enumerate(names)}; j = idx["demand"]
    exog = exog_of(names); L = {i: {} for i in range(len(names))}
    for e in exog:
        L[j][(idx[e], 0)] = "-?>"; L[idx[e]][(j, 0)] = "<?-"
        if e in WEATHER + ["driver", "decoy"]:
            L[idx[e]][(idx[e], -1)] = "-?>"
    for lag in DEMAND_LAGS:
        L[j][(j, -lag)] = "-?>"
    for g in [n for n in names if n in GEN]:
        gi = idx[g]
        L[j][(gi, 0)] = "o?o"; L[gi][(j, 0)] = "o?o"
        for lag in GEN_LAGS:
            L[j][(gi, -lag)] = "-?>"; L[gi][(j, -lag)] = "-?>"; L[gi][(gi, -lag)] = "-?>"
        for e in exog:
            L[gi][(idx[e], 0)] = "-?>"; L[idx[e]][(gi, 0)] = "<?-"
        for g2 in GEN:
            if g2 != g and g2 in idx:
                L[gi][(idx[g2], 0)] = "o?o"
    return L


def run_pcmciplus(V, alpha):
    names = list(V.columns)
    pc = PCMCI(dataframe=pp.DataFrame(V.values.astype(float), var_names=names),
               cond_ind_test=ParCorr(significance="analytic"), verbosity=0)
    res = pc.run_pcmciplus(tau_min=0, tau_max=max(DEMAND_LAGS), pc_alpha=alpha, link_assumptions=link_assumptions(names))
    j = names.index("demand"); parents, edges = set(), []
    for i, n in enumerate(names):
        for tau in range(res["graph"].shape[2]):
            e = res["graph"][i, j, tau]
            if e == "" or (i == j and tau == 0): continue
            edges.append((n, tau, e, float(res["val_matrix"][i, j, tau])))
            if e in ("-->", "o-o", "x-x", "o->", "x->"): parents.add(n)
    return dict(parents=parents, edges=edges, graph=res["graph"], val=res["val_matrix"], names=names)


def run_lpcmci_daily(V, alpha, tau_max=7):
    d = V.resample("D").mean().dropna()
    d["weekend"] = d["sat"] + d["sun"]
    names = ["demand", "weekend", "holiday", "hdd", "cdd", "humidity"] + [c for c in INJ if c in d] + GEN
    idx = {n: i for i, n in enumerate(names)}; j = 0
    exo = [n for n in names if n not in ["demand"] + GEN]
    L = {i: {} for i in range(len(names))}
    for e in exo:
        L[j][(idx[e], 0)] = "-?>"; L[idx[e]][(j, 0)] = "<?-"
        if e in ("hdd", "cdd", "humidity", "driver", "decoy"): L[idx[e]][(idx[e], -1)] = "-?>"
    for lag in [1, 7]: L[j][(j, -lag)] = "-?>"
    for g in GEN:
        gi = idx[g]; L[j][(gi, 0)] = "o?o"; L[gi][(j, 0)] = "o?o"
        L[j][(gi, -1)] = "o?>"; L[gi][(j, -1)] = "o?>"; L[gi][(gi, -1)] = "o?>"
        for e in exo: L[gi][(idx[e], 0)] = "-?>"; L[idx[e]][(gi, 0)] = "<?-"
        for g2 in GEN:
            if g2 != g: L[gi][(idx[g2], 0)] = "o?o"
    lp = LPCMCI(dataframe=pp.DataFrame(d[names].values, var_names=names), cond_ind_test=ParCorr(significance="analytic"), verbosity=0)
    G = lp.run_lpcmci(link_assumptions=L, tau_max=tau_max, pc_alpha=alpha)["graph"]
    parents = {names[i] for i in range(len(names)) for t in range(G.shape[2])
               if G[i, 0, t] in ("-->",) and not (i == 0 and t == 0)}
    return {("week" if p == "weekend" else p) for p in parents}


def notears_linear(X, lambda1=0.1, max_iter=60, h_tol=1e-8, rho_max=1e16, w_threshold=0.3):
    """Linear NOTEARS with least-squares loss (Zheng et al., 2018)."""
    n, d = X.shape

    def _adj(w): return (w[:d * d] - w[d * d:]).reshape(d, d)

    def _h(W):
        E = __import__("scipy").linalg.expm(W * W); h = np.trace(E) - d
        return h, E.T * W * 2

    def _func(w):
        W = _adj(w); R = X - X @ W
        loss = 0.5 / n * (R ** 2).sum(); G_loss = -1.0 / n * X.T @ R
        h, G_h = _h(W)
        obj = loss + 0.5 * rho * h * h + alpha * h + lambda1 * w.sum()
        G = G_loss + (rho * h + alpha) * G_h
        return obj, np.concatenate((G + lambda1, -G + lambda1), axis=None)

    w_est, rho, alpha, h = np.zeros(2 * d * d), 1.0, 0.0, np.inf
    bnds = [(0, 0) if i == j else (0, None) for _ in range(2) for i in range(d) for j in range(d)]
    for _ in range(max_iter):
        w_new, h_new = None, None
        while rho < rho_max:
            sol = optimize.minimize(_func, w_est, method="L-BFGS-B", jac=True, bounds=bnds)
            w_new = sol.x; h_new, _ = _h(_adj(w_new))
            if h_new > 0.25 * h: rho *= 10
            else: break
        w_est, h = w_new, h_new; alpha += rho * h
        if h <= h_tol or rho >= rho_max: break
    W = _adj(w_est); W[np.abs(W) < w_threshold] = 0
    return W


def run_notears(df, V, seed):
    F = pd.DataFrame(index=df.index)
    F["demand"] = df["demand"]
    F["d_lag48"] = df["demand"].shift(48); F["d_lag168"] = df["demand"].shift(168)
    F["d_roll24_lag48"] = df["demand"].rolling(24).mean().shift(48)
    for c in [c for c in V.columns if c != "demand" and c not in GEN]:
        F[c] = V[c]
    for c in GEN:
        F[c + "_lag48"] = df[c].shift(48)
    F = F.dropna().sample(4000, random_state=seed)
    W = notears_linear(((F - F.mean()) / F.std()).values)
    cols = list(F.columns)
    par = [cols[i] for i in np.nonzero(W[:, 0])[0]]
    return {("demand" if c.startswith("d_") else c.replace("_lag48", "")) for c in par}


def partial_corr_test(x, y, Z):
    if Z is not None and Z.shape[1] > 0:
        Z1 = np.column_stack([np.ones(len(x)), Z])
        bx = np.linalg.lstsq(Z1, x, rcond=None)[0]; by = np.linalg.lstsq(Z1, y, rcond=None)[0]
        rx, ry = x - Z1 @ bx, y - Z1 @ by; k = Z.shape[1]
    else:
        rx, ry = x - x.mean(), y - y.mean(); k = 0
    r = np.corrcoef(rx, ry)[0, 1]; n = len(x)
    tstat = r * np.sqrt((n - 2 - k) / max(1 - r * r, 1e-12))
    return r, 2 * stats.t.sf(abs(tstat), n - 2 - k)


def run_target_selection(V, alpha):
    y = V["demand"]; exog = exog_of(V.columns)
    hist = pd.concat({f"y_l{l}": y.shift(l) for l in DEMAND_LAGS}, axis=1)
    sel = set()
    for c in [c for c in V.columns if c != "demand"]:
        x = V[c]
        own = [] if c in CALENDAR + ["doy_s", "doy_c", "event_flag"] else [x.shift(1).rename("x_l1"), x.shift(24).rename("x_l24")]
        A = pd.concat([x.rename("x"), y.rename("y"), hist, V[exog].drop(columns=[c], errors="ignore")] + own, axis=1).dropna()
        _, pA = partial_corr_test(A["x"].values, A["y"].values, A.drop(columns=["x", "y"]).values)
        pB = 1.0
        if c in GEN:
            B = pd.concat([x.shift(-1).rename("xn"), y.rename("y"), x.rename("x0"), x.shift(1).rename("x1"),
                           V[exog].shift(-1).add_suffix("_n"), hist], axis=1).dropna()
            _, pB = partial_corr_test(B["xn"].values, B["y"].values, B.drop(columns=["xn", "y"]).values)
        if pA < alpha and (c not in GEN or pB >= alpha):
            sel.add(c)
    return sel | {"demand"}


AR_LAGS = (1, 2, 24, 168)


def ar_filter(resid, lags=AR_LAGS):
    """AR coefficients of the regression errors (Cochrane-Orcutt step)."""
    L = max(lags); E = np.column_stack([resid[L - l:len(resid) - l] for l in lags])
    return np.linalg.lstsq(E, resid[L:], rcond=None)[0]


def quasi_diff(Z, rho, lags=AR_LAGS):
    """Z_t - sum_l rho_l Z_{t-l}: removes the slowly autocorrelated part of the errors."""
    Z = np.asarray(Z, float); L = max(lags); out = Z[L:].copy()
    for r, l in zip(rho, lags):
        out -= r * Z[L - l:len(Z) - l]
    return out


def gls_ar(Xm, y, n_iter=2):
    """Feasible GLS with AR(lags) errors. Returns beta, transformed design, transformed residuals."""
    beta = np.linalg.lstsq(Xm, y, rcond=None)[0]
    for _ in range(n_iter):
        rho = ar_filter(y - Xm @ beta)
        Xs, ys = quasi_diff(Xm, rho), quasi_diff(y, rho)
        beta = np.linalg.lstsq(Xs, ys, rcond=None)[0]
    return beta, Xs, ys - Xs @ beta


def hac_wald_screen(V, df, alpha, bandwidth=48):
    """Slow-driver screen: regress demand on flexible bases of ALL exogenous candidate groups (no demand history,
    which is a descendant of them), fit by GLS with AR-prewhitened errors, and test each group with a Newey-West
    (Bartlett, 48 h) HAC Wald test on the prewhitened residuals."""
    cand = sorted({GROUP[c] for c in exog_of(V.columns)})
    Bs = {g: basis(V, df, g) for g in cand}
    X = pd.concat(list(Bs.values()), axis=1); X = (X - X.mean()) / (X.std() + 1e-12)
    Xm = np.column_stack([np.ones(len(X)), X.values]); y = V["demand"].values
    beta, Xs, e = gls_ar(Xm, y)                      # prewhitened (GLS) fit; HAC on what is left
    Xe = Xs * e[:, None]; S = Xe.T @ Xe
    for l in range(1, bandwidth + 1):
        G = Xe[l:].T @ Xe[:-l]; S += (1 - l / (bandwidth + 1)) * (G + G.T)
    XtX_inv = np.linalg.pinv(Xs.T @ Xs); Vb = XtX_inv @ S @ XtX_inv
    out, col = {}, 1
    for g, B in Bs.items():
        k = B.shape[1]; ix = np.arange(col, col + k); col += k
        b = beta[ix]; W = float(b @ np.linalg.pinv(Vb[np.ix_(ix, ix)]) @ b)
        out[g] = stats.chi2.sf(W, k)
    return {g for g, pv in out.items() if pv < alpha}, out


def score_discovery(parents, dataset):
    T = TRUTH[dataset]; groups = {GROUP.get(p, p) for p in parents}
    if dataset == "synthetic":
        found = groups & T["causes"]; prec = len(found) / max(len(groups), 1); rec = len(found) / len(T["causes"])
        return dict(precision=prec, recall=rec, F1=0 if prec + rec == 0 else 2 * prec * rec / (prec + rec),
                    false_parents=",".join(sorted(groups - T["causes"])) or "-")
    return dict(event_found=float("event" in groups), driver_found=float("driver" in groups),
                decoy_named_cause=float("decoy" in groups),
                false_parents=",".join(sorted(groups & {"decoy"})) or "-")

# %% [markdown]
# ## 5. Causal effect estimators and explanations
# All estimators use the **exogenous parents from PCMCI+ dynamics + HAC exogenous** as treatments and covariates. Demand history is not used: it is a descendant of the calendar and weather, so adjusting for it blocks their effect (it is included only in the "shortcut" negative control). Each method returns, for every hour, a contribution in MW per driver group, which is scored exactly like a SHAP attribution.

# %%
TEMP_KNOTS = np.arange(4, 34, 3)


def basis(V, df, group):
    """Flexible basis for one driver group (what the estimator is allowed to learn)."""
    h = df.index.hour.values
    if group == "hour":                               # hour-of-day fixed effects: the injected event is a block of
        return pd.get_dummies(h, prefix="hr", drop_first=True).set_index(V.index).astype(float)   # hours, so hour must be fully controlled
    if group == "temp":
        T = df["temp"]; B = pd.DataFrame({f"tk{k}": np.clip(T - k, 0, None) for k in TEMP_KNOTS}, index=V.index); B["tlin"] = T
        return B
    if group in ("driver", "decoy"):
        x = V[group]; qs = np.quantile(x, [0.2, 0.4, 0.6, 0.8])
        B = pd.DataFrame({f"{group}_k{i}": np.clip(x - q, 0, None) for i, q in enumerate(qs)}, index=V.index); B[group] = x
        heat = np.clip((df["temp"] - 10) / 20, 0, 1)
        for c in list(B.columns): B[c + "_xheat"] = B[c] * heat     # allows a heat interaction, as any flexible model would
        return B
    if group in ("week", "holiday"):                 # level + change of the daily profile (real weekends and
        h = df.index.hour.values                       # holidays have a different shape, not only a lower level)
        H = pd.DataFrame({f"h{k}{f.__name__}": f(2 * np.pi * k * h / 24) for k in (1, 2, 3) for f in (np.sin, np.cos)}, index=V.index)
        cols = ["sat", "sun"] if group == "week" else ["holiday"]
        B = V[cols].copy()
        for c in cols:
            for hc in H.columns:
                B[f"{c}_x_{hc}"] = V[c] * H[hc]
        return B
    cols = [c for c in V.columns if GROUP.get(c) == group]
    return V[cols].copy()


def treatment_groups(parents):
    return sorted({GROUP[p] for p in parents if GROUP.get(p) not in ("history", "gas", "solar")})


def time_fe(index):
    """Month and year fixed effects: slow seasonal and trend levels that are not one of the drivers."""
    m = pd.get_dummies(index.month, prefix="m", drop_first=True).set_index(index).astype(float)
    y = pd.get_dummies(index.year, prefix="y", drop_first=True).set_index(index).astype(float)
    return pd.concat([m, y], axis=1)


def est_graph_ols(V, df, parents, add_history=False, method="gls"):
    """method='gls': AR-prewhitened GLS. method='ols_fe': OLS in levels with month and year fixed effects.
    The daily and weekly profile is fitted nonparametrically with hour-of-week fixed effects (167 dummies) and
    split afterwards into its weekday part (mean per day) and its daily-profile part (the rest)."""
    groups = treatment_groups(parents)
    how = ("hour" in groups) or ("week" in groups)
    Bs = {g: basis(V, df, g) for g in groups if g not in ("hour", "week")}
    if how:
        Bs["hour-of-week"] = pd.get_dummies(df.index.dayofweek * 24 + df.index.hour, prefix="how",
                                            drop_first=True).set_index(V.index).astype(float)
    X = pd.concat(list(Bs.values()), axis=1)
    if method == "ols_fe":
        X = pd.concat([X, time_fe(V.index)], axis=1)
    if add_history:
        X = pd.concat([X] + [V["demand"].shift(l).rename(f"y_l{l}") for l in (24, 48, 168)], axis=1)
    D = pd.concat([V["demand"].rename("y"), X], axis=1).dropna()
    if add_history or method == "ols_fe":             # OLS in levels (shortcut: plain OLS with demand lags)
        coef = dict(zip(D.columns[1:], LinearRegression().fit(D.drop(columns="y"), D["y"]).coef_))
    else:                                             # GLS with AR-prewhitened errors (exogenous regressors)
        Xm = np.column_stack([np.ones(len(D)), D.drop(columns="y").values])
        beta, _, _ = gls_ar(Xm, D["y"].values); coef = dict(zip(D.columns[1:], beta[1:]))
    out = {}
    for g, B in Bs.items():
        c = pd.Series(np.nan, index=V.index); v = B.loc[D.index].values @ np.array([coef[k] for k in B.columns])
        c.loc[D.index] = v - v.mean(); out[g] = c
    if how:
        c = out.pop("hour-of-week"); day = c.groupby(c.index.normalize()).transform("mean")
        out["week"] = day - day.mean(); out["hour"] = c - day
    return out


def raw_inputs(V, df, groups):
    cols = {}
    for g in groups:
        if g == "hour": cols[g] = ["h_s1", "h_c1", "h_s2", "h_c2"]
        elif g == "temp": cols[g] = ["temp_raw"]
        else: cols[g] = [c for c in V.columns if GROUP.get(c) == g]
    X = V[[c for g in groups for c in cols[g] if c != "temp_raw"]].copy()
    if "temp" in groups: X["temp_raw"] = df["temp"].values
    return X, cols


def est_gcomputation(V, df, parents, seed, n_eval, n_bg):
    """g-formula with a boosted-tree outcome model on the exogenous parents (no history):
    contribution_g(t) = f(x_t) - E_b f(x_t with group g set to background row b)."""
    groups = treatment_groups(parents); X, cols = raw_inputs(V, df, groups)
    f = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08, max_leaf_nodes=63, random_state=seed).fit(X, V["demand"])
    rng = np.random.default_rng(seed)
    ev = rng.choice(len(X), min(n_eval, len(X)), replace=False); bg = rng.choice(len(X), n_bg, replace=False)
    Xe, Xb = X.values[ev], X.values[bg]; fx = f.predict(X.iloc[ev])
    out = {}
    for g in groups:
        ci = [X.columns.get_loc(c) for c in cols[g]]
        M = np.repeat(Xe[:, None, :], n_bg, axis=1); M[:, :, ci] = Xb[None, :, ci]
        base = f.predict(pd.DataFrame(M.reshape(-1, X.shape[1]), columns=X.columns)).reshape(len(ev), n_bg).mean(1)
        s = pd.Series(np.nan, index=V.index); c = fx - base; s.iloc[ev] = c - c.mean(); out[g] = s
    return out


def est_dml(V, df, parents, seed, folds=3):
    """Double/debiased ML, partially linear in a flexible basis of each treatment group:
    y = phi_g(T_g)' theta_g + m(other parents) + e, with cross-fitted boosted-tree nuisances."""
    groups = treatment_groups(parents); Xraw, cols = raw_inputs(V, df, groups); y = V["demand"].values
    out = {}
    kf = KFold(folds, shuffle=True, random_state=seed)
    hgb = lambda: HistGradientBoostingRegressor(max_iter=60, learning_rate=0.1, max_leaf_nodes=31, random_state=seed)
    for g in groups:
        ctrl = Xraw.drop(columns=cols[g]); B = basis(V, df, g).values
        if ctrl.shape[1] == 0:
            ry, rB = y - y.mean(), B - B.mean(0)
        else:
            ry = np.zeros_like(y); rB = np.zeros_like(B)
            for tr, te in kf.split(ctrl):
                ry[te] = y[te] - hgb().fit(ctrl.iloc[tr], y[tr]).predict(ctrl.iloc[te])
                for k in range(B.shape[1]):
                    rB[te, k] = B[te, k] - hgb().fit(ctrl.iloc[tr], B[tr, k]).predict(ctrl.iloc[te])
        theta = np.linalg.lstsq(rB, ry, rcond=None)[0]
        rho = ar_filter(ry - rB @ theta)                # prewhiten the orthogonalised regression (slow errors)
        theta = np.linalg.lstsq(quasi_diff(rB, rho), quasi_diff(ry, rho), rcond=None)[0]
        c = B @ theta; out[g] = pd.Series(c - c.mean(), index=V.index)
    return out

# %% [markdown]
# ### Explanations of a forecasting model (the SHAP side of the comparison)
# A boosted-tree forecaster with day-ahead inputs (calendar, weather, injected series, demand lags 48/168, lagged generation), explained with exact group Shapley values two ways: **marginal** (≈ interventional TreeSHAP) and **causal asymmetric** (calendar/weather/injected first; demand history and generation recomputed from them through learned g-models plus each background unit's own noise). Groups are scored like the paper's grouped SHAP.

# %%
def forecaster_frame(df, V):
    F = pd.DataFrame(index=df.index)
    F["demand"] = df["demand"]
    F["d_lag48"] = df["demand"].shift(48); F["d_lag168"] = df["demand"].shift(168)
    F["d_roll24_lag48"] = df["demand"].rolling(24).mean().shift(48)
    for c in [c for c in V.columns if c != "demand" and c not in GEN]:
        F[c] = V[c]
    for c in GEN:
        F[c + "_lag48"] = df[c].shift(48)
    return F.dropna()


def fc_groups(F):
    G = {}
    for c in F.columns:
        if c == "demand": continue
        if c.startswith("d_"): g = "history"
        elif c.endswith("_lag48"): g = "generation"
        else: g = GROUP.get(c, "other")
        if g in ("season", "other weather"): g = "other exogenous"
        G.setdefault(g, []).append(c)
    return G


def shapley_values(vals, players, first=None):
    """Exact Shapley values from coalition values; `first` = players that must precede all others (asymmetric)."""
    n = len(players); idx = range(n); phi = np.zeros((len(vals[frozenset()]), n))
    def sym(sub_players, base):
        m = len(sub_players)
        for i in sub_players:
            others = [p for p in sub_players if p != i]
            for r in range(m):
                w = factorial(r) * factorial(m - r - 1) / factorial(m)
                for S in itertools.combinations(others, r):
                    S = frozenset(S) | base
                    phi[:, i] += w * (vals[S | {i}] - vals[S])
    if first is None:
        sym(list(idx), frozenset())
    else:
        A = [i for i in idx if players[i] in first]; Bp = [i for i in idx if players[i] not in first]
        sym(A, frozenset()); sym(Bp, frozenset(A))
    return phi


def forecaster_shapley(df, V, seed, n_exp, n_bg, explain_index=None):
    F = forecaster_frame(df, V); G = fc_groups(F); players = list(G)
    cols = [c for g in players for c in G[g]]; ci = {c: i for i, c in enumerate(cols)}
    endo = [g for g in players if g in ("history", "generation")]; exo = [g for g in players if g not in endo]
    excols = [c for g in exo for c in G[g]]; encols = [c for g in endo for c in G[g]]
    tr = F.loc[:"2017-12-31"]; te = F.loc["2018-01-01":]
    f = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.08, max_leaf_nodes=63, random_state=seed).fit(tr[cols], tr["demand"])
    gm = {c: HistGradientBoostingRegressor(max_iter=150, learning_rate=0.1, random_state=seed).fit(tr[excols], tr[c]) for c in encols}
    E = te.loc[te.index.intersection(explain_index)] if explain_index is not None else te.sample(n_exp, random_state=seed)
    n_exp = len(E); Bk = tr.sample(n_bg, random_state=seed + 1)
    noise = {c: Bk[c].values - gm[c].predict(Bk[excols]) for c in encols}
    out = {}
    for causal in (False, True):
        vals, cache = {}, {}
        for r in range(len(players) + 1):
            for S in itertools.combinations(range(len(players)), r):
                Sg = {players[i] for i in S}
                M = np.repeat(Bk[cols].values[None], n_exp, 0)
                for g in Sg:
                    for c in G[g]: M[:, :, ci[c]] = E[c].values[:, None]
                flat = M.reshape(-1, len(cols))
                if causal:
                    key = frozenset(Sg & set(exo))
                    if key not in cache:
                        ex = pd.DataFrame(flat[:, [ci[c] for c in excols]], columns=excols)
                        cache[key] = {c: gm[c].predict(ex) + np.tile(noise[c], n_exp) for c in encols}
                    for g in endo:
                        if g not in Sg:
                            for c in G[g]: flat[:, ci[c]] = cache[key][c]
                vals[frozenset(S)] = f.predict(pd.DataFrame(flat, columns=cols)).reshape(n_exp, n_bg).mean(1)
        phi = shapley_values(vals, players, first=set(exo) if causal else None)
        out["causal asymmetric Shapley" if causal else "marginal Shapley (≈TreeSHAP)"] = {
            g: pd.Series(phi[:, players.index(g)], index=E.index) for g in players}
    mae = float(np.mean(np.abs(f.predict(te[cols]) - te["demand"])))
    return out, mae

# %% [markdown]
# ## 6. Scoring (same metrics as the paper)

# %%
def score_contrib(contrib, df, dataset):
    T = TRUTH[dataset]; row = {}
    for g, comp in T["scored"].items():
        s = contrib.get(g)
        if s is None:
            row[f"r {g}"], row[g] = 0.0, 0.0; continue
        s = s.dropna(); tru = df.loc[s.index, comp]
        m = s.index >= pd.Timestamp("2018-01-01")            # score on the 2018 test year, like the forecasters
        if True:
            if m.sum() > 50: s, tru = s[m], tru[m]
        a, t = s.values - s.values.mean(), tru.values - tru.values.mean()
        ok = a.std() > 0 and t.std() > 0
        row[f"r {g}"] = float(np.corrcoef(a, t)[0, 1]) if ok else 0.0
        row[g] = float(np.polyfit(t, a, 1)[0]) if ok else 0.0
    bs = [row[g] for g in T["scored"]]
    row["magnitude correctness"] = 1 - float(np.mean(np.abs(1 - np.array(bs))))
    if dataset == "injected":
        s = contrib.get("driver")
        if s is not None:
            s = s.dropna(); s = s[s.index >= pd.Timestamp("2018-01-01")]; t = df.loc[s.index, "C_driver_only"]
            row["driver (driver-only part)"] = float(np.polyfit(t - t.mean(), s - s.mean(), 1)[0]) if s.std() > 0 else 0.0
        row["magnitude correctness (identifiable)"] = 1 - float(np.mean([abs(1 - row["event"]),
                                                                           abs(1 - row.get("driver (driver-only part)", 0.0))]))
        dec, drv = contrib.get("decoy"), contrib.get("driver")
        row["decoy credit"] = 0.0 if dec is None or drv is None else float(dec.abs().mean() / max(drv.abs().mean(), 1e-9))
    return row

# %% [markdown]
# ## 7. Run all seeds on both datasets
# Each (dataset, seed) result is cached, and PCMCI+ separately (`*_pcmci_<tag>.pkl`). File names carry the notebook version tag (`CACHE_TAG`), so results from an older version are ignored and recomputed automatically.

# %%
def run_one(dataset, seed, real_base=None):
    path = os.path.join(CFG["CACHE_DIR"], f"{dataset}_seed{seed}_{CACHE_TAG}.pkl")
    if os.path.exists(path):
        return pickle.load(open(path, "rb"))
    t0 = time.time()
    if dataset == "synthetic":
        df = generate_benchmark(seed, "default", 1.0)
    else:
        df = inject_effects(real_base, seed)
    V = build_vars(df, injected=(dataset == "injected"))
    disc, contribs, timings = {}, {}, {}
    pc_path = os.path.join(CFG["CACHE_DIR"], f"{dataset}_seed{seed}_pcmci_{CACHE_TAG}.pkl")   # PCMCI+ is the slow step: cached alone
    t = time.time()
    if os.path.exists(pc_path):
        pc = pickle.load(open(pc_path, "rb"))
    else:
        pc = run_pcmciplus(V, CFG["PC_ALPHA"]); pickle.dump(pc, open(pc_path, "wb"))
    timings["PCMCI+"] = time.time() - t
    disc["PCMCI+"] = pc["parents"]
    t = time.time(); disc["LPCMCI (daily)"] = run_lpcmci_daily(V, CFG["PC_ALPHA"]); timings["LPCMCI"] = time.time() - t
    t = time.time(); disc["NOTEARS"] = run_notears(df, V, seed); timings["NOTEARS"] = time.time() - t
    t = time.time(); disc["Target 2-test"] = run_target_selection(V, CFG["PC_ALPHA"]); timings["Target"] = time.time() - t
    t = time.time(); screen, screen_p = hac_wald_screen(V, df, CFG["PC_ALPHA"]); timings["HAC screen"] = time.time() - t
    disc["Exogenous HAC screen"] = {c for c in V.columns if GROUP.get(c) in screen}
    exo = set(exog_of(V.columns))
    disc["PCMCI+ dynamics + HAC exogenous"] = {p for p in pc["parents"] if p not in exo} | disc["Exogenous HAC screen"]
    P = disc["PCMCI+ dynamics + HAC exogenous"]
    t = time.time(); contribs["Graph + spline GLS"] = est_graph_ols(V, df, P); timings["OLS"] = time.time() - t
    contribs["Graph + spline OLS (month/year FE)"] = est_graph_ols(V, df, P, method="ols_fe")
    t = time.time(); contribs["Graph + g-computation (trees)"] = est_gcomputation(V, df, P, seed, CFG["N_GCOMP"], CFG["N_GCOMP_BG"]); timings["gcomp"] = time.time() - t
    t = time.time(); contribs["Double ML (cross-fitted)"] = est_dml(V, df, P, seed); timings["DML"] = time.time() - t
    contribs["Shortcut regression with demand history"] = est_graph_ols(V, df, P, add_history=True)
    t = time.time(); shp, mae = forecaster_shapley(df, V, seed, CFG["N_EXPLAIN"], CFG["N_BG"]); timings["Shapley"] = time.time() - t
    contribs.update(shp)
    res = dict(dataset=dataset, seed=seed, disc=disc, pc=pc, screen_p=screen_p,
               scores={k: score_contrib(v, df, dataset) for k, v in contribs.items()},
               effect_size={k: {g: float(np.subtract(*np.percentile(s.dropna(), [99.5, 0.5]))) for g, s in v.items()}
                            for k, v in contribs.items()},              # range of the effect (0.5-99.5 percentile)
               forecaster_mae=mae, timings=timings)
    pickle.dump(res, open(path, "wb"))
    print(f"  {dataset} seed {seed}: {time.time()-t0:.0f}s  " + ", ".join(f"{k} {v:.0f}s" for k, v in timings.items()))
    return res

RUNS = []
if CFG["RUN_SYNTHETIC"]:
    for s in CFG["SEEDS"]:
        RUNS.append(run_one("synthetic", s))
if CFG["RUN_INJECTED"]:
    REAL = load_kaggle(CFG["REAL_ENERGY_CSV"], CFG["REAL_WEATHER_CSV"])
    print("real data:", REAL.shape, REAL.index.min(), "->", REAL.index.max())
    for s in CFG["SEEDS"]:
        RUNS.append(run_one("injected", s, REAL))
print(len(RUNS), "runs")

# %% [markdown]
# ## 8. Which causal discovery method finds the right causes?

# %%
for ds in ["synthetic", "injected"]:
    rr = [r for r in RUNS if r["dataset"] == ds]
    if not rr: continue
    rows = [dict(method=m, seed=r["seed"], **score_discovery(r["disc"][m], ds)) for r in rr for m in r["disc"]]
    D = pd.DataFrame(rows); num = D.select_dtypes("number").columns.drop("seed")
    tab = D.groupby("method")[list(num)].mean().round(2)
    tab["false parents (any seed)"] = D.groupby("method")["false_parents"].agg(lambda s: ",".join(sorted(set(",".join(s).split(",")) - {"-", ""})) or "-")
    print(f"\n=== {ds}: discovery, mean over {len(rr)} seeds ===")
    print(tab.to_string())

# %% [markdown]
# ## 9. How large are the effects? Causal estimators vs SHAP-type explanations
# Mean ± standard deviation over seeds. b = 1: the full true effect is credited; magnitude correctness = 1 − mean |1 − b|. Paper rows are the published TFT-SHAP and XGBoost-TreeSHAP five-seed means.

# %%
SUMMARY = {}
for ds in ["synthetic", "injected"]:
    rr = [r for r in RUNS if r["dataset"] == ds]
    if not rr: continue
    rows = [dict(method=m, seed=r["seed"], **sc) for r in rr for m, sc in r["scores"].items()]
    D = pd.DataFrame(rows)
    cols = list(TRUTH[ds]["scored"]) + (["driver (driver-only part)", "decoy credit", "magnitude correctness (identifiable)"]
                                         if ds == "injected" else []) + ["magnitude correctness"]
    mean = D.groupby("method")[cols].mean(); sd = D.groupby("method")[cols].std().fillna(0)
    tab = mean.round(2).astype(str) + " ± " + sd.round(2).astype(str)
    tab = tab.loc[mean.sort_values("magnitude correctness", ascending=False).index]
    pat = D.groupby("method")[[f"r {g}" for g in TRUTH[ds]["scored"]]].mean().round(2)
    SUMMARY[ds] = dict(mean=mean, sd=sd, D=D, pattern=pat)
    print(f"\n=== {ds}: effect size (b), {len(rr)} seeds ===")
    print(tab.to_string())
    print("\npattern r (mean):"); print(pat.loc[tab.index].to_string())
    print("\npaper reference:"); print(PAPER[ds].to_string())
    fmae = np.mean([r["forecaster_mae"] for r in rr]); print(f"\nforecaster MAE (2018), mean over seeds: {fmae:.0f} MW")

# %%
fig, axes = plt.subplots(1, len(SUMMARY), figsize=(7.5 * len(SUMMARY), 4.6), squeeze=False)
for ax, (ds, S) in zip(axes[0], SUMMARY.items()):
    mean = pd.concat([S["mean"][list(TRUTH[ds]["scored"])], PAPER[ds][list(TRUTH[ds]["scored"])]])
    sd = pd.concat([S["sd"][list(TRUTH[ds]["scored"])], PAPER[ds][list(TRUTH[ds]["scored"])] * 0])
    order = mean.mean(1).sort_values().index
    markers = ["o", "s", "^", "D"]; colors = ["#e76f51", "#2a9d8f", "#457b9d", "#f4a261"]
    for k, g in enumerate(TRUTH[ds]["scored"]):
        y = np.arange(len(order)) + (k - (len(TRUTH[ds]["scored"]) - 1) / 2) * 0.13
        ax.errorbar(mean.loc[order, g], y, xerr=sd.loc[order, g], fmt=markers[k], color=colors[k], ms=6, capsize=2, label=g)
    ax.axvline(1, c="k", ls="--", lw=1); ax.set_yticks(range(len(order))); ax.set_yticklabels(order, fontsize=8)
    for y_ in range(len(order)): ax.axhline(y_, c="#eee", lw=0.6, zorder=0)
    ax.set_xlabel("slope b on the true effect (1 = full effect)")
    ax.set_title({"synthetic": "Synthetic benchmark", "injected": "Real data, injected effects"}[ds])
    ax.legend(fontsize=8, loc="lower right")
plt.tight_layout(); plt.show()

# %% [markdown]
# ### 9b. Real (not injected) drivers: a model-free check
# Injected effects act instantly (a step at 08:00, a driver that acts in the same hour). Real effects build up: buildings respond to temperature with delay, and the Sunday dip ramps in over the night. An estimator can recover injected effects and still misjudge real ones, so every causal estimator is also checked on a real effect that can be measured **without a model**: the Sunday drop, Sunday minus Tuesday–Thursday of the same week (weeks without holidays). Estimators are fitted on the real series without injection.
#
# The temperature sensitivity is reported, but not used as a check: it has no model-free estimate here, and the estimates move from about 35 to 320 MW/°C with the specification (with or without delayed temperature, month controls), so the real temperature effect is **not reliably identified** by any of these methods on this data.

# %%
CAUSAL_ESTS = ["Graph + spline GLS", "Graph + spline OLS (month/year FE)"]
REAL_CHECK = None
if CFG["RUN_INJECTED"]:
    Vr, dr = build_vars(REAL), REAL
    d = dr[["demand", "holiday"]].resample("D").mean(); d["dow"] = d.index.dayofweek; d["week"] = d.index.to_period("W")
    wk = d[~d["week"].isin(set(d.loc[d["holiday"] > 0, "week"]))]
    nat_sun = (wk[wk.dow == 6].groupby("week")["demand"].mean() - wk[wk.dow.isin([1, 2, 3])].groupby("week")["demand"].mean()).dropna()
    nat_lo, nat_hi = nat_sun.mean() - 1.96 * nat_sun.std() / np.sqrt(len(nat_sun)), nat_sun.mean() + 1.96 * nat_sun.std() / np.sqrt(len(nat_sun))
    # the Sunday effect relative to Tue-Thu in each model: Sunday contribution minus the mean contribution on Tue-Thu
    rows = []
    P_real = {c for c in Vr.columns if c in ("demand",) or GROUP.get(c) in ("hour", "week", "holiday", "temp")}
    for name, meth in [("Graph + spline GLS", "gls"), ("Graph + spline OLS (month/year FE)", "ols_fe")]:
        c = est_graph_ols(Vr, dr, P_real, method=meth)
        wkc = c["week"].resample("D").mean(); dow = wkc.index.dayofweek
        sunday = wkc[dow == 6].mean() - wkc[(dow >= 1) & (dow <= 3)].mean()
        td = pd.DataFrame({"c": c["temp"], "t": dr["temp"]}).resample("D").mean()
        cool = np.polyfit(td.t[td.t > 24], td.c[td.t > 24], 1)[0]; heat = -np.polyfit(td.t[td.t < 12], td.c[td.t < 12], 1)[0]
        rows.append(dict(estimator=name, **{"Sunday effect (MW)": sunday, "passes Sunday check": nat_lo * 1.15 <= sunday <= nat_hi * 0.85,
                                           "cooling (MW/°C)": cool, "heating (MW/°C)": heat}))
    REAL_CHECK = pd.DataFrame(rows).set_index("estimator")
    print(f"Model-free Sunday drop: {nat_sun.mean():.0f} MW (95% interval {nat_lo:.0f} to {nat_hi:.0f}, {len(nat_sun)} weeks); "
          "paper natural experiment (test year): -3303 MW")
    print("A model passes if its Sunday effect lies within the interval widened by 15%.\n")
    print(REAL_CHECK.round(0).to_string())

# %% [markdown]
# ### 9c. Per-seed results and paired comparison
# On real data each seed draws new event days and a new driver/decoy path, so the spread over seeds is the sampling uncertainty of the injected-truth test (60 event days against strongly day-correlated noise: about ±0.05–0.1 on the event slope). With five seeds the smallest two-sided Wilcoxon p is 0.0625, so "better in k of 5 seeds" is reported. On real data, the score that decides is **magnitude correctness (identifiable)**.

# %%
SCORE = {"synthetic": "magnitude correctness", "injected": "magnitude correctness (identifiable)"}
for ds, S in SUMMARY.items():
    col = SCORE[ds]
    mc = S["D"].pivot(index="method", columns="seed", values=col).round(2)
    mc["mean"] = mc.mean(1).round(2); mc["min"] = mc.drop(columns="mean").min(1)
    print(f"\n=== {ds}: {col} per seed ===\n" + mc.sort_values("mean", ascending=False).to_string())

# %% [markdown]
# ## 10. The best causal model, and its causal graph
# **Selection rule (fixed before looking at the final scores):** among the causal estimators, keep those that pass the model-free Sunday check on real data, then take the highest mean magnitude correctness (identifiable on real data, standard on the benchmark). Graph structure: dynamics from PCMCI+, exogenous causes from the HAC screen (last seed). Edge labels: the size of each effect (range over hours, 0.5th–99.5th percentile, MW) and, where the truth is known, the slope b. Exogenous edges that PCMCI+ found but the screen rejected are dotted grey; on real data the calendar and weather sizes are estimates whose truth is unknown.

# %%
_have = set.intersection(*[set(S["mean"].index) for S in SUMMARY.values()])
eligible = [e for e in CAUSAL_ESTS if e in _have and (REAL_CHECK is None or bool(REAL_CHECK.loc[e, "passes Sunday check"]))]
eligible = eligible or [e for e in CAUSAL_ESTS if e in _have]
BEST = max(eligible, key=lambda e: SUMMARY["injected"]["mean"].loc[e, SCORE["injected"]] if "injected" in SUMMARY
           else SUMMARY["synthetic"]["mean"].loc[e, "magnitude correctness"])
print(f"Eligible after the real-data check: {eligible}\nBest causal model: {BEST}\n")
for ds, S in SUMMARY.items():
    D = S["D"]; col = SCORE[ds]; mc = D.pivot(index="seed", columns="method", values=col)
    print(f"{ds}: {BEST} {col} = {mc[BEST].mean():.2f} ± {mc[BEST].std():.2f}")
    for ref in ["marginal Shapley (≈TreeSHAP)", "causal asymmetric Shapley", "Graph + spline GLS", "Double ML (cross-fitted)"]:
        if ref != BEST and ref in mc:
            diff = mc[BEST] - mc[ref]
            print(f"   vs {ref}: {mc[ref].mean():.2f}; better in {(diff > 0).sum()}/{len(diff)} seeds, mean gain {diff.mean():+.2f}")
    pt = PAPER[ds]
    if ds == "injected":
        print(f"   paper TFT-SHAP (published): event b {pt.loc['TFT, SHAP (paper)', 'event']:.2f}, full-effect MC "
              f"{pt.loc['TFT, SHAP (paper)', 'magnitude correctness']:.2f}; {BEST}: event b {S['mean'].loc[BEST, 'event']:.2f}")
    else:
        print(f"   paper TFT-SHAP (published) MC {pt.loc['TFT, SHAP (paper)', 'magnitude correctness']:.2f}")

# %%
ROLE_COLOR = {"cause": "#2a9d8f", "history": "#457b9d", "consequence": "#e76f51", "irrelevant": "#adb5bd",
              "hidden": "#ffffff", "injected": "#9b5de5", "unknown": "#e9c46a"}


def draw_graph(ax, run, best, dataset, title):
    from matplotlib.patches import FancyArrowPatch
    pc = run["pc"]; names = pc["names"]
    nodes = ["hour", "week", "holiday", "temp", "season", "other weather", "solar", "gas", "history"]
    if dataset == "injected": nodes = ["event", "driver", "decoy", "hour", "week", "holiday", "history", "temp", "season",
                                       "other weather", "solar", "gas"]
    else: nodes = nodes + ["hidden driver"]
    ang = np.linspace(90, 450, len(nodes), endpoint=False); R = 1.08
    pos = {n: (R * np.cos(np.radians(a)), R * np.sin(np.radians(a))) for n, a in zip(nodes, ang)}
    def role(n):
        if dataset == "synthetic":
            return {"history": "history", "gas": "consequence", "hidden driver": "hidden"}.get(
                n, "cause" if n in TRUTH["synthetic"]["causes"] else "irrelevant")
        return {"event": "injected", "driver": "injected", "decoy": "irrelevant", "history": "history",
                "gas": "consequence"}.get(n, "unknown")
    ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.42, 1.42); ax.set_aspect("equal"); ax.axis("off"); ax.set_title(title, fontsize=11)
    ax.add_patch(plt.Circle((0, 0), 0.22, color="#264653", zorder=3))
    ax.text(0, 0, "demand", ha="center", va="center", color="white", fontsize=9, fontweight="bold", zorder=4)
    for n in nodes:
        x, y = pos[n]; rl = role(n)
        ax.add_patch(plt.Circle((x, y), 0.18, facecolor=ROLE_COLOR[rl], edgecolor="#333", ls="--" if rl == "hidden" else "-", zorder=3))
        ax.text(x, y, n.replace(" ", "\n"), ha="center", va="center", fontsize=7.5, zorder=4,
                color="black" if rl in ("irrelevant", "hidden", "unknown") else "white")
    agg = {}
    for n, tau, e, v in pc["edges"]:
        g = GROUP[n]; d = agg.setdefault(g, dict(e=set(), lags=set(), v=0.0, items=[]))
        d["items"].append((tau, e, v))
        if g in ("gas", "solar"):                        # orientation from the same-hour edge if there is one
            e0 = [x for t, x, _ in d["items"] if t == 0]
            d["e"] = set(e0) if e0 else {x for _, x, _ in d["items"]}
        else:
            d["e"].add(e)
        d["lags"].add(tau); d["v"] = max(d["v"], abs(v))
    final = {GROUP.get(p, p) for p in run["disc"]["PCMCI+ dynamics + HAC exogenous"]}
    for g in list(agg):
        if g not in final and g not in ("history", "gas", "solar"):
            agg[g]["e"] = {"rejected"}
    for g in run["disc"].get("Exogenous HAC screen", set()):
        gg = GROUP.get(g, g)
        if gg not in agg:
            agg[gg] = dict(e={"screen"}, lags={0}, v=0.3)
    sizes = run["effect_size"][best]; scores = run["scores"][best]
    def arrow(p, q, color, lw, ls="-", head=True):
        p, q = np.array(p, float), np.array(q, float); u = (q - p) / np.linalg.norm(q - p)
        rp = 0.22 if np.allclose(p, 0) else 0.18; rq = 0.22 if np.allclose(q, 0) else 0.18
        ax.add_patch(FancyArrowPatch(p + u * (rp + .02), q - u * (rq + .02), arrowstyle="-|>" if head else "-",
                                     mutation_scale=16, color=color, lw=lw, ls=ls, zorder=2))
    for g, d in agg.items():
        if g not in pos: continue
        p = np.array(pos[g]); lw = 1 + 6 * min(d["v"], 0.8)
        if "<--" in d["e"]:
            arrow((0, 0), p, "#e76f51", lw)
        elif d["e"] & {"-->", "x->", "screen"}:
            arrow(p, (0, 0), "#1d3557" if g != "history" else "#457b9d", lw)
        else:                                           # x-x / o-o: linked, orientation not determined
            arrow(p, (0, 0), "#6c757d", lw, ls=":", head=False)
        lab = []
        if d["e"] == {"screen"}: lab.append("HAC screen")
        if d["e"] == {"rejected"}: lab.append("rejected by screen")
        if g in ("gas", "solar"):
            lab = [f"lag {-t if t else 0}: {x} ({vv:+.2f})" for t, x, vv in sorted(d["items"])]
        if g in sizes: lab.append(f"range {sizes[g]:,.0f} MW")
        if g in scores: lab.append(f"b={scores[g]:.2f}")
        if not lab:
            lags = sorted(d["lags"])
            lab.append(f"lags {lags[0]}–{lags[-1]} h" if len(lags) > 2 else "lag " + ",".join(str(-l) if l else "0" for l in lags))
        mid = 0.56 * p + 0.1 * np.array([-p[1], p[0]])
        ax.text(mid[0], mid[1], "\n".join(lab), fontsize=6.8, ha="center", va="center",
                bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.6), zorder=5)
    if dataset == "synthetic":
        arrow(pos["hidden driver"], (0, 0), "#999", 1.2, ls="--"); arrow(pos["hidden driver"], pos["gas"], "#999", 1.2, ls="--")
    legend = [("cause", "true cause"), ("history", "demand history"), ("consequence", "consequence"),
              ("irrelevant", "no effect / decoy")] + ([("injected", "injected cause"), ("unknown", "real, truth unknown")]
                                                       if dataset == "injected" else [("hidden", "hidden (true SCM)")])
    return [plt.Line2D([], [], marker="o", ls="", ms=9, mfc=ROLE_COLOR[r], mec="#333", label=l) for r, l in legend]


for ds, S in SUMMARY.items():
    run = [r for r in RUNS if r["dataset"] == ds][-1]
    fig, ax = plt.subplots(figsize=(8.5, 8))
    h = draw_graph(ax, run, BEST, ds, f"{'Synthetic benchmark' if ds == 'synthetic' else 'Real data, injected effects'}: "
                                    f"final graph, effects by {BEST} (seed {run['seed']})")
    ax.legend(handles=h, loc="lower center", ncol=3, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.06))
    plt.tight_layout(); plt.show()
    if ds == "injected":
        lag_gen = sorted({(n, -t) for n, t, e, v in run["pc"]["edges"] if n in GEN and e in ("-->", "x->") and t > 0})
        print("Reading the real-data graph: the injected event and driver are exact tests (b shown). Calendar, temperature and other\n"
              "weather are real causes whose sizes are estimates (truth unknown). Gas: strongly linked to demand in the same hour,\n"
              "orientation not determined by PCMCI+ on real data (x-x: conflicting orientation tests; on the benchmark it is\n"
              f"correctly oriented demand -> gas). Weak lagged generation edges {lag_gen} (partial correlation about 0.05) are most\n"
              "likely anticipation, not causes. Generation is kept out of the effect estimation.")

# %% [markdown]
# ## 11. Explaining the GLS model
# The best causal model is additive, so its explanation is exact: for every hour, each driver's term is its contribution in MW, and the terms sum to the fitted demand. For an additive model these terms are also its exact interventional SHAP values, so the built-in explanation and SHAP coincide. Three figures on the real data with injected effects (seed 0):
#
# 1. **One day, decomposed**: an event day and a normal weekday of the same week.
# 2. **The shape of each effect, with 95% bands** (Newey–West standard errors on the prewhitened model).
# 3. **GLS vs the forecaster's Shapley explanation**, on the same hours, against the injected truth.

# %%
VIZ = dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", grid="#e6e5e1",
           c1="#2a78d6", c2="#eb6834", c3="#1baf7a", c4="#eda100", c5="#e87ba4")
plt.rcParams.update({"axes.facecolor": VIZ["surface"], "figure.facecolor": VIZ["surface"], "axes.edgecolor": VIZ["grid"],
                     "axes.labelcolor": VIZ["ink2"], "xtick.color": VIZ["ink2"], "ytick.color": VIZ["ink2"],
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": VIZ["grid"],
                     "grid.linewidth": 0.6, "axes.titlesize": 11, "axes.titlecolor": VIZ["ink"], "lines.linewidth": 2})


def gls_explain(V, df, parents, bandwidth=48):
    """Refit the best model and return its terms, fitted demand and HAC covariance of the coefficients."""
    groups = treatment_groups(parents)
    Bs = {g: basis(V, df, g) for g in groups if g not in ("hour", "week")}
    Bs["weekly profile"] = pd.get_dummies(df.index.dayofweek * 24 + df.index.hour, prefix="how",
                                          drop_first=True).set_index(V.index).astype(float)
    names = list(Bs); X = pd.concat([Bs[g] for g in names], axis=1)
    Xm = np.column_stack([np.ones(len(X)), X.values]); y = V["demand"].values
    beta, Xs, e = gls_ar(Xm, y)
    Xe = Xs * e[:, None]; S = Xe.T @ Xe
    for l in range(1, bandwidth + 1):
        G_ = Xe[l:].T @ Xe[:-l]; S += (1 - l / (bandwidth + 1)) * (G_ + G_.T)
    inv = np.linalg.pinv(Xs.T @ Xs); cov = inv @ S @ inv
    cols = ["const"] + list(X.columns); ix = {c: i for i, c in enumerate(cols)}
    terms = {g: pd.Series(Bs[g].values @ beta[[ix[c] for c in Bs[g].columns]], index=V.index) for g in names}
    fitted = pd.Series(Xm @ beta, index=V.index)
    return dict(terms=terms, fitted=fitted, beta=beta, cov=cov, ix=ix, Bs=Bs)


def band(B, ex, k=1.96):
    """Fitted curve and 95% band for one term: B (rows x basis cols) times its coefficients."""
    idx = [ex["ix"][c] for c in B.columns]; b = ex["beta"][idx]; C = ex["cov"][np.ix_(idx, idx)]
    f = B.values @ b; se = np.sqrt(np.einsum("ij,jk,ik->i", B.values, C, B.values))
    return f, f - k * se, f + k * se

if CFG["RUN_INJECTED"]:
    run0 = [r for r in RUNS if r["dataset"] == "injected" and r["seed"] == 0][0]
    df0 = inject_effects(REAL, 0); V0 = build_vars(df0, injected=True)
    EX = gls_explain(V0, df0, run0["disc"]["PCMCI+ dynamics + HAC exogenous"])
    print("terms:", list(EX["terms"]))

# %% [markdown]
# ### Figure 1. One day, decomposed
# Top: real demand (with the injected effects) and the GLS fit from the driver terms alone. Bottom: each driver's term in MW, centred on its average, so a term above 0 pushes that hour above normal; dashed lines are the injected truth. The gap between demand and fit is what no driver term explains; GLS models it as autocorrelated error, and it is drawn as its own dotted line, so demand = average + driver terms + unexplained part, exactly. It is not only slow drift: in winter it peaks at 08:00 and 19:00, because the weekly profile is the same all year while the real shape of the day changes with the season (lighting, heating). That does not bias the injected effects, which are random, but on real data it means part of demand is credited to no driver. A season-dependent profile (hour-of-week × month) is the natural next refinement.

# %%
if CFG["RUN_INJECTED"]:
    t18 = df0.index >= pd.Timestamp("2018-01-01")
    ev_days = pd.Index(df0.index[t18 & (df0["event_flag"] > 0)].normalize().unique())
    ev_day = next(d for d in ev_days if d.dayofweek in (1, 2, 3))
    week = pd.date_range(ev_day - pd.Timedelta(days=ev_day.dayofweek), periods=7, freq="D")
    calm = [d for d in week if d.dayofweek in (1, 2, 3) and d != ev_day and d not in ev_days and df0.loc[str(d.date()), "holiday"].max() == 0]
    days = [ev_day] + calm[:1]
    show = [("weekly profile", "Weekly profile", VIZ["c1"]), ("temp", "Temperature", VIZ["c2"]),
            ("event", "Event", VIZ["c3"]), ("driver", "Driver", VIZ["c4"]), ("other weather", "Other weather", VIZ["c5"])]
    show = [s for s in show if s[0] in EX["terms"]]
    centred = {g: EX["terms"][g] - EX["terms"][g].mean() for g, _, _ in show}
    resid = df0["demand"] - EX["fitted"]; centred["drift"] = resid - resid.mean()
    show = show + [("drift", "Unexplained: seasonal day shape + slow drift", VIZ["ink2"])]
    truth = {"event": df0["C_event"] - df0["C_event"].mean(), "driver": df0["C_driver"]}
    fig, axes = plt.subplots(2, len(days), figsize=(13, 7.2), sharex="col", gridspec_kw=dict(height_ratios=[1, 1.35]))
    axes = np.atleast_2d(axes).reshape(2, len(days))
    lo = min(centred[g].loc[str(d.date())].min() for g, _, _ in show for d in days)
    hi = max(centred[g].loc[str(d.date())].max() for g, _, _ in show for d in days)
    for j, d in enumerate(days):
        sl = str(d.date()); hrs = np.arange(24)
        a = axes[0, j]
        a.plot(hrs, df0.loc[sl, "demand"].values, color=VIZ["ink"], lw=2, label="demand (real + injected)")
        a.plot(hrs, EX["fitted"].loc[sl].values, color=VIZ["c1"], lw=2, ls=(0, (4, 2)), label="GLS fit")
        a.set_title(("Event day" if j == 0 else "Normal weekday") + f", {d:%a %d %b %Y}")
        a.set_ylabel("MW"); a.legend(fontsize=8, frameon=False, loc="upper left")
        b = axes[1, j]
        b.axhline(0, color=VIZ["ink2"], lw=0.8)
        for g, lab, col in show:
            yv = centred[g].loc[sl].values
            b.plot(hrs, yv, color=col, lw=2 if g != "drift" else 1.5, ls="-" if g != "drift" else (0, (1, 1.5)), label=lab)
            if g in truth:
                b.plot(hrs, truth[g].loc[sl].values, color=col, lw=1.2, ls=(0, (3, 2)))
        b.set_ylim(lo - 300, hi + 300); b.set_xlim(0, 23); b.set_xticks(range(0, 24, 3))
        b.set_xlabel("hour of day"); b.set_ylabel("contribution (MW)")
    h = [plt.Line2D([], [], color=c, lw=2 if g != "drift" else 1.5, ls="-" if g != "drift" else (0, (1, 1.5)), label=l)
         for g, l, c in show] + [plt.Line2D([], [], color=VIZ["ink2"], lw=1.2, ls=(0, (3, 2)), label="injected truth")]
    fig.legend(handles=h, fontsize=8, frameon=False, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Each hour splits exactly into driver terms plus an unexplained part the model treats as error",
                 x=0.01, ha="left", fontsize=12, color=VIZ["ink"])
    plt.tight_layout(rect=(0, 0.06, 1, 1)); plt.show()
    j_ev = df0.loc[str(ev_day.date())]
    print(f"Event day {ev_day:%Y-%m-%d}: credited event effect {EX['terms']['event'].loc[str(ev_day.date())][j_ev['event_flag'] > 0].mean():.0f} MW "
          f"in event hours (truth -1500 MW).")

# %% [markdown]
# ### Figure 2. The shape of each effect, with 95% bands
# (a) Temperature, (b) the weekly profile over the 168 hours of the week, (c) the injected event per seed with its 95% interval against the true −1,500 MW, and (d) the driver: estimated vs true contribution (driver-only part), with the identity line. Bands are Newey–West (48 h) intervals on the prewhitened model; they are narrow because the data are 35,000 hours, and they cover estimation noise only, not a wrong model shape.

# %%
if CFG["RUN_INJECTED"]:
    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    # (a) temperature
    Tg = np.linspace(np.percentile(df0["temp"], 0.5), np.percentile(df0["temp"], 99.5), 200)
    Bt = pd.DataFrame({f"tk{k}": np.clip(Tg - k, 0, None) for k in TEMP_KNOTS}); Bt["tlin"] = Tg
    f, l, u = band(Bt[EX["Bs"]["temp"].columns], EX); m0 = np.interp(np.median(df0["temp"]), Tg, f)
    a = ax[0, 0]; a.fill_between(Tg, l - m0, u - m0, color=VIZ["c2"], alpha=0.2, lw=0); a.plot(Tg, f - m0, color=VIZ["c2"])
    a.set_xlabel("temperature (°C)"); a.set_ylabel("effect on demand (MW, 0 at median temperature)")
    a.set_title("(a) Temperature: real effect, truth unknown")
    # (b) weekly profile
    Bw = EX["Bs"]["weekly profile"]; how = df0.index.dayofweek * 24 + df0.index.hour
    first = pd.Series(np.arange(len(how)), index=how).groupby(level=0).first().sort_index()
    f, l, u = band(Bw.iloc[first.values], EX); f, l, u = f - f.mean(), l - f.mean(), u - f.mean()
    a = ax[0, 1]; hh = np.arange(168)
    a.fill_between(hh, l, u, color=VIZ["c1"], alpha=0.2, lw=0); a.plot(hh, f, color=VIZ["c1"])
    a.set_xticks(range(12, 168, 24)); a.set_xticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    for x0 in range(24, 168, 24): a.axvline(x0, color=VIZ["grid"], lw=0.8)
    a.set_ylabel("MW, relative to the weekly mean"); a.set_title("(b) Weekly profile: real effect, truth unknown")
    # (c) event per seed
    a = ax[1, 0]; rows = []
    for r in sorted([r for r in RUNS if r["dataset"] == "injected"], key=lambda r: r["seed"]):
        dfs = inject_effects(REAL, r["seed"]); Vs = build_vars(dfs, injected=True)
        ex = gls_explain(Vs, dfs, r["disc"]["PCMCI+ dynamics + HAC exogenous"])
        i = ex["ix"]["event_flag"]; rows.append((r["seed"], ex["beta"][i], 1.96 * np.sqrt(ex["cov"][i, i])))
    for s_, b_, ci in rows:
        a.errorbar(b_, s_, xerr=ci, fmt="o", color=VIZ["c3"], ms=8, capsize=3, lw=2)
    a.axvline(-1500, color=VIZ["ink"], ls=(0, (4, 2)), lw=1.2); a.annotate("truth −1,500 MW", (-1500, 4.35), xytext=(4, 0), textcoords="offset points", fontsize=8, color=VIZ["ink2"])
    a.set_yticks([s_ for s_, _, _ in rows]); a.set_yticklabels([f"seed {s_}" for s_, _, _ in rows]); a.set_ylim(-0.6, 4.7)
    a.set_xlabel("estimated event effect (MW), 95% interval"); a.set_title("(c) Injected event: estimate per seed")
    # (d) driver
    a = ax[1, 1]; tt = df0["C_driver_only"][t18]; ee = EX["terms"]["driver"][t18]; ee = ee - ee.mean()
    a.scatter(tt, ee, s=8, color=VIZ["c4"], alpha=0.35, lw=0)
    lim = [min(tt.min(), ee.min()), max(tt.max(), ee.max())]; a.plot(lim, lim, color=VIZ["ink"], ls=(0, (4, 2)), lw=1.2)
    bslope = np.polyfit(tt - tt.mean(), ee, 1)[0]
    a.annotate(f"slope b = {bslope:.2f}\n(1 = correct size)", (0.04, 0.9), xycoords="axes fraction", fontsize=9, color=VIZ["ink"], va="top")
    a.set_xlabel("true driver contribution, driver-only part (MW)"); a.set_ylabel("GLS driver term (MW)")
    a.set_title("(d) Injected driver: estimated vs true, 2018 hours")
    fig.suptitle("Each effect has a shape and an uncertainty; the injected effects are recovered", x=0.01, ha="left", fontsize=12, color=VIZ["ink"])
    plt.tight_layout(); plt.show()
    print("Event estimates (MW, 95% interval):", [f"seed {s_}: {b_:.0f} ± {ci:.0f}" for s_, b_, ci in rows])

# %% [markdown]
# ### Figure 3. GLS vs the forecaster's Shapley explanation, same hours
# The same test hours explained two ways (all 2018 event hours plus as many random normal hours): the GLS terms, and marginal Shapley values (≈ interventional TreeSHAP) of the boosted-tree forecaster. (a) Event: credit in event hours relative to the same method's credit in non-event hours, against the true −1,500 MW. (b) Driver: credit against the true contribution, identity line dashed.

# %%
if CFG["RUN_INJECTED"]:
    te_idx = df0.index[(df0.index >= pd.Timestamp("2018-01-01")) & (df0.index <= df0.index.max() - pd.Timedelta(hours=1))]
    ev_hours = te_idx[df0.loc[te_idx, "event_flag"] > 0]
    calm_hours = pd.Index(np.random.default_rng(0).choice(te_idx[df0.loc[te_idx, "event_flag"] == 0], len(ev_hours), replace=False))
    shp, _ = forecaster_shapley(df0, V0, 0, None, CFG["N_BG"], explain_index=ev_hours.union(calm_hours))
    msh = shp["marginal Shapley (≈TreeSHAP)"]; E_idx = msh["event"].index
    ev = df0.loc[E_idx, "event_flag"] > 0
    gls_ev = EX["terms"]["event"].loc[E_idx]; sh_ev = msh["event"]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    a = ax[0]; rng = np.random.default_rng(0)
    for k, (lab, s, col) in enumerate([("GLS (causal model)", gls_ev, VIZ["c1"]), ("Shapley, forecaster", sh_ev, VIZ["c2"])]):
        rel = s - s[~ev].mean()
        a.scatter(rel[ev], k + rng.uniform(-0.15, 0.15, ev.sum()), s=22, color=col, alpha=0.8, lw=0, label=lab)
        a.annotate(f"mean {rel[ev].mean():,.0f} MW", (rel[ev].mean(), k + 0.3), fontsize=8, color=VIZ["ink2"], ha="center")
    a.axvline(-1500, color=VIZ["ink"], ls=(0, (4, 2)), lw=1.2)
    a.set_yticks([0, 1]); a.set_yticklabels(["GLS (causal model)", "Shapley, forecaster"]); a.set_ylim(-0.5, 1.6)
    a.set_xlabel("credit to the event in event hours (MW); truth −1,500 dashed"); a.set_title(f"(a) Event, {int(ev.sum())} event hours")
    a = ax[1]; tt = df0.loc[E_idx, "C_driver_only"]; tf = df0.loc[E_idx, "C_driver"]; slopes = {}
    for lab, s, col in [("GLS (causal model)", EX["terms"]["driver"].loc[E_idx], VIZ["c1"]), ("Shapley, forecaster", msh["driver"], VIZ["c2"])]:
        s = s - s.mean(); b_only = np.polyfit(tt - tt.mean(), s, 1)[0]; b_full = np.polyfit(tf - tf.mean(), s, 1)[0]
        slopes[lab] = (b_only, b_full)
        a.scatter(tt - tt.mean(), s, s=14, color=col, alpha=0.6, lw=0, label=f"{lab}: b = {b_only:.2f} (full effect {b_full:.2f})")
    lim = [tt.min() - tt.mean(), tt.max() - tt.mean()]; a.plot(lim, lim, color=VIZ["ink"], ls=(0, (4, 2)), lw=1.2)
    a.set_xlabel("true driver contribution, driver-only part (MW)"); a.set_ylabel("credit to the driver (MW)")
    a.set_title("(b) Driver"); a.legend(fontsize=8, frameon=False, loc="upper left")
    g_ev = (gls_ev - gls_ev[~ev].mean())[ev].mean(); s_ev = (sh_ev - sh_ev[~ev].mean())[ev].mean()
    fig.suptitle(f"Same hours, seed 0: event credited {g_ev:,.0f} MW by GLS and {s_ev:,.0f} MW by Shapley (truth −1,500); "
                 f"driver b {slopes['GLS (causal model)'][0]:.2f} vs {slopes['Shapley, forecaster'][0]:.2f}",
                 x=0.01, ha="left", fontsize=11, color=VIZ["ink"])
    plt.tight_layout(); plt.show()
    print("One seed only; across 5 seeds the Shapley driver slope varies widely (see Section 9: 0.76 ± 0.35 on the driver-only part).")


# ======================================================================================================================
# ==== PART 8: Part 8. Is past demand a shortcut? TFT with and without demand history, against the causal estimate
# ======================================================================================================================

# %% [markdown]
# # Part 8. Is past demand a shortcut? TFT with and without demand history, against the causal estimate
#
# The same compact TFT is trained with and without demand history and lagged generation. Both versions are explained on the same hours (SHAP by expected gradients, and the grouped Shapley value) and scored next to the causal graph + GLS estimate on those hours, with the same seeds and injected effects.
#
# **In the paper:** Section 6.10. Part 1 runs the same test for TFT, XGBoost and Ridge with SHAP. This part adds the grouped Shapley value and the causal estimate on the same hours. The summary ends with a convergence check of the real-data TFTs: the real-data lag test should only be reported if it passes.
#
# Both datasets (synthetic benchmark, real Spanish demand with injected effects), five seeds.
#
# Outputs in `results/demand_history_shortcut/`: `followup_results.csv`, `followup_summary.txt`, `followup_plot.png`.
#
# **Settings.** The plan (`smoke` or `deep`), the number of seeds and the training length come from the command line (`--plan`, `--seeds`, `--max-epochs`, ...). Everything else is set in the first lines of this part. Finished runs are saved and skipped when the part is run again.

# %%
# The launcher passes the shared settings as GLOBAL_SETTINGS (plan, seeds, training length, explained hours, search size).
import json
from pathlib import Path
print(GLOBAL_SETTINGS)

# %%
import os, sys, time, pickle, itertools, warnings
from math import factorial
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression
import torch, torch.nn as nn

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)

# %% [markdown]
# ## Settings

# %%
# ----------------------------------------------------------------------------- settings
OUT_DIR = "results/demand_history_shortcut"
CFG = dict(CACHE_DIR=OUT_DIR + "/cache", PC_ALPHA=0.01)
class ARGS:
    smoke = GLOBAL_SETTINGS["plan"] == "smoke"   # tiny run (1 seed, 2 epochs)
    synthetic_only = False   # True: skip the real data
    energy = "energy_dataset.csv"    # Kaggle files: found automatically in data/, or give a full path
    weather = "weather_features.csv"

TWO = dict(
    SEEDS=list(range(GLOBAL_SETTINGS["seeds"])),
    DATASETS=["synthetic"] if ARGS.synthetic_only else ["synthetic", "injected"],
    VARIANTS=["with lags", "no lags"],
    WINDOW=24, D_MODEL=32, N_HEADS=4, DROPOUT=0.1,
    EPOCHS=GLOBAL_SETTINGS["max_epochs"], PATIENCE=GLOBAL_SETTINGS["patience"], BATCH=256, LR=3e-4,      # slow, long training
    TRAIN_STRIDE=2,
    N_EXPLAIN=200, N_EVENT_EXPLAIN=60, N_BG=15,
    USE_SHAP_EG=True, N_EG_BG=200, EG_SAMPLES=200,
)
if ARGS.smoke:
    TWO.update(SEEDS=[0], EPOCHS=2, PATIENCE=1, TRAIN_STRIDE=8, N_EXPLAIN=40, N_EVENT_EXPLAIN=20, N_BG=5, N_EG_BG=20, EG_SAMPLES=20)
CACHE_TAG = "fu20261105" + ("_smoke" if ARGS.smoke else "")
os.makedirs(CFG["CACHE_DIR"], exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))

# Convergence checks for the real-data TFTs (results are only trusted if they pass)
CHECK = dict(MAX_MAE_NO_LAGS=1500.0, MIN_EVENT_SLOPE=0.6)


def _find(name, folders=(".", "Data", "data", os.path.join("..", "Data"), os.path.join("..", "data"))):
    if os.path.exists(name):
        return name
    for f in folders:
        cand = os.path.join(f, os.path.basename(name))
        if os.path.exists(cand):
            return cand
    return None

# %% [markdown]
# ## Functions: data, injection, causal screen + GLS, TFT, explainers, scoring
# Nothing to change here.

# %%
# ----------------------------------------------------------------------------- data
def spanish_holidays(years):
    """Spanish national holidays incl. Good Friday."""
    out = []
    for y in years:
        a = y % 19; b, c = divmod(y, 100); d, e = divmod(b, 4)
        f = (b + 8) // 25; g = (b - f + 1) // 3
        h = (19 * a + b - d - g + 15) % 30; i, k = divmod(c, 4)
        l = (32 + 2 * e + 2 * i - h - k) % 7; m = (a + 11 * h + 22 * l) // 451
        month = (h + l - 7 * m + 114) // 31; day = ((h + l - 7 * m + 114) % 31) + 1
        easter = pd.Timestamp(y, month, day)
        out += [easter - pd.Timedelta(days=2)]
        out += [pd.Timestamp(y, mo, dd) for mo, dd in
                [(1, 1), (1, 6), (5, 1), (8, 15), (10, 12), (11, 1), (12, 6), (12, 8), (12, 25)]]
    return pd.DatetimeIndex(out).normalize()


def ar1(n, phi, sd_stationary, rng):
    e = rng.normal(0, sd_stationary * np.sqrt(1 - phi ** 2), n)
    x = np.empty(n); x[0] = rng.normal(0, sd_stationary)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e[t]
    return x


def generate_benchmark(seed=0, variant="default", latent_strength=1.0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2015-01-01", "2018-12-31 23:00", freq="h")
    n = len(idx)
    hour = idx.hour.values; dow = idx.dayofweek.values; doy = idx.dayofyear.values
    tt = np.arange(n)

    # calendar components
    C_hour = 2600 * np.cos(2 * np.pi * (hour - 13) / 24) + 900 * np.cos(4 * np.pi * (hour - 20) / 24)
    C_week = np.where(dow == 5, -1800.0, np.where(dow == 6, -3200.0, 0.0))
    hol = np.isin(idx.normalize(), spanish_holidays(range(2015, 2019))).astype(float)
    C_holiday = -2500.0 * hol

    # weather: 5 cities, seasonal + daily + persistent anomaly, averaged
    common_anom = ar1(n, 0.998, 2.5, rng)
    temps = []
    for base, amp, damp in [(15, 10, 6), (16.5, 8, 4), (18, 8, 5), (19, 10, 7), (14.5, 7, 4)]:
        city = (base + amp * np.cos(2 * np.pi * (doy - 200) / 365.25)
                + damp * np.cos(2 * np.pi * (hour - 15) / 24)
                + common_anom + ar1(n, 0.99, 1.0, rng))
        temps.append(city)
    T = np.mean(temps, axis=0)
    clouds = 100 / (1 + np.exp(-ar1(n, 0.98, 1.5, rng)))
    humidity = np.clip(65 + 0.25 * (clouds - 50) - 1.2 * (T - 16) + rng.normal(0, 5, n), 5, 100)
    pressure = 1015 + ar1(n, 0.995, 6, rng)
    wind_speed = np.abs(3 + ar1(n, 0.97, 2.0, rng))

    # temperature response with regime change in cooling
    change = {"default": "2016-07-01", "extrapolation": "2017-07-01", "stationary": None}[variant]
    cool_mult = np.ones(n) if change is None else np.where(idx >= pd.Timestamp(change), 1.6, 1.0)
    C_temp = 250 * np.clip(16 - T, 0, None) + cool_mult * 150 * np.clip(T - 22, 0, None) ** 1.3

    # hidden driver (economic activity): daily AR(0.995) interpolated to hours
    n_days = n // 24 + 2
    lat_d = ar1(n_days, 0.995, 1.0, rng)
    latent = np.interp(tt / 24.0, np.arange(n_days), lat_d)
    C_latent = 1200 * latent_strength * latent
    C_AR = ar1(n, 0.95, 500, rng)
    eps = rng.normal(0, 150, n)
    D = 28000 + C_hour + C_week + C_holiday + C_temp + C_latent + C_AR + eps

    # generation, produced after demand
    sun = np.clip(np.cos(2 * np.pi * (hour - 13.5) / 24), 0, None) ** 1.5
    solar = 4500 * sun * (1 + 0.3 * np.cos(2 * np.pi * (doy - 172) / 365.25)) * (1 - 0.7 * clouds / 100)
    wind = 18000 * np.minimum(1, (wind_speed / 11) ** 3)
    nuclear = np.full(n, 7000.0)
    for _ in range(12):
        s = rng.integers(0, n - 720); nuclear[s:s + 720] -= 1000
    hydro = 3000 + 1500 * np.cos(2 * np.pi * (doy - 60) / 365.25) + ar1(n, 0.99, 300, rng)
    residual = D - solar - wind - nuclear - hydro
    gas = np.clip(0.55 * residual + 600 * latent_strength * latent + rng.normal(0, 200, n), 0, None)
    coal = np.clip(residual - gas, 0, None)

    df = pd.DataFrame(dict(
        time=idx, demand=D, temp=T, humidity=humidity, pressure=pressure, wind_speed=wind_speed,
        clouds=clouds, holiday=hol, gen_solar=solar, gen_wind=wind, gen_nuclear=nuclear,
        gen_hydro=hydro, gen_gas=gas, gen_coal=coal,
        C_hour=C_hour, C_week=C_week, C_holiday=C_holiday, C_temp=C_temp,
        C_latent=C_latent, C_AR=C_AR))
    return df.set_index("time")


def load_kaggle(energy_csv, weather_csv):
    """Kaggle 'Hourly energy demand generation and weather' (Spain 2015-2018), cleaned:
    impossible weather readings set to missing per city BEFORE averaging over the 5 cities, local Madrid time,
    every hour present (short gaps interpolated, never dropped: dropping a row would shift all lags after it)."""
    en = pd.read_csv(energy_csv)
    en["time"] = pd.to_datetime(en["time"], utc=True).dt.tz_convert("Europe/Madrid").dt.tz_localize(None)
    en = en.drop_duplicates("time").set_index("time")           # autumn DST: repeated local hour dropped
    w = pd.read_csv(weather_csv)
    w["time"] = pd.to_datetime(w["dt_iso"], utc=True).dt.tz_convert("Europe/Madrid").dt.tz_localize(None)
    w = w.drop_duplicates(["time", "city_name"])
    w.loc[~w["pressure"].between(900, 1100), "pressure"] = np.nan
    w.loc[w["wind_speed"] > 50, "wind_speed"] = np.nan
    w.loc[w["humidity"] <= 0, "humidity"] = np.nan
    wm = w.groupby("time")[["temp", "humidity", "pressure", "wind_speed", "clouds_all"]].mean()
    wm["temp"] = wm["temp"] - 273.15
    df = pd.DataFrame(index=en.index)
    df["demand"] = en["total load actual"]
    m = {"gen_solar": "generation solar", "gen_wind": "generation wind onshore", "gen_nuclear": "generation nuclear",
         "gen_hydro": "generation hydro water reservoir", "gen_gas": "generation fossil gas",
         "gen_coal": "generation fossil hard coal"}
    for k, c in m.items():
        df[k] = en[c]
    df.loc[(df["gen_nuclear"] == 0) | (df["gen_gas"] == 0), ["gen_nuclear", "gen_gas", "gen_solar"]] = np.nan  # 4 rows coded 0
    df = df.join(wm.rename(columns={"clouds_all": "clouds"}), how="left")
    df = df.asfreq("h")                                          # spring DST: missing local hour inserted
    n_missing = int(df["demand"].isna().sum())
    df = df.interpolate(method="time", limit=12, limit_direction="both")
    assert df.notna().all().all(), "gap longer than 12 h in the real data"
    assert (np.diff(df.index.values).astype("timedelta64[h]").astype(int) == 1).all()
    df["holiday"] = np.isin(df.index.normalize(), spanish_holidays(range(2015, 2019))).astype(float)
    print(f"real data: {len(df)} contiguous hours, {n_missing} missing demand hours interpolated")
    return df


DRIVER_PERSISTENCE = 0.95    # hourly AR coefficient of the injected driver (slow: memory of ~20 hours).
# At 0.99 the driver is as slow as the drift of real demand and no hourly method separates it from that drift.


def inject_effects(df, seed):
    """Event (-1,500 MW, 08-21h, 60 random weekdays, flagged), heat-dependent nonlinear driver (<= ~900 MW),
    decoy correlated 0.8 with the driver and without effect. Returns the modified frame and the true components."""
    rng = np.random.default_rng(1000 + seed)
    out = df.copy(); idx = out.index; n = len(out)
    days = pd.Index(idx.normalize().unique())
    weekdays = days[(days.dayofweek < 5) & ~days.isin(spanish_holidays(range(days.year.min(), days.year.max() + 1)))]
    ev_days = pd.DatetimeIndex(rng.choice(weekdays, 60, replace=False))
    flag = (idx.normalize().isin(ev_days) & (idx.hour >= 8) & (idx.hour <= 21)).astype(float)
    x = ar1(n, DRIVER_PERSISTENCE, 1.0, rng)                        # slow driver, unit variance
    heat = np.clip((out["temp"].values - 10) / 20, 0, 1)            # 0 below 10 °C, 1 above 30 °C
    drv_eff = 900.0 / (1 + np.exp(-1.5 * x)) * (0.3 + 0.7 * heat)   # nonlinear, grows with heat, 0 to 900 MW
    decoy = 0.8 * x + 0.6 * ar1(n, DRIVER_PERSISTENCE, 1.0, rng)    # corr ~0.8, same timescale, no effect
    out["event_flag"] = flag; out["driver"] = x; out["decoy"] = decoy
    out["C_event"] = -1500.0 * flag; out["C_driver"] = drv_eff - drv_eff.mean()
    # Part of the driver effect that does not depend on temperature alone: the driver's average effect grows with heat,
    # and that temperature-only part (about 30% of its variance) is shared with temperature by any method.
    e_logistic = np.mean(900.0 / (1 + np.exp(-1.5 * np.random.default_rng(0).normal(size=200000))))
    only = drv_eff - e_logistic * (0.3 + 0.7 * heat)
    out["C_driver_only"] = only - only.mean()
    out["demand"] = out["demand"] - 1500.0 * flag + drv_eff
    return out


# ----------------------------------------------------------------------------- variables
DEMAND_LAGS = [1, 2, 24, 48, 168]
GEN_LAGS = [1, 24]
CALENDAR = ["h_s1", "h_c1", "h_s2", "h_c2", "sat", "sun", "holiday"]
WEATHER = ["hdd", "cdd", "humidity", "pressure", "wind_speed", "clouds"]
GEN = ["gen_solar", "gen_gas"]
INJ = ["event_flag", "driver", "decoy"]
GROUP = {"demand": "history", "h_s1": "hour", "h_c1": "hour", "h_s2": "hour", "h_c2": "hour",
         "sat": "week", "sun": "week", "holiday": "holiday", "hdd": "temp", "cdd": "temp",
         "doy_s": "season", "doy_c": "season", "humidity": "other weather", "pressure": "other weather",
         "wind_speed": "other weather", "clouds": "other weather", "gen_solar": "solar", "gen_gas": "gas",
         "event_flag": "event", "driver": "driver", "decoy": "decoy"}
TRUTH = {
    "synthetic": dict(causes={"hour", "week", "holiday", "temp", "history"},
                      scored={"hour": "C_hour", "week": "C_week", "holiday": "C_holiday", "temp": "C_temp"},
                      consequences={"gas"}, irrelevant={"season", "other weather", "solar"}),
    "injected": dict(causes={"event", "driver"}, scored={"event": "C_event", "driver": "C_driver"},
                     consequences=set(), irrelevant={"decoy"}),
}


def build_vars(df, injected=False):
    h = df.index.hour.values; dw = df.index.dayofweek.values; dy = df.index.dayofyear.values
    V = pd.DataFrame(index=df.index)
    V["demand"] = df["demand"].values
    V["h_s1"] = np.sin(2 * np.pi * h / 24); V["h_c1"] = np.cos(2 * np.pi * h / 24)
    V["h_s2"] = np.sin(4 * np.pi * h / 24); V["h_c2"] = np.cos(4 * np.pi * h / 24)
    V["sat"] = (dw == 5).astype(float); V["sun"] = (dw == 6).astype(float); V["holiday"] = df["holiday"].values
    V["hdd"] = np.clip(18 - df["temp"].values, 0, None); V["cdd"] = np.clip(df["temp"].values - 18, 0, None)
    V["doy_s"] = np.sin(2 * np.pi * dy / 365.25); V["doy_c"] = np.cos(2 * np.pi * dy / 365.25)
    for c in ["humidity", "pressure", "wind_speed", "clouds"]:
        V[c] = df[c].values
    if injected:
        for c in INJ:
            V[c] = df[c].values
    for c in GEN:
        V[c] = df[c].values
    return V


def exog_of(names):
    return [n for n in names if n not in ("demand",) and n not in GEN]

# ----------------------------------------------------------------------------- GLS and the HAC screen (causal discovery of exogenous causes)
AR_LAGS = (1, 2, 24, 168)


def ar_filter(resid, lags=AR_LAGS):
    """AR coefficients of the regression errors (Cochrane-Orcutt step)."""
    L = max(lags); E = np.column_stack([resid[L - l:len(resid) - l] for l in lags])
    return np.linalg.lstsq(E, resid[L:], rcond=None)[0]


def quasi_diff(Z, rho, lags=AR_LAGS):
    """Z_t - sum_l rho_l Z_{t-l}: removes the slowly autocorrelated part of the errors."""
    Z = np.asarray(Z, float); L = max(lags); out = Z[L:].copy()
    for r, l in zip(rho, lags):
        out -= r * Z[L - l:len(Z) - l]
    return out


def gls_ar(Xm, y, n_iter=2):
    """Feasible GLS with AR(lags) errors. Returns beta, transformed design, transformed residuals."""
    beta = np.linalg.lstsq(Xm, y, rcond=None)[0]
    for _ in range(n_iter):
        rho = ar_filter(y - Xm @ beta)
        Xs, ys = quasi_diff(Xm, rho), quasi_diff(y, rho)
        beta = np.linalg.lstsq(Xs, ys, rcond=None)[0]
    return beta, Xs, ys - Xs @ beta


def hac_wald_screen(V, df, alpha, bandwidth=48):
    """Slow-driver screen: regress demand on flexible bases of ALL exogenous candidate groups (no demand history,
    which is a descendant of them), fit by GLS with AR-prewhitened errors, and test each group with a Newey-West
    (Bartlett, 48 h) HAC Wald test on the prewhitened residuals."""
    cand = sorted({GROUP[c] for c in exog_of(V.columns)})
    Bs = {g: basis(V, df, g) for g in cand}
    X = pd.concat(list(Bs.values()), axis=1); X = (X - X.mean()) / (X.std() + 1e-12)
    Xm = np.column_stack([np.ones(len(X)), X.values]); y = V["demand"].values
    beta, Xs, e = gls_ar(Xm, y)                      # prewhitened (GLS) fit; HAC on what is left
    Xe = Xs * e[:, None]; S = Xe.T @ Xe
    for l in range(1, bandwidth + 1):
        G = Xe[l:].T @ Xe[:-l]; S += (1 - l / (bandwidth + 1)) * (G + G.T)
    XtX_inv = np.linalg.pinv(Xs.T @ Xs); Vb = XtX_inv @ S @ XtX_inv
    out, col = {}, 1
    for g, B in Bs.items():
        k = B.shape[1]; ix = np.arange(col, col + k); col += k
        b = beta[ix]; W = float(b @ np.linalg.pinv(Vb[np.ix_(ix, ix)]) @ b)
        out[g] = stats.chi2.sf(W, k)
    return {g for g, pv in out.items() if pv < alpha}, out

# ----------------------------------------------------------------------------- causal graph + GLS estimator
TEMP_KNOTS = np.arange(4, 34, 3)


def basis(V, df, group):
    """Flexible basis for one driver group (what the estimator is allowed to learn)."""
    h = df.index.hour.values
    if group == "hour":                               # hour-of-day fixed effects: the injected event is a block of
        return pd.get_dummies(h, prefix="hr", drop_first=True).set_index(V.index).astype(float)   # hours, so hour must be fully controlled
    if group == "temp":
        T = df["temp"]; B = pd.DataFrame({f"tk{k}": np.clip(T - k, 0, None) for k in TEMP_KNOTS}, index=V.index); B["tlin"] = T
        return B
    if group in ("driver", "decoy"):
        x = V[group]; qs = np.quantile(x, [0.2, 0.4, 0.6, 0.8])
        B = pd.DataFrame({f"{group}_k{i}": np.clip(x - q, 0, None) for i, q in enumerate(qs)}, index=V.index); B[group] = x
        heat = np.clip((df["temp"] - 10) / 20, 0, 1)
        for c in list(B.columns): B[c + "_xheat"] = B[c] * heat     # allows a heat interaction, as any flexible model would
        return B
    if group in ("week", "holiday"):                 # level + change of the daily profile (real weekends and
        h = df.index.hour.values                       # holidays have a different shape, not only a lower level)
        H = pd.DataFrame({f"h{k}{f.__name__}": f(2 * np.pi * k * h / 24) for k in (1, 2, 3) for f in (np.sin, np.cos)}, index=V.index)
        cols = ["sat", "sun"] if group == "week" else ["holiday"]
        B = V[cols].copy()
        for c in cols:
            for hc in H.columns:
                B[f"{c}_x_{hc}"] = V[c] * H[hc]
        return B
    cols = [c for c in V.columns if GROUP.get(c) == group]
    return V[cols].copy()


def treatment_groups(parents):
    return sorted({GROUP[p] for p in parents if GROUP.get(p) not in ("history", "gas", "solar")})


def time_fe(index):
    """Month and year fixed effects: slow seasonal and trend levels that are not one of the drivers."""
    m = pd.get_dummies(index.month, prefix="m", drop_first=True).set_index(index).astype(float)
    y = pd.get_dummies(index.year, prefix="y", drop_first=True).set_index(index).astype(float)
    return pd.concat([m, y], axis=1)


def est_graph_ols(V, df, parents, add_history=False, method="gls"):
    """method='gls': AR-prewhitened GLS. method='ols_fe': OLS in levels with month and year fixed effects.
    The daily and weekly profile is fitted nonparametrically with hour-of-week fixed effects (167 dummies) and
    split afterwards into its weekday part (mean per day) and its daily-profile part (the rest)."""
    groups = treatment_groups(parents)
    how = ("hour" in groups) or ("week" in groups)
    Bs = {g: basis(V, df, g) for g in groups if g not in ("hour", "week")}
    if how:
        Bs["hour-of-week"] = pd.get_dummies(df.index.dayofweek * 24 + df.index.hour, prefix="how",
                                            drop_first=True).set_index(V.index).astype(float)
    X = pd.concat(list(Bs.values()), axis=1)
    if method == "ols_fe":
        X = pd.concat([X, time_fe(V.index)], axis=1)
    if add_history:
        X = pd.concat([X] + [V["demand"].shift(l).rename(f"y_l{l}") for l in (24, 48, 168)], axis=1)
    D = pd.concat([V["demand"].rename("y"), X], axis=1).dropna()
    if add_history or method == "ols_fe":             # OLS in levels (shortcut: plain OLS with demand lags)
        coef = dict(zip(D.columns[1:], LinearRegression().fit(D.drop(columns="y"), D["y"]).coef_))
    else:                                             # GLS with AR-prewhitened errors (exogenous regressors)
        Xm = np.column_stack([np.ones(len(D)), D.drop(columns="y").values])
        beta, _, _ = gls_ar(Xm, D["y"].values); coef = dict(zip(D.columns[1:], beta[1:]))
    out = {}
    for g, B in Bs.items():
        c = pd.Series(np.nan, index=V.index); v = B.loc[D.index].values @ np.array([coef[k] for k in B.columns])
        c.loc[D.index] = v - v.mean(); out[g] = c
    if how:
        c = out.pop("hour-of-week"); day = c.groupby(c.index.normalize()).transform("mean")
        out["week"] = day - day.mean(); out["hour"] = c - day
    return out

# ----------------------------------------------------------------------------- Shapley and scoring (same as the paper)
def shapley_values(vals, players, first=None):
    """Exact Shapley values from coalition values; `first` = players that must precede all others (asymmetric)."""
    n = len(players); idx = range(n); phi = np.zeros((len(vals[frozenset()]), n))
    def sym(sub_players, base):
        m = len(sub_players)
        for i in sub_players:
            others = [p for p in sub_players if p != i]
            for r in range(m):
                w = factorial(r) * factorial(m - r - 1) / factorial(m)
                for S in itertools.combinations(others, r):
                    S = frozenset(S) | base
                    phi[:, i] += w * (vals[S | {i}] - vals[S])
    if first is None:
        sym(list(idx), frozenset())
    else:
        A = [i for i in idx if players[i] in first]; Bp = [i for i in idx if players[i] not in first]
        sym(A, frozenset()); sym(Bp, frozenset(A))
    return phi


def score_contrib(contrib, df, dataset):
    T = TRUTH[dataset]; row = {}
    for g, comp in T["scored"].items():
        s = contrib.get(g)
        if s is None:
            row[f"r {g}"], row[g] = 0.0, 0.0; continue
        s = s.dropna(); tru = df.loc[s.index, comp]
        m = s.index >= pd.Timestamp("2018-01-01")            # score on the 2018 test year, like the forecasters
        if True:
            if m.sum() > 50: s, tru = s[m], tru[m]
        a, t = s.values - s.values.mean(), tru.values - tru.values.mean()
        ok = a.std() > 0 and t.std() > 0
        row[f"r {g}"] = float(np.corrcoef(a, t)[0, 1]) if ok else 0.0
        row[g] = float(np.polyfit(t, a, 1)[0]) if ok else 0.0
    bs = [row[g] for g in T["scored"]]
    row["magnitude correctness"] = 1 - float(np.mean(np.abs(1 - np.array(bs))))
    if dataset == "injected":
        s = contrib.get("driver")
        if s is not None:
            s = s.dropna(); s = s[s.index >= pd.Timestamp("2018-01-01")]; t = df.loc[s.index, "C_driver_only"]
            row["driver (driver-only part)"] = float(np.polyfit(t - t.mean(), s - s.mean(), 1)[0]) if s.std() > 0 else 0.0
        row["magnitude correctness (identifiable)"] = 1 - float(np.mean([abs(1 - row["event"]),
                                                                           abs(1 - row.get("driver (driver-only part)", 0.0))]))
        dec, drv = contrib.get("decoy"), contrib.get("driver")
        row["decoy credit"] = 0.0 if dec is None or drv is None else float(dec.abs().mean() / max(drv.abs().mean(), 1e-9))
    return row

# ----------------------------------------------------------------------------- TFT, explainers
# ---- the TFT (paper design: variable selection, LSTM encoder, gated residual networks, interpretable attention) ----
class GLU(nn.Module):
    def __init__(s, d_in, d_out):
        super().__init__(); s.fc = nn.Linear(d_in, 2 * d_out)
    def forward(s, x):
        a, b = s.fc(x).chunk(2, dim=-1); return a * torch.sigmoid(b)

class GRN(nn.Module):
    def __init__(s, d_in, d_hid, d_out, dropout):
        super().__init__()
        s.skip = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()
        s.fc1, s.fc2 = nn.Linear(d_in, d_hid), nn.Linear(d_hid, d_hid)
        s.drop, s.glu, s.norm = nn.Dropout(dropout), GLU(d_hid, d_out), nn.LayerNorm(d_out)
    def forward(s, x):
        return s.norm(s.skip(x) + s.glu(s.drop(s.fc2(torch.nn.functional.elu(s.fc1(x))))))

class VariableSelection(nn.Module):
    def __init__(s, n_var, d, dropout):
        super().__init__(); s.n = n_var
        s.emb = nn.ModuleList([nn.Linear(1, d) for _ in range(n_var)])
        s.var_grn = nn.ModuleList([GRN(d, d, d, dropout) for _ in range(n_var)])
        s.sel = GRN(n_var * d, d, n_var, dropout)
    def forward(s, x):                                                      # x: (B, T, V)
        e = torch.stack([s.emb[i](x[..., i:i + 1]) for i in range(s.n)], dim=-2)          # (B, T, V, d)
        w = torch.softmax(s.sel(e.flatten(-2)), dim=-1)                                  # (B, T, V)
        h = torch.stack([s.var_grn[i](e[..., i, :]) for i in range(s.n)], dim=-2)
        return (w.unsqueeze(-1) * h).sum(-2), w

class InterpretableMHA(nn.Module):
    def __init__(s, d, n_heads, dropout):
        super().__init__(); s.h, s.dk = n_heads, d // n_heads
        s.q = nn.ModuleList([nn.Linear(d, s.dk) for _ in range(n_heads)])
        s.k = nn.ModuleList([nn.Linear(d, s.dk) for _ in range(n_heads)])
        s.v, s.out, s.drop = nn.Linear(d, s.dk), nn.Linear(s.dk, d), nn.Dropout(dropout)
    def forward(s, q, k, v):
        V_ = s.v(v); heads = []
        for i in range(s.h):
            a = torch.softmax(s.q[i](q) @ s.k[i](k).transpose(1, 2) / np.sqrt(s.dk), dim=-1)
            heads.append(s.drop(a) @ V_)
        return s.out(torch.stack(heads).mean(0))

class TFT(nn.Module):
    def __init__(s, n_var, d, n_heads, dropout):
        super().__init__()
        s.vsn = VariableSelection(n_var, d, dropout)
        s.lstm = nn.LSTM(d, d, batch_first=True)
        s.gate_lstm, s.norm_lstm = GLU(d, d), nn.LayerNorm(d)
        s.enrich = GRN(d, d, d, dropout)
        s.attn = InterpretableMHA(d, n_heads, dropout)
        s.gate_attn, s.norm_attn = GLU(d, d), nn.LayerNorm(d)
        s.ff = GRN(d, d, d, dropout)
        s.gate_out, s.norm_out = GLU(d, d), nn.LayerNorm(d)
        s.head = nn.Linear(d, 1)
    def forward(s, x):                                                      # x: (B, T, V), scaled
        v, _ = s.vsn(x)
        h, _ = s.lstm(v)
        h = s.norm_lstm(v + s.gate_lstm(h))
        e = s.enrich(h)
        a = s.norm_attn(e[:, -1:] + s.gate_attn(s.attn(e[:, -1:], e, e)))  # the target hour attends over the window
        z = s.norm_out(h[:, -1:] + s.gate_out(s.ff(a)))
        return s.head(z)[:, 0, 0]


# ---- features and groups -------------------------------------------------------------------------------------
CAL = ["h_s1", "h_c1", "h_s2", "h_c2", "sat", "sun", "holiday"]
WX = ["hdd", "cdd", "humidity", "pressure", "wind_speed", "clouds", "doy_s", "doy_c"]
HIST = ["d_lag48", "d_lag168", "d_roll24_lag48"]
GENL = ["gen_gas_lag48", "gen_solar_lag48"]


def tft_frame(df, V, dataset):
    F = V[CAL + WX + (["event_flag", "driver", "decoy"] if dataset == "injected" else [])].copy()
    F["d_lag48"] = df["demand"].shift(48); F["d_lag168"] = df["demand"].shift(168)
    F["d_roll24_lag48"] = df["demand"].rolling(24).mean().shift(48)
    F["gen_gas_lag48"] = df["gen_gas"].shift(48); F["gen_solar_lag48"] = df["gen_solar"].shift(48)
    F["demand"] = df["demand"]
    return F.dropna()


def tft_groups(dataset, variant):
    if dataset == "synthetic":
        G = {"hour": CAL[:4], "week": ["sat", "sun"], "holiday": ["holiday"], "temp": ["hdd", "cdd"], "other": WX[2:]}
    else:   # 7 groups at most, so the grouped Shapley value stays exact
        G = {"calendar": CAL, "weather": WX, "event": ["event_flag"], "driver": ["driver"], "decoy": ["decoy"]}
    if variant == "with lags":
        G.update(history=HIST, generation=GENL)
    return G


def windows(F, cols, T):
    X = F[cols].values.astype(np.float32); y = F["demand"].values.astype(np.float32)
    W = np.lib.stride_tricks.sliding_window_view(X, (T, X.shape[1]))[:, 0]
    return np.ascontiguousarray(W), y[T - 1:], F.index[T - 1:]


class FittedTFT:
    def __init__(self, Wtr, ytr, Wva, yva, seed):
        torch.manual_seed(seed); np.random.seed(seed)
        flat = Wtr.reshape(-1, Wtr.shape[2]); self.mu, self.sd = flat.mean(0), flat.std(0) + 1e-6
        self.ym, self.ys = float(ytr.mean()), float(ytr.std())
        self.net = TFT(Wtr.shape[2], TWO["D_MODEL"], TWO["N_HEADS"], TWO["DROPOUT"]).to(DEVICE)
        opt = torch.optim.AdamW(self.net.parameters(), lr=TWO["LR"], weight_decay=1e-4); loss_fn = nn.HuberLoss()
        Xt = torch.tensor(self.scale(Wtr)); Yt = torch.tensor(((ytr - self.ym) / self.ys).astype(np.float32))
        Xv = torch.tensor(self.scale(Wva)); Yv = torch.tensor(((yva - self.ym) / self.ys).astype(np.float32))
        best, best_state, bad = np.inf, None, 0
        for ep in range(TWO["EPOCHS"]):
            self.net.train(); perm = torch.randperm(len(Xt))
            for b in range(0, len(Xt), TWO["BATCH"]):
                ib = perm[b:b + TWO["BATCH"]]
                opt.zero_grad(); loss = loss_fn(self.net(Xt[ib].to(DEVICE)), Yt[ib].to(DEVICE)); loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0); opt.step()
            self.net.eval()
            with torch.no_grad():
                vl = float(np.mean([loss_fn(self.net(Xv[b:b + 2048].to(DEVICE)), Yv[b:b + 2048].to(DEVICE)).item()
                                    for b in range(0, len(Xv), 2048)]))
            if vl < best - 1e-4:
                best, bad = vl, 0; best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
            else:
                bad += 1
                if bad >= TWO["PATIENCE"]: break
        self.net.load_state_dict(best_state); self.net.eval(); self.epochs = ep + 1

    def scale(self, W):
        return ((W - self.mu) / self.sd).astype(np.float32)

    def predict(self, W, batch=4096):
        out = []
        with torch.no_grad():
            for b in range(0, len(W), batch):
                out.append(self.net(torch.tensor(self.scale(W[b:b + batch])).to(DEVICE)).cpu().numpy())
        return np.concatenate(out) * self.ys + self.ym


def grouped_shapley(model, Wexp, Wbg, cols, G):
    """Exact grouped Shapley value: a group = its features over all 24 lags; absent groups come from background windows."""
    ci = {c: i for i, c in enumerate(cols)}; players = list(G); nE, nB, T, Fd = len(Wexp), len(Wbg), *Wexp.shape[1:]
    vals = {}
    for r in range(len(players) + 1):
        for S in itertools.combinations(range(len(players)), r):
            M = np.repeat(Wbg[None], nE, axis=0)
            for gi in S:
                idx = [ci[c] for c in G[players[gi]]]
                M[:, :, :, idx] = Wexp[:, None][:, :, :, idx]
            vals[frozenset(S)] = model.predict(M.reshape(-1, T, Fd)).reshape(nE, nB).mean(1)
    return shapley_values(vals, players), players


def eg_shap(model, Wexp, Wbg_eg, cols, G):
    """Expected gradients (shap.GradientExplainer) summed over lags and over the features of each group, in MW."""
    import shap
    class _Out(nn.Module):
        def __init__(s, net): super().__init__(); s.net = net
        def forward(s, x): return s.net(x).unsqueeze(-1)
    ex = shap.GradientExplainer(_Out(model.net), torch.tensor(model.scale(Wbg_eg)).to(DEVICE))
    sv = ex.shap_values(torch.tensor(model.scale(Wexp)).to(DEVICE), nsamples=TWO["EG_SAMPLES"])
    sv = np.asarray(sv[0] if isinstance(sv, list) else sv).reshape(len(Wexp), Wexp.shape[1], Wexp.shape[2]) * model.ys
    ci = {c: i for i, c in enumerate(cols)}
    return np.stack([sv[:, :, [ci[c] for c in G[g]]].sum((1, 2)) for g in G], axis=1), list(G)


def score_groups(phi, players, idx, df, dataset):
    contrib = {g: pd.Series(phi[:, k], index=idx) for k, g in enumerate(players)}
    row = score_contrib(contrib, df, dataset)
    tot = np.abs(phi).sum()
    row["history share"] = float(np.abs(phi[:, players.index("history")]).sum() / tot) if "history" in players else 0.0
    return row

# %%
# ----------------------------------------------------------------------------- one paired run
def run_pair(dataset, seed, real):
    path = os.path.join(CFG["CACHE_DIR"], f"{dataset}_seed{seed}_{CACHE_TAG}.pkl")
    if os.path.exists(path):
        return pickle.load(open(path, "rb"))
    t0 = time.time()
    df = generate_benchmark(seed, "default", 1.0) if dataset == "synthetic" else inject_effects(real, seed)
    V = build_vars(df, injected=(dataset == "injected")); F = tft_frame(df, V, dataset)
    rng = np.random.default_rng(seed)
    test_idx = F.index[F.index >= pd.Timestamp("2018-01-01")]
    exp_idx = pd.Index(rng.choice(test_idx, TWO["N_EXPLAIN"], replace=False))
    if dataset == "injected":
        ev = test_idx[df.loc[test_idx, "event_flag"] > 0]
        exp_idx = exp_idx.union(pd.Index(rng.choice(ev, min(TWO["N_EVENT_EXPLAIN"], len(ev)), replace=False)))
    exp_idx = exp_idx.sort_values()
    out = dict(dataset=dataset, seed=seed, scores={}, mae={}, epochs={})
    for variant in TWO["VARIANTS"]:
        G = tft_groups(dataset, variant); cols = [c for g in G.values() for c in g]
        W, y, widx = windows(F, cols, TWO["WINDOW"])
        tr = widx <= pd.Timestamp("2017-09-30 23:00")
        va = (widx > pd.Timestamp("2017-09-30 23:00")) & (widx < pd.Timestamp("2018-01-01"))
        te = widx >= pd.Timestamp("2018-01-01"); tr_i = np.where(tr)[0][::TWO["TRAIN_STRIDE"]]
        m = FittedTFT(W[tr_i], y[tr_i], W[va], y[va], seed)
        out["mae"][variant] = float(np.mean(np.abs(m.predict(W[te]) - y[te]))); out["epochs"][variant] = m.epochs
        pos = widx.get_indexer(exp_idx); Wexp = W[pos]
        Wbg = W[rng.choice(np.where(tr)[0], TWO["N_BG"], replace=False)]
        phi, players = grouped_shapley(m, Wexp, Wbg, cols, G)
        out["scores"][f"TFT {variant}, grouped Shapley"] = score_groups(phi, players, exp_idx, df, dataset)
        if TWO["USE_SHAP_EG"]:
            try:
                phi2, pl2 = eg_shap(m, Wexp, W[rng.choice(np.where(tr)[0], TWO["N_EG_BG"], replace=False)], cols, G)
                out["scores"][f"TFT {variant}, SHAP (expected gradients)"] = score_groups(phi2, pl2, exp_idx, df, dataset)
            except Exception as ex:
                print(f"  SHAP (expected gradients) skipped: {repr(ex)[:150]}")
    # causal graph + GLS: exogenous causes from the HAC screen (PCMCI+ only adds demand/generation dynamics,
    # which the estimator never uses), then GLS; scored on exactly the same hours
    screen, _ = hac_wald_screen(V, df, CFG["PC_ALPHA"])
    parents = {c for c in V.columns if GROUP.get(c) in screen}
    out["causes found"] = sorted(screen)
    c = est_graph_ols(V, df, parents)
    if dataset == "injected":
        c = {k: c[k] for k in ("event", "driver", "decoy") if k in c}
    out["scores"]["Causal graph + GLS"] = score_contrib({k: s.loc[exp_idx] for k, s in c.items()}, df, dataset)
    pickle.dump(out, open(path, "wb"))
    print(f"{dataset} seed {seed}: {time.time() - t0:.0f}s | MAE {({k: round(v) for k, v in out['mae'].items()})} "
          f"| epochs {out['epochs']} | causes found {out['causes found']}", flush=True)
    return out

# %% [markdown]
# ## Run all seeds

# %%
print("device:", DEVICE, "| seeds:", TWO["SEEDS"], "| datasets:", TWO["DATASETS"], "| smoke:", ARGS.smoke)
REAL = None
if "injected" in TWO["DATASETS"]:
    e, w = _find(ARGS.energy), _find(ARGS.weather)
    if not (e and w):
        raise FileNotFoundError("Kaggle files not found: put energy_dataset.csv and weather_features.csv in data/ "
                                "(see data/README.md), or set ARGS.synthetic_only = True.")
    REAL = load_kaggle(e, w)
RUNS = [run_pair(ds, s, REAL) for ds in TWO["DATASETS"] for s in TWO["SEEDS"]]

# %% [markdown]
# ## Results
# Mean ± sd over seeds. Real data uses the identifiable score (event + driver-only part). The **CHECK** line says whether the real-data TFTs trained well enough for the lag test to be reported.

# %%
rows = [dict(dataset=r["dataset"], seed=r["seed"], method=m, **sc) for r in RUNS for m, sc in r["scores"].items()]
D = pd.DataFrame(rows); D.to_csv(OUT_DIR + "/followup_results.csv", index=False)
lines = []
for ds in TWO["DATASETS"]:
    rr = [r for r in RUNS if r["dataset"] == ds]; Dd = D[D.dataset == ds]
    key = "magnitude correctness" if ds == "synthetic" else "magnitude correctness (identifiable)"
    cols = [c for c in list(TRUTH[ds]["scored"]) + ["driver (driver-only part)", "decoy credit", key,
                                                    "magnitude correctness", "history share"] if c in Dd]
    cols = list(dict.fromkeys(cols))
    mean, sd = Dd.groupby("method")[cols].mean(), Dd.groupby("method")[cols].std().fillna(0)
    tab = (mean.round(2).astype(str) + " ± " + sd.round(2).astype(str)).loc[mean.sort_values(key, ascending=False).index]
    lines += [f"\n=== {ds}, {len(rr)} seeds (score: {key}) ===", tab.to_string()]
    mae = pd.DataFrame([r["mae"] for r in rr])
    lines.append("TFT test MAE 2018 (MW), mean: " + ", ".join(f"{k} {v:.0f}" for k, v in mae.mean().items()))
    P = Dd.pivot(index="seed", columns="method", values=key)
    for a, b in [("TFT no lags, SHAP (expected gradients)", "TFT with lags, SHAP (expected gradients)"),
                 ("TFT no lags, grouped Shapley", "TFT with lags, grouped Shapley"),
                 ("Causal graph + GLS", "TFT no lags, grouped Shapley"),
                 ("Causal graph + GLS", "TFT with lags, grouped Shapley")]:
        if a in P and b in P:
            d = P[a] - P[b]; lines.append(f"  {a} vs {b}: {d.mean():+.2f}, higher in {(d > 0).sum()}/{len(d)} seeds")
    if ds == "injected":   # is the real-data TFT good enough to judge its explanation?
        ok_mae = mae["no lags"].mean() < CHECK["MAX_MAE_NO_LAGS"] and mae["with lags"].mean() < mae["no lags"].mean()
        ev = Dd[Dd.method.str.startswith("TFT")].groupby("method")["event"].mean()
        ok_ev = bool((ev >= CHECK["MIN_EVENT_SLOPE"]).all())
        lines.append(f"  CHECK real-data TFT: MAE ok = {ok_mae} (no-lag MAE < {CHECK['MAX_MAE_NO_LAGS']:.0f} and lags help), "
                     f"event learned = {ok_ev} (event slope >= {CHECK['MIN_EVENT_SLOPE']} for every TFT explainer)")
        lines.append("  -> " + ("results usable" if ok_mae and ok_ev else
                                "TFT still undertrained: do not report the real-data lag test yet"))
txt = "\n".join(lines); print(txt); open(OUT_DIR + "/followup_summary.txt", "w", encoding="utf-8").write(txt)

# %%
fig, axes = plt.subplots(1, len(TWO["DATASETS"]), figsize=(7 * len(TWO["DATASETS"]), 4.2), squeeze=False)
for ax, ds in zip(axes[0], TWO["DATASETS"]):
    key = "magnitude correctness" if ds == "synthetic" else "magnitude correctness (identifiable)"
    g_ = D[D.dataset == ds].groupby("method")[key]; mu, sd = g_.mean().sort_values(), g_.std().fillna(0)
    y = np.arange(len(mu))
    ax.errorbar(mu, y, xerr=sd[mu.index], fmt="o", ms=7, capsize=3, color="#2F6690")
    for yi, v in zip(y, mu): ax.annotate(f"{v:.2f}", (v, yi), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8)
    ax.set_yticks(y); ax.set_yticklabels(mu.index, fontsize=8); ax.set_xlim(0, 1.05); ax.set_xlabel(key)
    ax.set_title({"synthetic": "Synthetic benchmark", "injected": "Real demand, injected effects"}[ds])
fig.suptitle("Same seeds, same hours: TFT with / without lags vs causal graph + GLS", x=0.01, ha="left")
plt.tight_layout(); plt.savefig(OUT_DIR + "/followup_plot.png", dpi=150); plt.show()
print("saved followup_results.csv, followup_summary.txt, followup_plot.png in", OUT_DIR)
