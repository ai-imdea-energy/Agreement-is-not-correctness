# Agreement Is Not Correctness

Code for the paper **"Agreement Is Not Correctness: Validating Intrinsic and Post-hoc Explanations of Electricity Demand Forecasts Against Ground Truth"** (Ahmad J. A. Hijazi, IMDEA Energy).

The paper treats an explanation of a forecast as a claim that can be wrong. Hourly electricity demand is forecast one day ahead with XGBoost, LSTM, Transformer, Temporal Fusion Transformer (TFT) and RETAIN models. Their explanations (SHAP variants, Integrated Gradients, LIME, built-in importances) are scored against a synthetic copy of the system in which the true effect of every driver is recorded at every hour, against effects of known size injected into the real data, and against a causal estimate.

## Contents

```
agreement_is_not_correctness.py       every experiment of the paper, in one script for the terminal
Agreement_Is_Not_Correctness.ipynb    the same code as a notebook
src/                                  model, training and attribution modules used by Part 3
data/                                 the data used in the paper, Spain and Portugal (sources in data/README.md)
requirements.txt
```

## The notebook

| Part | Content | In the paper |
|---|---|---|
| 1 | Main pipeline: hyperparameter search, real data, injected effects, synthetic benchmark (seven versions), benchmark fitted to the real data, demand history as a shortcut, statistics, figures | Sections 6.1 to 6.6 and 6.10 |
| 2 | Real-data checks: size check, natural experiments, signatures, benchmark sensitivity | Section 6.5 |
| 3 | Intrinsic and post-hoc explanations of the same RETAIN model, one change at a time | Section 6.6 |
| 4 | Renewable sources under a hidden demand driver | Section 6.4 |
| 5 | Replication on Portugal | Section 6.7 |
| 6 | Combining SHAP over the window and over forecasts, LIME, quality indicators, correlated inputs | Sections 4.3, 4.4, 6.8 |
| 7 | Causal reference | Sections 4.5, 6.9 |
| 8 | Shortcut, paired with the grouped Shapley value and the causal estimate | Section 6.10 |

One settings cell at the top sets the plan for every part:

| Plan | Seeds | Neural training | Explained hours | Hyperparameter search |
|---|---|---|---|---|
| `smoke` | 1 (2 in Part 7) | 2 or 3 epochs | under 200 | 1 or 2 trials |
| `deep` | 10 | up to 150 epochs, patience 15 | 1,000 | 60 trials, up to 40 epochs each |

## Setup

Python 3.12.

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

The data is included in `data/`, so nothing else needs to be downloaded. Its sources are listed in [`data/README.md`](data/README.md).

## Running from a terminal (virtual machine, server)

```bash
python agreement_is_not_correctness.py --plan smoke        # every part with tiny settings: checks the whole script
nohup python agreement_is_not_correctness.py --plan deep > run.log 2>&1 &     # the runs of the paper, keeps going after logout
tail -f run.log
```

Useful options: `--parts 1,5` runs only some parts, `--stages bench,stats` and `--models TFT,XGBoost` restrict Part 1, and `--seeds`, `--max-epochs`, `--patience`, `--n-explain`, `--tune-trials`, `--tune-epochs` change the deep plan. `--help` lists everything. Each part runs in its own process and writes its own log to `results/logs/`. If one part fails, the others still run and the summary at the end names the failed ones. Running the same command again skips everything that is already finished.

## Running as a notebook

```bash
python -m jupyterlab
```

1. Set `PLAN = "smoke"` in the settings cell and run all cells. This checks every part with tiny settings.
2. Set `PLAN = "deep"` and run all cells.

The notebook holds the same code as the script, part by part.

The deep plan is a long run. It trains several hundred neural models, and Part 1 alone trains about 550. A GPU is used automatically when available and is strongly recommended. Every finished run is saved under `results/`, so after an interruption, run all again and the notebook continues where it stopped. The parts are independent of each other. Each part begins by clearing the names of the previous one and contains its own copy of the generator, models and scoring, so it can also be run on its own after the settings cell.

## Reproducibility

* All runs use fixed seeds. Results on other hardware or library versions can differ in the last digits, mostly for the neural models.
* Nothing selects seeds, runs or settings by test-set results. Hyperparameters are chosen on the validation period only.
* The real-data pipelines start with an automatic leakage audit: one demand value is raised by 100,000 MW, all features are rebuilt, and the run stops if any feature issued before that value changes.

## Citation

```
A. J. A. Hijazi, "Agreement Is Not Correctness: Validating Intrinsic and Post-hoc Explanations of
Electricity Demand Forecasts Against Ground Truth", 2026.
```
## License

The code is released under the MIT License (see `LICENSE`).
The data in `data/` belongs to its original sources and keeps their terms (see `data/README.md`).
Part 7 uses tigramite (GPL-3.0). If you use the causal part, please cite the tigramite papers for PCMCI+ and LPCMCI.
