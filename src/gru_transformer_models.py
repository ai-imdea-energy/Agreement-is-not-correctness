"""
gru_transformer_models.py

Self-contained additions to the LSTM family in notebook 04: a GRU forecaster
and a Transformer forecaster, built to plug into the existing explainability
pipeline in notebook 05 with minimal changes.

Role each model plays in the explainability comparison (matching the
LSTM baseline / RETAIN split you already have):

- GRUForecaster:          post-hoc only. Explain it with SHAP exactly the way
                           you already explain model_base. No intrinsic
                           explanation of its own, it's a second "black box"
                           reference point, mainly useful for checking whether
                           GRU vs. LSTM changes accuracy at all before you
                           spend time explaining it.
- TransformerForecaster:  intrinsic (self-attention) AND post-hoc (SHAP), the
                           same dual role RETAIN plays. Its self-attention is
                           turned into a per-feature importance array in the
                           same (n_instances, n_features) shape as your
                           existing retain_importance, so it drops straight
                           into intrinsic_posthoc_agreement(), faithfulness_recall(),
                           deletion_insertion_auc(), etc. without changes to
                           those functions.

Conventions mirrored from your existing pipeline so this fits in with minimal
edits:
- Input is raw-scale (n, seq_len, n_features) numpy, same as X_train/X_test
  in notebook 04.
- Normalization stats are stored ON the model as buffers named
  _feature_mean / _feature_std / _target_mean / _target_std, same names you
  already read off model_base and model_lstm_day elsewhere.
- predict_with_model(model, X, device) takes raw-scale X, returns raw-scale
  predictions, same signature as the one already imported in notebook 05.
- get_transformer_attention_importance(...) returns the same shape and
  raw-value convention as retain_importance.

IMPORTANT — one thing this script cannot verify for you: the exact
column order of your feature_names / dayahead_features list has to match
whatever X_train/X_test you feed in, same requirement your existing LSTM
pipeline already has. Nothing here changes that.

No network access was available while writing this, so it has NOT been
run against a live PyTorch install. It is syntax-checked and the forward-pass
shapes are worked through by hand below, but run the smoke test at the
bottom (`python gru_transformer_models.py`, synthetic data, no real data
needed) before trusting it on your actual dataset.
"""

import math
import time

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# 1. Normalization helpers (mirrors the pattern already used for model_base /
#    model_retain / model_dinn: stats computed once on the training set,
#    stored on the model, applied inside forward() so predict_with_model can
#    always be called with raw-scale data).
# ---------------------------------------------------------------------------

def compute_norm_stats(X_train, y_train, eps=1e-8):
    """X_train: (n, seq_len, n_features) raw. y_train: (n,) raw."""
    feature_mean = X_train.reshape(-1, X_train.shape[-1]).mean(axis=0)
    feature_std = X_train.reshape(-1, X_train.shape[-1]).std(axis=0) + eps
    target_mean = y_train.mean()
    target_std = y_train.std() + eps
    return (
        torch.tensor(feature_mean, dtype=torch.float32),
        torch.tensor(feature_std, dtype=torch.float32),
        torch.tensor(float(target_mean), dtype=torch.float32),
        torch.tensor(float(target_std), dtype=torch.float32),
    )


# ---------------------------------------------------------------------------
# 2. GRU forecaster — post-hoc-only role, same shape contract as your
#    baseline LSTM: stacked recurrent layers, take the last timestep's
#    hidden state, linear head to a single value.
# ---------------------------------------------------------------------------

class GRUForecaster(nn.Module):
    def __init__(self, n_features, hidden_size=64, num_layers=2, dropout=0.2,
                 feature_mean=None, feature_std=None, target_mean=None, target_std=None):
        super().__init__()
        self.gru = nn.GRU(
            input_size=n_features, hidden_size=hidden_size, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )
        # normalization stats live on the model so predict_with_model never has to
        # know about them separately -- register as buffers so .to(device) moves them too
        self.register_buffer("_feature_mean", feature_mean if feature_mean is not None else torch.zeros(n_features))
        self.register_buffer("_feature_std", feature_std if feature_std is not None else torch.ones(n_features))
        self.register_buffer("_target_mean", target_mean if target_mean is not None else torch.tensor(0.0))
        self.register_buffer("_target_std", target_std if target_std is not None else torch.tensor(1.0))

    def forward(self, x_raw):
        """x_raw: (batch, seq_len, n_features) raw scale. Returns (batch,) raw-scale prediction."""
        x_norm = (x_raw - self._feature_mean) / self._feature_std
        out, _ = self.gru(x_norm)
        last = out[:, -1, :]                      # (batch, hidden_size)
        pred_norm = self.head(last).squeeze(-1)    # (batch,)
        return pred_norm * self._target_std + self._target_mean


# ---------------------------------------------------------------------------
# 3. Transformer forecaster — intrinsic + post-hoc role. A small, from-scratch
#    encoder (not nn.TransformerEncoderLayer) so attention weights are easy
#    to pull out cleanly via need_weights=True, rather than fighting
#    PyTorch's default encoder which doesn't expose them without hooks.
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    """Standard sinusoidal positions, added once to the projected input."""

    def __init__(self, d_model, max_len=200):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, : x.size(1), :]


class SelfAttentionBlock(nn.Module):
    """One encoder block: self-attention + residual/norm, then a small
    feedforward + residual/norm. Returns both the block's output and its
    raw attention weights (averaged over heads) so the caller can keep them."""

    def __init__(self, d_model, nhead, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_model * 2, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # need_weights=True + average_attn_weights=True -> attn_weights shape (batch, seq_len, seq_len)
        attn_out, attn_weights = self.attn(x, x, x, need_weights=True, average_attn_weights=True)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x, attn_weights


class TransformerForecaster(nn.Module):
    def __init__(self, n_features, d_model=64, nhead=4, num_layers=2, dropout=0.2,
                 feature_mean=None, feature_std=None, target_mean=None, target_std=None):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        self.blocks = nn.ModuleList([SelfAttentionBlock(d_model, nhead, dropout) for _ in range(num_layers)])
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_model // 2, 1),
        )
        self.register_buffer("_feature_mean", feature_mean if feature_mean is not None else torch.zeros(n_features))
        self.register_buffer("_feature_std", feature_std if feature_std is not None else torch.ones(n_features))
        self.register_buffer("_target_mean", target_mean if target_mean is not None else torch.tensor(0.0))
        self.register_buffer("_target_std", target_std if target_std is not None else torch.tensor(1.0))

    def forward(self, x_raw, return_attention=False):
        """x_raw: (batch, seq_len, n_features) raw scale.
        Returns (batch,) raw-scale prediction, or (prediction, attn_weights_last_layer)
        if return_attention=True, where attn_weights_last_layer is (batch, seq_len, seq_len)."""
        x_norm = (x_raw - self._feature_mean) / self._feature_std
        h = self.pos_encoding(self.input_proj(x_norm))
        attn_weights = None
        for block in self.blocks:
            h, attn_weights = block(h)   # keep only the LAST layer's attention for explanation
        last = h[:, -1, :]                       # representation at the final (most recent) timestep
        pred_norm = self.head(last).squeeze(-1)
        prediction = pred_norm * self._target_std + self._target_mean
        if return_attention:
            return prediction, attn_weights
        return prediction


# ---------------------------------------------------------------------------
# 4. Shared training loop (Adam + MSE + early stopping), same shape of
#    knobs as notebook 04's smoke-test-budget pattern.
# ---------------------------------------------------------------------------

def train_model(model, X_train, y_train, X_val, y_val, device,
                 epochs=50, lr=1e-3, patience=8, batch_size=64, verbose=True):
    """X_*, y_*: raw-scale numpy arrays. Returns the trained model (best-val-loss
    weights restored) and a small history dict."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32, device=device)

    n = len(X_train_t)
    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        epoch_loss = 0.0
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            xb = X_train_t[idx].to(device)
            yb = y_train_t[idx].to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(idx)
        epoch_loss /= n

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = loss_fn(val_pred, y_val_t).item()

        history["train_loss"].append(epoch_loss)
        history["val_loss"].append(val_loss)
        if verbose and (epoch % 5 == 0 or epoch == epochs - 1):
            print(f"  epoch {epoch:3d}  train_loss={epoch_loss:.2f}  val_loss={val_loss:.2f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                if verbose:
                    print(f"  early stopping at epoch {epoch}, best val_loss={best_val_loss:.2f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


# ---------------------------------------------------------------------------
# 4b. Random hyperparameter search, same spirit as train_utils.random_search
#     used for LSTM baseline / DINN / RETAIN in notebook 04, but written from
#     scratch here rather than calling into that function directly, GRU and
#     Transformer aren't LSTM subclasses so its internals may not generalize
#     cleanly, safer to have an explicit, readable version for these two.
# ---------------------------------------------------------------------------

def random_search(model_class, param_dist, X_train, y_train, X_val, y_val, device,
                   n_iter=6, epochs=12, patience=4, fixed_kwargs=None, seed=42, verbose=True):
    """param_dist: dict of {kwarg_name: [candidate values]}, same shape as
    notebook 04's baseline_param_dist. Samples n_iter random combinations,
    trains each at the given (small, search-budget) epochs/patience, keeps
    whichever combination gets the best validation MAE.

    Returns (best_params, best_val_mae, results_list) where results_list has
    one dict per trial, useful for sanity-checking the search itself.
    """
    fixed_kwargs = fixed_kwargs or {}
    rng = np.random.RandomState(seed)
    feature_mean, feature_std, target_mean, target_std = compute_norm_stats(X_train, y_train)

    param_names = list(param_dist.keys())
    best_params, best_val_mae, results = None, float("inf"), []

    for trial in range(n_iter):
        params = {name: param_dist[name][rng.randint(len(param_dist[name]))] for name in param_names}
        model_kwargs = {**fixed_kwargs, **params, "feature_mean": feature_mean, "feature_std": feature_std,
                         "target_mean": target_mean, "target_std": target_std}
        # "lr" and "batch_size" are train_model args, not model constructor args, split them out
        train_kwargs = {}
        for k in ("lr", "batch_size"):
            if k in model_kwargs:
                train_kwargs[k] = model_kwargs.pop(k)

        model = model_class(**model_kwargs)
        model, history = train_model(model, X_train, y_train, X_val, y_val, device,
                                      epochs=epochs, patience=patience, verbose=False, **train_kwargs)
        val_pred = predict_with_model(model, X_val, device)
        val_mae = float(np.mean(np.abs(val_pred - y_val)))
        results.append({"params": params, "val_mae": val_mae})

        if verbose:
            print(f"  trial {trial + 1}/{n_iter}  params={params}  val_mae={val_mae:.2f}")

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_params = params

    return best_params, best_val_mae, results


# ---------------------------------------------------------------------------
# 5. Inference helpers — same signature as the predict_with_model you
#    already import in notebook 05, plus the attention-importance extractor
#    for the Transformer.
# ---------------------------------------------------------------------------

def predict_with_model(model, X_raw, device, batch_size=512):
    """X_raw: (n, seq_len, n_features) raw numpy. Returns (n,) raw numpy predictions."""
    model = model.to(device).eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(X_raw), batch_size):
            xb = torch.tensor(X_raw[start : start + batch_size], dtype=torch.float32, device=device)
            preds.append(model(xb).cpu().numpy())
    return np.concatenate(preds)


def get_transformer_attention_importance(model, X_raw, device, batch_size=256):
    """Turns the Transformer's self-attention into a per-instance, per-feature
    importance array shaped (n, n_features) -- the same shape as retain_importance,
    so it plugs directly into intrinsic_posthoc_agreement(), faithfulness_recall(), etc.

    APPROXIMATION, stated plainly: raw self-attention is over TIMESTEPS, not
    features, there is no native per-feature attention the way RETAIN's
    variable-level beta gives you. This combines the two the same way RETAIN
    itself combines its visit-level alpha with a hidden-state contribution:
    importance[i, f] = sum over timesteps t of ( attention paid to t, from the
    model's final-timestep query ) * |raw feature value at (t, f)|.

    Treat this as a hypothesis to test, not a ground truth, the exact same
    way you tested RETAIN's attention against SHAP rather than assuming it
    was a faithful explanation. Run this through your existing
    intrinsic_posthoc_agreement() and faithfulness_recall() before trusting
    it in the report.
    """
    model = model.to(device).eval()
    n, seq_len, n_features = X_raw.shape
    importance = np.zeros((n, n_features), dtype=np.float32)

    with torch.no_grad():
        for start in range(0, n, batch_size):
            xb_raw = X_raw[start : start + batch_size]
            xb = torch.tensor(xb_raw, dtype=torch.float32, device=device)
            _, attn = model(xb, return_attention=True)          # attn: (batch, seq_len, seq_len)
            # attention FROM the last (most recent) timestep's query TO every key timestep
            last_query_attn = attn[:, -1, :].cpu().numpy()       # (batch, seq_len)
            # broadcast that per-timestep weight onto each timestep's raw |feature value|,
            # then sum over timesteps -> one importance number per feature
            batch_importance = np.einsum("bt,btf->bf", last_query_attn, np.abs(xb_raw))
            importance[start : start + batch_size] = batch_importance

    return importance


# ---------------------------------------------------------------------------
# 6. Comparison harness — trains both models, evaluates MAE, returns a table
#    in the same shape as the "Model / Nowcast MAE / Day-ahead MAE" table
#    from notebook 04, so you can just append rows to your existing one.
# ---------------------------------------------------------------------------

def run_comparison(X_train, y_train, X_val, y_val, X_test, y_test, device,
                    epochs=50, patience=8, verbose=True):
    """All X_*: (n, seq_len, n_features) raw numpy. All y_*: (n,) raw numpy.
    Returns (results_dict, models_dict). results_dict has MAE for each model,
    models_dict has the trained model objects for downstream SHAP/attention work."""
    n_features = X_train.shape[-1]
    feature_mean, feature_std, target_mean, target_std = compute_norm_stats(X_train, y_train)

    results = {}
    models = {}

    if verbose:
        print("Training GRU...")
    gru = GRUForecaster(n_features, feature_mean=feature_mean, feature_std=feature_std,
                         target_mean=target_mean, target_std=target_std)
    gru, _ = train_model(gru, X_train, y_train, X_val, y_val, device, epochs=epochs, patience=patience, verbose=verbose)
    gru_test_pred = predict_with_model(gru, X_test, device)
    results["GRU"] = float(np.mean(np.abs(gru_test_pred - y_test)))
    models["GRU"] = gru

    if verbose:
        print("Training Transformer...")
    transformer = TransformerForecaster(n_features, feature_mean=feature_mean, feature_std=feature_std,
                                         target_mean=target_mean, target_std=target_std)
    transformer, _ = train_model(transformer, X_train, y_train, X_val, y_val, device,
                                  epochs=epochs, patience=patience, verbose=verbose)
    transformer_test_pred = predict_with_model(transformer, X_test, device)
    results["Transformer"] = float(np.mean(np.abs(transformer_test_pred - y_test)))
    models["Transformer"] = transformer

    return results, models


# ---------------------------------------------------------------------------
# 7. SHAP compatibility note (not executable here — SHAP wasn't installed in
#    the environment this was written in, add this into notebook 05 directly
#    once GRU/Transformer are trained):
#
#     explainer_gru = shap.GradientExplainer(gru, background_sample)
#     shap_values_gru = explainer_gru.shap_values(X_test_sample)
#
#     explainer_transformer = shap.GradientExplainer(transformer, background_sample)
#     shap_values_transformer = explainer_transformer.shap_values(X_test_sample)
#
# Exactly the same call pattern already used for model_base in notebook 03,
# GradientExplainer works on any differentiable PyTorch model, nothing
# GRU/Transformer-specific needed.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Smoke test — synthetic data, no real dataset needed. Confirms the forward
# pass, training loop, and attention extraction all run end to end and
# produce the right shapes, before you point this at real price data.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Running smoke test on synthetic data (not your real dataset)...")
    rng = np.random.RandomState(0)
    n_train, n_val, n_test = 400, 100, 100
    seq_len, n_features = 24, 10

    def make_fake_data(n):
        X = rng.randn(n, seq_len, n_features).astype(np.float32) * 50 + 500
        # target is a noisy linear function of the last timestep, just so training loss can go down
        y = (X[:, -1, :3].sum(axis=1) + rng.randn(n) * 10).astype(np.float32)
        return X, y

    X_train, y_train = make_fake_data(n_train)
    X_val, y_val = make_fake_data(n_val)
    X_test, y_test = make_fake_data(n_test)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    t0 = time.time()
    results, models = run_comparison(X_train, y_train, X_val, y_val, X_test, y_test, device,
                                      epochs=15, patience=5, verbose=True)
    print(f"\ndone in {time.time() - t0:.1f}s")
    print("Test MAE:", results)

    importance = get_transformer_attention_importance(models["Transformer"], X_test, device)
    print("Transformer attention-importance shape:", importance.shape, "expected:", (n_test, n_features))
    assert importance.shape == (n_test, n_features), "shape mismatch — see get_transformer_attention_importance"

    print("\nSmoke test passed: forward pass, training loop, and attention extraction all ran without error.")
