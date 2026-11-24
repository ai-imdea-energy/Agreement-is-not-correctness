"""
src/train_utils.py

Training utilities for all three LSTM models (baseline, DINN-style
intrinsic, RETAIN-attention). Two phases, two functions:

1. random_search()        - hyperparameter tuning (unchanged from earlier,
                             this part was never broken)
2. train_improved_lstm()  - final training with the full recipe: AdamW,
                             SmoothL1Loss by default (or any custom loss,
                             e.g. DINN's), ReduceLROnPlateau, gradient
                             clipping. Works for all three models via
                             model_class/model_kwargs/loss_fn, not just
                             the baseline.

Both route data through SequenceDataset for normalization. That
normalization is what fixes the collapse-to-mean bug from earlier: a
custom version of this loop built directly on raw TensorDataset (no
normalization) meant every residual sat in SmoothL1Loss's purely-linear
regime (direction only, no magnitude), which combined with tight
gradient clipping left the model with no real path past "predict near
the mean".
"""

import copy
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import ParameterSampler
from sklearn.metrics import mean_absolute_error


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Phase 1: hyperparameter tuning (unchanged — this part was working)
# ---------------------------------------------------------------------------

def train_one_model(model, train_loader, val_loader, loss_fn, epochs=30,
                     lr=1e-3, weight_decay=1e-5, patience=5, device="cpu",
                     verbose=False):
    """
    Used only during random_search trials (cheap, short runs to compare
    hyperparameter candidates) — the final, full-quality fit uses
    train_improved_lstm below instead.

    loss_fn(y_pred, y_true, X_batch) -> scalar tensor.
    Returns (best_model_state_dict, best_val_mae [normalized scale], history dict).
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_mae = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0
    history = {"train_loss": [], "val_mae": []}

    for epoch in range(epochs):
        model.train()
        epoch_loss, n_batches = 0.0, 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch, X_batch)
            if not torch.isfinite(loss):
                print(f"NaN/inf loss — epoch {epoch}, batch {n_batches}")
                print("y_pred finite:", torch.isfinite(y_pred).all().item())
                break
            loss.backward()
            if not all(
                parameter.grad is None or torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
            ):
                print(f"NaN/inf gradient — epoch {epoch}, batch {n_batches}")
                break
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        model.eval()
        val_preds, val_true = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_pred = model(X_batch)
                val_preds.append(y_pred.cpu().numpy())
                val_true.append(y_batch.numpy())
        val_preds = np.concatenate(val_preds)
        val_true = np.concatenate(val_true)
        if not np.isfinite(val_preds).all() or not np.isfinite(val_true).all():
            print(f"NaN/inf validation output — epoch {epoch}")
            return best_state, float("inf"), history
        val_mae = mean_absolute_error(val_true, val_preds)

        history["train_loss"].append(epoch_loss / max(n_batches, 1))
        history["val_mae"].append(val_mae)

        if verbose:
            print(f"  epoch {epoch+1:3d}/{epochs} | train_loss={history['train_loss'][-1]:.4f} | val_mae={val_mae:.4f}")

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                if verbose:
                    print(f"  early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
                break

    return best_state, best_val_mae, history


def random_search(model_class, model_arg_names, param_distributions,
                   loss_fn_factory, X_train_seq, y_train_seq,
                   n_iter=12, val_frac=0.15, batch_size=256,
                   epochs=25, patience=5, device="cpu", seed=42, verbose=True,
                   fixed_model_kwargs=None):
    """
    Random search over model + training hyperparameters, using the LAST
    val_frac of the training portion as a chronological validation set.

    model_class         : e.g. LSTMBaseline or RETAINLSTM
    model_arg_names      : which keys in param_distributions are __init__
                           kwargs for model_class
    param_distributions  : dict, sklearn ParameterSampler-compatible
    loss_fn_factory      : callable(sampled_params_dict) -> loss_fn(yp, yt, xb)
    fixed_model_kwargs    : dict of __init__ kwargs that are fixed, not
                           searched — this is where n_features goes,
                           since it's determined by your data, not a
                           hyperparameter choice.

    Returns (best_params, best_val_mae, best_state_dict, all_trial_results)
    """
    from sequence_utils import SequenceDataset

    fixed_model_kwargs = fixed_model_kwargs or {}
    rng = np.random.RandomState(seed)
    sampler = list(ParameterSampler(param_distributions, n_iter=n_iter, random_state=rng))

    split_idx = int(len(X_train_seq) * (1 - val_frac))
    X_tr, X_val = X_train_seq[:split_idx], X_train_seq[split_idx:]
    y_tr, y_val = y_train_seq[:split_idx], y_train_seq[split_idx:]

    ds_tr = SequenceDataset(X_tr, y_tr)
    ds_val = SequenceDataset(X_val, y_val, ds_tr.feature_mean, ds_tr.feature_std,
                              ds_tr.target_mean, ds_tr.target_std)

    best_val_mae = float("inf")
    best_params = None
    best_state = None
    all_results = []

    for trial_i, params in enumerate(sampler):
        set_seed(seed + trial_i)

        model_kwargs = {k: v for k, v in params.items() if k in model_arg_names}
        model_kwargs.update(fixed_model_kwargs)
        model = model_class(**model_kwargs)

        loss_fn = loss_fn_factory(params)
        trial_batch_size = params.get("batch_size", batch_size)
        trial_lr = params.get("lr", 1e-3)
        trial_wd = params.get("weight_decay", 1e-5)

        train_loader = DataLoader(ds_tr, batch_size=trial_batch_size, shuffle=True)
        val_loader = DataLoader(ds_val, batch_size=trial_batch_size, shuffle=False)

        state, val_mae_norm, _ = train_one_model(
            model, train_loader, val_loader, loss_fn,
            epochs=epochs, lr=trial_lr, weight_decay=trial_wd,
            patience=patience, device=device, verbose=False,
        )
        val_mae_denorm = val_mae_norm * ds_tr.target_std

        all_results.append({**params, "val_mae": val_mae_denorm})
        if verbose:
            print(f"trial {trial_i+1}/{n_iter}: val_mae={val_mae_denorm:.2f}  params={params}")

        if val_mae_denorm < best_val_mae:
            best_val_mae = val_mae_denorm
            best_params = params
            best_state = state

    return best_params, best_val_mae, best_state, all_results


# ---------------------------------------------------------------------------
# Phase 2: final training — the fixed, generalized "improved" loop
# ---------------------------------------------------------------------------

def smooth_l1_loss_fn(y_pred, y_true, X_batch, beta=1.0):
    """Default final-fit loss. Only sensible on NORMALIZED targets (see
    module docstring) — beta=1.0 assumes residuals are roughly unit-scale,
    which is true post-normalization and false on raw MW values."""
    return F.smooth_l1_loss(y_pred, y_true, beta=beta)


def train_improved_lstm(model_class, model_kwargs, loss_fn,
                         X_train, y_train, X_val, y_val,
                         params, device,
                         epochs=60, patience=12, lr_patience=5,
                         grad_clip=5.0, verbose_every=5):
    """
    Final-fit training loop: AdamW + ReduceLROnPlateau + gradient
    clipping, on top of proper target/feature normalization via
    SequenceDataset. Works for any of the three models:

        train_improved_lstm(LSTMBaseline, {}, smooth_l1_loss_fn, ...)
        train_improved_lstm(LSTMBaseline, {}, dinn_loss_wrapper, ...)   # lam baked into the wrapper
        train_improved_lstm(RETAINLSTM, {}, smooth_l1_loss_fn, ...)

    model_kwargs : __init__ kwargs EXCLUDING n_features (added
                   automatically from X_train.shape[-1], so the same
                   params/model_kwargs can be reused for nowcast and
                   day-ahead even though they have different feature
                   counts).
    loss_fn       : loss_fn(y_pred, y_true, X_batch) -> scalar tensor,
                   same signature as train_one_model's, operating on
                   NORMALIZED y_pred/y_true.
    params        : dict with "lr", "weight_decay", "batch_size" (the
                   training-side hyperparameters; typically your
                   best_params from random_search).

    Returns (model, history [list of per-epoch dicts], best_epoch, best_val_loss).
    History's val_mae is already denormalized (real target units);
    val_loss is in whatever scale loss_fn operates in (normalized).
    """
    from sequence_utils import SequenceDataset

    n_features = X_train.shape[-1]

    ds_train = SequenceDataset(X_train, y_train)
    ds_val = SequenceDataset(X_val, y_val, ds_train.feature_mean, ds_train.feature_std,
                              ds_train.target_mean, ds_train.target_std)

    batch_size = params.get("batch_size", 256)
    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True,
                               drop_last=False, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False,
                             drop_last=False, pin_memory=(device.type == "cuda"))

    full_model_kwargs = {**model_kwargs, "n_features": n_features}
    model = model_class(**full_model_kwargs).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=params.get("lr", 1e-3),
        weight_decay=params.get("weight_decay", 1e-5),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=lr_patience, min_lr=1e-6,
    )

    target_mean = float(ds_train.target_mean)
    target_std = float(ds_train.target_std)

    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb, xb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        val_preds_norm, val_targets_norm = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                pred = model(xb)
                val_losses.append(loss_fn(pred, yb, xb).item())
                val_preds_norm.append(pred.cpu())
                val_targets_norm.append(yb.cpu())

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))

        val_preds_norm = torch.cat(val_preds_norm).numpy()
        val_targets_norm = torch.cat(val_targets_norm).numpy()
        val_preds = val_preds_norm * target_std + target_mean
        val_targets = val_targets_norm * target_std + target_mean
        val_mae = mean_absolute_error(val_targets, val_preds)

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)

        history.append({
            "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
            "val_mae": val_mae, "lr": current_lr,
        })

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            epochs_without_improvement += 1

        if epoch == 1 or epoch % verbose_every == 0:
            print(f"epoch {epoch:03d} | train_loss={train_loss:.5f} | val_loss={val_loss:.5f} | "
                  f"val_mae={val_mae:.2f} | lr={current_lr:.2e}")

        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch} (best was epoch {best_epoch}).")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    # stash normalization stats on the model so predict_with_model can
    # denormalize later without ds_train needing to be kept around
    model._target_mean = target_mean
    model._target_std = target_std
    model._feature_mean = ds_train.feature_mean
    model._feature_std = ds_train.feature_std

    return model, history, best_epoch, best_val_loss


def predict_with_model(model, X, device, batch_size=512):
    """
    Predicts in real target units (MW, or EUR/MWh for price), using the
    normalization stats train_improved_lstm attached to the model. Use
    this instead of calling model(x) directly whenever you want
    human-readable output.
    """
    from torch.utils.data import TensorDataset

    feature_mean = getattr(model, "_feature_mean", None)
    feature_std = getattr(model, "_feature_std", None)
    target_mean = getattr(model, "_target_mean", 0.0)
    target_std = getattr(model, "_target_std", 1.0)

    X_arr = np.asarray(X, dtype=np.float32)
    if feature_mean is not None:
        X_arr = (X_arr - feature_mean) / feature_std

    loader = DataLoader(TensorDataset(torch.from_numpy(X_arr)), batch_size=batch_size, shuffle=False)

    predictions = []
    model.eval()
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device)
            pred = model(xb)
            predictions.append(pred.cpu())

    preds_norm = torch.cat(predictions).numpy().reshape(-1)
    return preds_norm * target_std + target_mean