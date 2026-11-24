"""
src/attribution_utils.py

Fixes the intrinsic-vs-post-hoc comparison.

Problem in 05: intrinsic_posthoc_agreement(retain_importance, shap_values_base)
compared RETAIN's explanation of the RETAIN model with SHAP's explanation of
the LSTM baseline. Two explainers AND two models differ, so the 0.12 mixes
"attention vs SHAP disagree" with "two networks use different features".

This module provides:
  retain_exact_contributions  exact, signed RETAIN attribution (Choi et al. 2016,
                              eq. for omega(y, x_{t,k})), sums to the prediction
  shap_gradient               GradientExplainer on ANY of your models, with a
                              TRAIN-set background (05 used test data as background)
  agreement_suite             Spearman + weighted Kendall + top-k Jaccard + sign
                              agreement, each with a random-ranking chance level
  decompose_disagreement      separates method-, proxy- and model-disagreement
  dinn_diagnostics            shows how much of the DINN score is just |r_true|

All attributions are computed on NORMALISED inputs, as in 05, so they are in
normalised-target units. Multiply by model._target_std for MW.
"""

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr, weightedtau


# ---------------------------------------------------------------------------
# 1. Exact RETAIN attribution
# ---------------------------------------------------------------------------

def retain_exact_contributions(model, X_norm, device="cpu", batch_size=512, check=True):
    """
    RETAINLSTM (lstm_models.py) computes, in eval mode:
        pred = b_out + sum_t sum_h w_out[h] * alpha_t * beta_t[h] * (W_emb[h,:] @ x_t + b_emb[h])
    With alpha, beta fixed by the forward pass this is linear in x, so the
    contribution of feature k at timestep t is exactly
        c[t, k] = x[t, k] * sum_h alpha_t * beta_t[h] * w_out[h] * W_emb[h, k]
    and   pred = c.sum() + bias_part.   (Choi et al., NeurIPS 2016, Sec. 3.3)

    v1's retain_feature_importance dropped w_out and x and took abs() before
    summing, so it measured attention *magnitude*, not contribution to output.

    Returns dict:
      contrib    (n, T, F) signed contributions
      bias_part  (n,)      the part not attributable to any input feature
      pred       (n,)      model output (normalised units)
      max_abs_completeness_error   should be ~1e-5; if not, the model class changed
    """
    model.eval()
    W = model.embed.weight.detach()        # (H, F)
    b_emb = model.embed.bias.detach()      # (H,)
    w_out = model.output.weight.detach()[0]  # (H,)
    b_out = model.output.bias.detach()[0]

    contribs, biases, preds = [], [], []
    with torch.no_grad():
        for s in range(0, len(X_norm), batch_size):
            x = torch.as_tensor(np.asarray(X_norm[s:s + batch_size], dtype=np.float32), device=device)
            pred, alpha, beta = model(x, return_attention=True)       # (B,), (B,T), (B,T,H)
            gate = alpha.unsqueeze(-1) * beta * w_out                  # (B,T,H)
            coef = gate @ W                                           # (B,T,F)
            contribs.append((coef * x).cpu().numpy())
            biases.append((b_out + (gate * b_emb).sum(dim=(1, 2))).cpu().numpy())
            preds.append(pred.cpu().numpy())

    out = {"contrib": np.concatenate(contribs), "bias_part": np.concatenate(biases),
           "pred": np.concatenate(preds)}
    recon = out["contrib"].sum(axis=(1, 2)) + out["bias_part"]
    out["max_abs_completeness_error"] = float(np.max(np.abs(recon - out["pred"])))
    if check and out["max_abs_completeness_error"] > 1e-3:
        raise RuntimeError(f"RETAIN decomposition does not sum to prediction "
                           f"(max err {out['max_abs_completeness_error']:.3g}). "
                           "Did RETAINLSTM's forward change?")
    return out


# ---------------------------------------------------------------------------
# 2. SHAP on any model, train background
# ---------------------------------------------------------------------------

class _AddOutputDim(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        out = out[0] if isinstance(out, tuple) else out
        return out.unsqueeze(-1)


def shap_gradient(model, X_train_norm, X_explain_norm, device="cpu",
                  n_background=300, nsamples=200, seed=0):
    """
    GradientExplainer (expected gradients). Background drawn from TRAIN data.
    Returns (n, T, F) signed SHAP values in normalised-target units.
    """
    import shap
    rng = np.random.default_rng(seed)
    bg_idx = rng.choice(len(X_train_norm), size=min(n_background, len(X_train_norm)), replace=False)
    bg = torch.as_tensor(np.asarray(X_train_norm[bg_idx], dtype=np.float32), device=device)
    xs = torch.as_tensor(np.asarray(X_explain_norm, dtype=np.float32), device=device)
    model.eval()
    expl = shap.GradientExplainer(_AddOutputDim(model).to(device), bg)
    raw = expl.shap_values(xs, nsamples=nsamples, rseed=seed)
    raw = raw[0] if isinstance(raw, list) else np.asarray(raw)[..., 0]
    return raw


def collapse_time(attr_seq, how="signed"):
    """(n,T,F) -> (n,F). 'signed' keeps opposing contributions cancelling,
    which is what a feature's net effect on the output actually is.
    'abs' is what 05 did; report both, lead with signed."""
    return attr_seq.sum(axis=1) if how == "signed" else np.abs(attr_seq).sum(axis=1)


# ---------------------------------------------------------------------------
# 3. Agreement suite with chance levels
# ---------------------------------------------------------------------------

def _topk(v, k):
    return set(np.argsort(-np.abs(v))[:k])


def agreement_suite(A, B, k=5, n_null=200, seed=0):
    """
    Per-instance agreement between two (n, F) signed attribution matrices.

    spearman  rank corr of |A| vs |B| over ALL features. With ~40 features,
              most near zero, this is dominated by how the noise tail is
              ordered, which drags it toward 0 even when the top agrees.
    wtau      weighted Kendall tau (hyperbolic weights): top ranks count most.
    jaccard@k overlap of the top-k features by |attribution|.
    sign@k    among B's top-k, fraction where A has the same sign.
    Each comes with a chance level from random rankings (same n features).
    """
    n, F = A.shape
    rows = []
    for i in range(n):
        a, b = A[i], B[i]
        rs = spearmanr(np.abs(a), np.abs(b))[0]
        wt = weightedtau(np.abs(a), np.abs(b))[0]
        ta, tb = _topk(a, k), _topk(b, k)
        jac = len(ta & tb) / len(ta | tb)
        tb_list = list(tb)
        sign = np.mean(np.sign(a[tb_list]) == np.sign(b[tb_list]))
        rows.append((rs, wt, jac, sign))
    per = pd.DataFrame(rows, columns=["spearman", "wtau", f"jaccard@{k}", f"sign@{k}"]).fillna(0.0)

    rng = np.random.default_rng(seed)
    null = []
    for _ in range(n_null):
        a, b = rng.random(F), rng.random(F)
        ta, tb = _topk(a, k), _topk(b, k)
        null.append((spearmanr(a, b)[0], weightedtau(a, b)[0], len(ta & tb) / len(ta | tb), 0.5))
    null = pd.DataFrame(null, columns=per.columns).mean()

    summary = pd.DataFrame({"mean": per.mean(), "median": per.median(),
                            "std": per.std(), "chance": null})
    summary["above_chance"] = summary["mean"] - summary["chance"]
    return per, summary


def decompose_disagreement(retain_exact, retain_proxy, shap_retain, shap_lstm, k=5):
    """
    Four comparisons, one table. Read it as:
      (1) original_05        proxy RETAIN vs SHAP on LSTM   -> should reproduce ~0.12
      (2) same_model_proxy   proxy RETAIN vs SHAP on RETAIN -> removes model confound
      (3) same_model_exact   exact RETAIN vs SHAP on RETAIN -> THE intrinsic-vs-post-hoc number
      (4) model_gap          SHAP on RETAIN vs SHAP on LSTM -> how differently the two
                                                                models actually reason
    (1)->(2) = model confound, (2)->(3) = proxy artefact, (3) = genuine method gap.
    All inputs (n, F); proxy is v1's abs-valued matrix, others signed.
    """
    pairs = {
        "1_original_05 (proxy RETAIN vs SHAP-LSTM)": (retain_proxy, shap_lstm),
        "2_same_model_proxy (proxy RETAIN vs SHAP-RETAIN)": (retain_proxy, shap_retain),
        "3_same_model_exact (exact RETAIN vs SHAP-RETAIN)": (retain_exact, shap_retain),
        "4_model_gap (SHAP-RETAIN vs SHAP-LSTM)": (shap_retain, shap_lstm),
    }
    rows = {}
    for name, (a, b) in pairs.items():
        _, s = agreement_suite(a, b, k=k)
        rows[name] = s["mean"]
    out = pd.DataFrame(rows).T
    out.loc["chance"] = s["chance"]   # chance level depends only on n_features and k
    return out


# ---------------------------------------------------------------------------
# 4. DINN score diagnostics
# ---------------------------------------------------------------------------

def dinn_diagnostics(importance, r_true, r_pred, divergence, feature_names, min_abs_r=0.05):
    """
    dinn_feature_importance returns |r_true| * 1/(1+|div|). r_true is a
    property of the DATA, identical for every model trained on it, so if
    corr(importance, |r_true|) ~ 1 the "intrinsic explanation" is mostly a
    correlation table, not something the model learned.

    Returns a table with both factors separated, plus:
      data_share   Spearman(importance, |r_true|)
      fidelity     1/(1+|div|) restricted to features with |r_true| >= min_abs_r
                   (removes the 0-minus-0 degenerate case without multiplying
                   the data back in). Report THIS as the model-dependent part.
    """
    fid = 1.0 / (1.0 + np.abs(divergence))
    fid_masked = np.where(np.abs(r_true) >= min_abs_r, fid, np.nan)
    table = pd.DataFrame({"importance": importance, "abs_r_true": np.abs(r_true),
                          "r_pred": r_pred, "fidelity": fid, "fidelity_masked": fid_masked},
                         index=feature_names).sort_values("importance", ascending=False)
    data_share = spearmanr(importance, np.abs(r_true))[0]
    return table, float(data_share)
