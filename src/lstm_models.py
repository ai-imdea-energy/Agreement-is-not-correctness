"""
src/lstm_models.py

Three LSTM variants, mapped directly to what's in the paper-summary table:

1. LSTMBaseline        - a plain sequence model, no explainability built in.
                          Post-hoc SHAP is applied to this one in
                          05_explainability_metrics.ipynb.

2. dinn_loss            - NOT a different architecture. DINN (Ashraf & Dua)
                          shows interpretability can come entirely from the
                          LOSS FUNCTION, same architecture as #1. Train
                          LSTMBaseline with this loss instead of plain MSE
                          and you get an intrinsically-explainable model
                          for free — no new class needed.

3. RETAINLSTM           - genuinely different architecture (Choi et al.,
                          cited in the Herrera XAI survey's RNN section).
                          Two-level attention (time + feature) gives you
                          built-in explanations as a direct model output,
                          not a separate step.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. Plain LSTM baseline
# ---------------------------------------------------------------------------

class LSTMBaseline(nn.Module):
    def __init__(self, n_features, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x):
        # x: (batch, seq_len, n_features)
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]          # (batch, hidden_size)
        return self.head(last_hidden).squeeze(-1)   # (batch,)


# ---------------------------------------------------------------------------
# 2. DINN-style correlation-augmented loss (Ashraf & Dua)
# ---------------------------------------------------------------------------

def batch_feature_correlations(X_last, y):
    """
    Vectorized Pearson correlation of every feature (last dim of X_last)
    against y, computed once for the whole batch.

    X_last : (batch, n_features) — DINN's original formulation is for
             tabular data; here we use the sequence's LAST timestep as
             the feature vector, i.e. the most recent observation, which
             is the closest sequence analogue to a tabular "feature at
             prediction time".
    y      : (batch,)
    Returns: (n_features,) correlation of each feature with y.
    """
    X_centered = X_last - X_last.mean(dim=0, keepdim=True)
    y_centered = y - y.mean()
    num = (X_centered * y_centered.unsqueeze(1)).sum(dim=0)
    # Keep the square-root derivative finite when a batch has near-zero variance.
    denom = torch.sqrt(
        (X_centered ** 2).sum(dim=0) * (y_centered ** 2).sum() + 1e-8
    )
    return num / denom


def dinn_loss(y_pred, y_true, X_last, lam=0.3):
    """
    L = [lam/(1+lam)] * MSE + [1/(1+lam)] * mean_j( (r_j|true - r_j|pred)^2 )

    Exactly the DINN loss formula, adapted to a sequence model by using
    the last timestep as the tabular feature vector. lam is the same
    hyperparameter DINN tunes (paper explored 0.1-0.4) — trades off
    prediction accuracy against how tightly the model's own predictions
    have to preserve the input-target correlation structure.

    NOTE: correlation over a mini-batch is noisy for small batches — use
    a batch size >= 128 (ideally 256+) for this term to be meaningful.
    This is a deliberate, documented adaptation for mini-batch SGD; the
    original DINN paper used full-batch gradient descent with momentum.
    """
    mse = F.mse_loss(y_pred, y_true)
    r_true = batch_feature_correlations(X_last, y_true)
    r_pred = batch_feature_correlations(X_last, y_pred)
    corr_term = ((r_true - r_pred) ** 2).mean()

    w_mse = lam / (1 + lam)
    w_corr = 1 / (1 + lam)
    return w_mse * mse + w_corr * corr_term


def dinn_feature_importance(model, X, y_true, device="cpu", return_diagnostics=False):
    """
    The intrinsic explanation for a DINN-trained model: since the loss
    itself is built from r_j|true vs r_j|pred, the natural "importance"
    of feature j is how much predicted correlation r_j|pred DIVERGES
    from the true correlation r_j|true it's being trained to match —
    small divergence = the model has faithfully captured that feature's
    real relationship to the target = high trust in that feature's role.

    IMPORTANT: a feature with near-zero variance (e.g. a generation
    source a country barely uses) has r_true ~= 0 and r_pred ~= 0 almost
    by construction -- zero minus zero is zero divergence, which a naive
    1/(1+divergence) score rewards with a perfect 1.0 despite there
    being no real relationship for the model to have faithfully
    captured. Multiplying by |r_true| fixes this: a feature the model
    couldn't possibly have a meaningful relationship with contributes
    ~0 to its own importance, regardless of how small the divergence is.

    Returns (n_features,) array: |r_true| * 1/(1 + |divergence|).
    High score now requires BOTH a real true relationship to the target
    AND the model having faithfully captured it -- not either alone.

    If return_diagnostics=True, also returns (r_true, r_pred, divergence)
    so you can inspect why a feature scored the way it did.
    """
    model.eval()
    with torch.no_grad():
        X_t = torch.from_numpy(X.astype("float32")).to(device)
        y_t = torch.from_numpy(y_true.astype("float32")).to(device)
        y_pred = model(X_t)
        X_last = X_t[:, -1, :]
        r_true = batch_feature_correlations(X_last, y_t)
        r_pred = batch_feature_correlations(X_last, y_pred)
        divergence = torch.abs(r_true - r_pred)
        importance = torch.abs(r_true) * (1.0 / (1.0 + divergence))

    if return_diagnostics:
        return importance.cpu().numpy(), r_true.cpu().numpy(), r_pred.cpu().numpy(), divergence.cpu().numpy()
    return importance.cpu().numpy()



# ---------------------------------------------------------------------------
# 3. RETAIN-style two-level attention LSTM (Choi et al., via Herrera survey §4.3.3)
# ---------------------------------------------------------------------------

class RETAINLSTM(nn.Module):
    """
    Simplified RETAIN: two GRUs process the REVERSED sequence (RETAIN =
    "REverse Time ATtention", the reversal is what lets the alpha
    attention be interpreted as 'how much does going further back in
    time matter'), producing:

      alpha (batch, seq_len) - scalar attention per timestep, softmax
                                over time, sums to 1 per instance
      beta  (batch, seq_len, hidden) - feature-level gating per timestep

    context = sum_t alpha_t * (beta_t (elementwise*) embed_t)
    prediction = linear(context)

    Both alpha and beta are genuine model outputs (return_attention=True),
    not a post-hoc approximation — this IS the model's own explanation of
    which timesteps and which embedded directions it relied on.
    """

    def __init__(self, n_features, hidden_size=64, dropout=0.2):
        super().__init__()
        self.embed = nn.Linear(n_features, hidden_size)
        self.gru_alpha = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.gru_beta = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.alpha_fc = nn.Linear(hidden_size, 1)
        self.beta_fc = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, x, return_attention=False):
        # x: (batch, seq_len, n_features)
        emb = self.embed(x)                       # (batch, seq_len, hidden)
        emb_rev = torch.flip(emb, dims=[1])

        g_alpha, _ = self.gru_alpha(emb_rev)       # (batch, seq_len, hidden)
        g_beta, _ = self.gru_beta(emb_rev)         # (batch, seq_len, hidden)

        alpha_scores = self.alpha_fc(g_alpha).squeeze(-1)   # (batch, seq_len)
        alpha = torch.softmax(alpha_scores, dim=1)          # sums to 1 over time
        beta = torch.tanh(self.beta_fc(g_beta))             # (batch, seq_len, hidden)

        # flip attention back to original chronological order for interpretability
        alpha = torch.flip(alpha, dims=[1])
        beta = torch.flip(beta, dims=[1])

        weighted = alpha.unsqueeze(-1) * beta * emb          # (batch, seq_len, hidden)
        context = weighted.sum(dim=1)                        # (batch, hidden)
        context = self.dropout(context)
        pred = self.output(context).squeeze(-1)               # (batch,)

        if return_attention:
            return pred, alpha, beta
        return pred


def retain_feature_importance(model, X, device="cpu", batch_size=512):
    """
    Converts RETAIN's (alpha, beta) into a single (n_samples, n_features)
    importance matrix directly comparable to SHAP output: for each
    instance, sum |beta_t * alpha_t| across time per hidden dimension,
    then project back to input-feature space via the embed layer's
    weight matrix (since beta operates in embedding space, not raw
    feature space).

    Returns (n_samples, n_features) — same shape SHAP would produce,
    which is what intrinsic_posthoc_agreement() needs to compare the two.
    """
    model.eval()
    all_importance = []
    embed_weight = model.embed.weight.detach()   # (hidden, n_features)

    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            X_batch = torch.from_numpy(X[start:start + batch_size].astype("float32")).to(device)
            _, alpha, beta = model(X_batch, return_attention=True)
            # alpha: (batch, seq_len), beta: (batch, seq_len, hidden)
            weighted_hidden = (alpha.unsqueeze(-1) * beta).sum(dim=1)   # (batch, hidden)
            # project hidden-space importance back to input-feature space
            feature_importance = weighted_hidden @ embed_weight        # (batch, n_features)
            all_importance.append(feature_importance.abs().cpu().numpy())

    return np.concatenate(all_importance, axis=0)