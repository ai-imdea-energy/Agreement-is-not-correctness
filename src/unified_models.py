"""
src/unified_models.py

GRU / Transformer variants that train through train_utils.train_improved_lstm,
so all neural models share one recipe (AdamW, SmoothL1, plateau LR, clipping).

Why needed: gru_transformer_models.py registers _feature_mean/_target_mean/...
as tensor BUFFERS and normalises inside forward(). train_improved_lstm
normalises in SequenceDataset and then assigns plain floats/arrays to those same
attribute names, which (a) raises TypeError on a registered buffer and (b) would
normalise twice. These subclasses drop the buffers and the internal scaling.
Architectures, and therefore any tuned hyperparameters, are unchanged.
"""

from gru_transformer_models import GRUForecaster, TransformerForecaster

_BUFFERS = ("_feature_mean", "_feature_std", "_target_mean", "_target_std")


def _drop_buffers(module):
    for name in _BUFFERS:
        module._buffers.pop(name, None)


class GRUPlain(GRUForecaster):
    def __init__(self, n_features, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__(n_features, hidden_size, num_layers, dropout)
        _drop_buffers(self)

    def forward(self, x):                      # x already normalised
        out, _ = self.gru(x)
        return self.head(out[:, -1, :]).squeeze(-1)


class TransformerPlain(TransformerForecaster):
    def __init__(self, n_features, d_model=64, nhead=4, num_layers=2, dropout=0.2):
        super().__init__(n_features, d_model, nhead, num_layers, dropout)
        _drop_buffers(self)

    def forward(self, x, return_attention=False):
        h = self.pos_encoding(self.input_proj(x))
        attn = None
        for block in self.blocks:
            h, attn = block(h)
        pred = self.head(h[:, -1, :]).squeeze(-1)
        return (pred, attn) if return_attention else pred
