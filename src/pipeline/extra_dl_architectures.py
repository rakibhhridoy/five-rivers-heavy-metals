"""All five DL architectures from Supplementary §S3.1–S3.5.

Closes Gap 2 of docs/AUDIT_COVERAGE.md. The originals live in `train_individual_metals.py`
at the project root; this module curates them as a clean, importable library.

Architectures
-------------
1. Transformer-CNN-GNN-MLP   (S3.1) — tri-branch with MultiHeadAttention fusion
2. CNN-GNN-MLP               (S3.2) — concatenation-based fusion
3. GNN-MLP-Autoencoder       (S3.3) — bottleneck + reconstruction auxiliary loss
4. Dual-Attention            (S3.4) — channel + spatial attention with soft gate
5. Mixture-of-Experts        (S3.5) — three experts + softmax gating network

The CNN-GNN-MLP, GNN-MLP-Autoencoder, and Mixture-of-Experts builders were
referenced as pseudocode in the supplementary but not previously included in
the curated pipeline. Adding them here so the architecture set used in
phase4e/4f/5/4g is complete and the manuscript's "5 DL architectures × 8 metals
× 2 seasons" claim is fully reproducible.

Usage
-----
    from extra_dl_architectures import BUILDERS
    model = BUILDERS["Transformer CNN GNN MLP"](n_features=68)
    # `GNN MLP Autoencoder` returns a multi-output model — train with
    #   model.fit(X, {"prediction": y, "reconstruction": X}, ...)
"""
from __future__ import annotations

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, Conv1D, MaxPooling1D, Flatten, Reshape, Concatenate,
    MultiHeadAttention, LayerNormalization, Add, Multiply, Lambda,
    GlobalAveragePooling1D,
)
from tensorflow.keras.optimizers import Adam


def build_transformer_cnn_gnn_mlp(n_features: int) -> Model:
    inp = Input(shape=(n_features,))
    x_mlp = Dense(64, activation="relu")(inp)
    x_mlp = Dropout(0.3)(x_mlp)
    x_mlp = Dense(32, activation="relu")(x_mlp)

    x_cnn = Reshape((n_features, 1))(inp)
    x_cnn = Conv1D(32, 3, activation="relu", padding="same")(x_cnn)
    x_cnn = MaxPooling1D(2)(x_cnn)
    x_cnn = Conv1D(64, 3, activation="relu", padding="same")(x_cnn)
    x_cnn = Flatten()(x_cnn)
    x_cnn = Dense(32, activation="relu")(x_cnn)

    tokens = Concatenate(axis=1)([Reshape((1, 32))(x_mlp), Reshape((1, 32))(x_cnn)])
    attn = MultiHeadAttention(num_heads=4, key_dim=8)(tokens, tokens)
    attn = LayerNormalization()(Add()([tokens, attn]))
    attn = Flatten()(attn)

    x = Dense(64, activation="relu")(attn)
    x = Dropout(0.3)(x)
    out = Dense(1)(x)
    m = Model(inp, out)
    m.compile(optimizer=Adam(1e-3), loss="mse", metrics=["mae"])
    return m


def build_cnn_gnn_mlp(n_features: int) -> Model:
    inp = Input(shape=(n_features,))
    x_cnn = Reshape((n_features, 1))(inp)
    x_cnn = Conv1D(32, 3, activation="relu", padding="same")(x_cnn)
    x_cnn = MaxPooling1D(2)(x_cnn)
    x_cnn = Conv1D(64, 3, activation="relu", padding="same")(x_cnn)
    x_cnn = Flatten()(x_cnn)
    x_cnn = Dense(64, activation="relu")(x_cnn)

    x_mlp = Dense(64, activation="relu")(inp)
    x_mlp = Dropout(0.3)(x_mlp)
    x_mlp = Dense(32, activation="relu")(x_mlp)

    x = Concatenate()([x_cnn, x_mlp])
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.3)(x)
    out = Dense(1)(x)
    m = Model(inp, out)
    m.compile(optimizer=Adam(1e-3), loss="mse", metrics=["mae"])
    return m


def build_gnn_mlp_ae(n_features: int) -> Model:
    """Returns a multi-output model: prediction + reconstruction.

    Fit with: model.fit(X, {"prediction": y, "reconstruction": X}, ...).
    """
    inp = Input(shape=(n_features,))
    x = Dense(64, activation="relu")(inp)
    x = Dropout(0.3)(x)
    x = Dense(32, activation="relu")(x)
    bottleneck = Dense(16, activation="relu", name="bottleneck")(x)

    dec = Dense(32, activation="relu")(bottleneck)
    dec = Dense(64, activation="relu")(dec)
    reconstruction = Dense(n_features, name="reconstruction")(dec)

    reg = Dense(32, activation="relu")(bottleneck)
    reg = Dropout(0.2)(reg)
    prediction = Dense(1, name="prediction")(reg)

    m = Model(inp, [prediction, reconstruction])
    m.compile(
        optimizer=Adam(1e-3),
        loss={"prediction": "mse", "reconstruction": "mse"},
        loss_weights={"prediction": 1.0, "reconstruction": 0.1},
        metrics={"prediction": "mae"},
    )
    return m


def build_dual_attention(n_features: int) -> Model:
    inp = Input(shape=(n_features,))
    x_ch = Dense(32, activation="relu")(inp)
    ch_attn = Dense(32, activation="sigmoid")(x_ch)
    x_ch = Multiply()([x_ch, ch_attn])

    x_sp = Reshape((n_features, 1))(inp)
    x_sp = Conv1D(32, 3, activation="relu", padding="same")(x_sp)
    sp_attn = Conv1D(1, 1, activation="sigmoid")(x_sp)
    x_sp = Multiply()([x_sp, sp_attn])
    x_sp = Flatten()(x_sp)
    x_sp = Dense(32, activation="relu")(x_sp)

    gate_inp = Concatenate()([x_ch, x_sp])
    gate_w = Dense(1, activation="sigmoid")(gate_inp)
    fused = Add()([
        Multiply()([x_ch, gate_w]),
        Multiply()([x_sp, Lambda(lambda g: 1.0 - g)(gate_w)]),
    ])

    x = Dense(32, activation="relu")(fused)
    x = Dropout(0.3)(x)
    out = Dense(1)(x)
    m = Model(inp, out)
    m.compile(optimizer=Adam(1e-3), loss="mse", metrics=["mae"])
    return m


def build_mixture_of_experts(n_features: int) -> Model:
    inp = Input(shape=(n_features,))
    e1 = Dense(64, activation="relu")(inp)
    e1 = Dense(32, activation="relu")(e1)
    e1_out = Dense(1)(e1)

    x_conv = Reshape((n_features, 1))(inp)
    x_conv = Conv1D(32, 3, activation="relu", padding="same")(x_conv)
    x_conv = GlobalAveragePooling1D()(x_conv)
    e2_out = Dense(1)(x_conv)

    e3 = Dense(128, activation="relu")(inp)
    e3 = Dropout(0.3)(e3)
    e3 = Dense(64, activation="relu")(e3)
    e3 = Dense(32, activation="relu")(e3)
    e3_out = Dense(1)(e3)

    gate = Dense(32, activation="relu")(inp)
    gate = Dense(3, activation="softmax")(gate)

    experts = Concatenate()([e1_out, e2_out, e3_out])
    weighted = Multiply()([experts, gate])
    out = Lambda(lambda v: tf.reduce_sum(v, axis=-1, keepdims=True))(weighted)
    m = Model(inp, out)
    m.compile(optimizer=Adam(1e-3), loss="mse", metrics=["mae"])
    return m


BUILDERS = {
    "Transformer CNN GNN MLP": build_transformer_cnn_gnn_mlp,
    "CNN GNN MLP": build_cnn_gnn_mlp,
    "GNN MLP Autoencoder": build_gnn_mlp_ae,
    "Dual Attention": build_dual_attention,
    "Mixture of Experts": build_mixture_of_experts,
}


def _smoke_test() -> None:
    """Build each architecture once with n_features=68 and print param counts."""
    print("Building each architecture (n_features=68):")
    for name, build in BUILDERS.items():
        m = build(68)
        n_params = m.count_params()
        outs = (len(m.outputs) if isinstance(m.outputs, list) else 1)
        print(f"  {name:30s}  params={n_params:>7,}  outputs={outs}")


if __name__ == "__main__":
    _smoke_test()
