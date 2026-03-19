from __future__ import annotations

import tensorflow as tf
from keras import layers, Model

from .config import ModelConfig


class TemporalAttention(layers.Layer):
    def __init__(self, attention_units: int = 64, **kwargs):
        super().__init__(**kwargs)
        self.score_dense = layers.Dense(attention_units, activation="tanh")
        self.weight_dense = layers.Dense(1)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        # inputs shape: (batch, time_steps, features)
        score = self.score_dense(inputs)
        weight_logits = self.weight_dense(score)
        weights = tf.nn.softmax(weight_logits, axis=1)
        context = tf.reduce_sum(weights * inputs, axis=1)
        return context


def _lstm_attention_block(x: tf.Tensor, cfg: ModelConfig, block_idx: int) -> tf.Tensor:
    out = x
    for i, units in enumerate(cfg.lstm_units_per_block, start=1):
        out = layers.LSTM(units, return_sequences=True, name=f"block{block_idx}_lstm{i}")(out)
    out = TemporalAttention(cfg.attention_units, name=f"block{block_idx}_attention")(out)
    return out


def build_icnn_lstm_plus(cfg: ModelConfig) -> Model:
    inputs = layers.Input(shape=(cfg.input_features, 1), name="event_input")

    x = inputs
    for i, filters in enumerate(cfg.cnn_filters, start=1):
        x = layers.Conv1D(
            filters=filters,
            kernel_size=cfg.cnn_kernel_size,
            padding="same",
            activation=cfg.cnn_activation,
            name=f"cnn_{i}",
        )(x)

    x = layers.MaxPooling1D(pool_size=cfg.pool_size, name="max_pool")(x)

    branches = []
    for b in range(1, cfg.num_parallel_lstm_blocks + 1):
        branches.append(_lstm_attention_block(x, cfg, block_idx=b))

    x = layers.Concatenate(name="concat_branches")(branches)
    x = layers.Flatten(name="flatten_concat")(x)
    x = layers.Dense(cfg.dense_units, activation=cfg.hidden_activation, name="dense_1")(x)
    x = layers.Dropout(cfg.dropout_rate, name="dropout_1")(x)
    outputs = layers.Dense(cfg.second_dense_units, activation=cfg.output_activation, name="class_output")(x)

    model = Model(inputs=inputs, outputs=outputs, name="iCNN_LSTM_plus")

    optimizer = tf.keras.optimizers.Adam(learning_rate=cfg.learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
        ],
    )
    return model


def freeze_for_incremental_update(model: Model) -> None:
    # Freeze all feature extractor layers and keep outer classifier layers trainable.
    for layer in model.layers:
        layer.trainable = False

    for layer_name in ("dense_1", "dropout_1", "class_output"):
        layer = model.get_layer(layer_name)
        layer.trainable = True

    model.compile(
        optimizer=model.optimizer,
        loss=model.loss,
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
        ],
    )
