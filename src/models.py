"""The two models under comparison: a classical baseline and a multi-task network."""
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from .data import ROUTES, SENTIMENTS

SEED = 7


def baseline() -> dict:
    """
    TF-IDF over words and character n-grams into logistic regression, one per task.
    Character n-grams catch misspellings and hashtags ("#fail", "delayyyed").
    It's the bar the neural network has to clear to justify itself.
    """
    def make():
        feats = FeatureUnion([
            ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True)),
        ])
        return Pipeline([("features", feats),
                         ("model", LogisticRegression(C=4.0, max_iter=3000, class_weight="balanced"))])
    return {"sentiment": make(), "route": make()}


def multitask_network(train_texts, vocab_size: int = 20000, seq_len: int = 48, embed_dim: int = 96,
                      units: int = 48, dropout: float = 0.35):
    """
    One encoder, two heads. Sentiment and route are closely related (anything
    routed to a team is negative), so learning them together lets each task
    help the other, and serving needs a single forward pass.
    """
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    keras.utils.set_random_seed(SEED)
    vectorize = layers.TextVectorization(max_tokens=vocab_size, output_sequence_length=seq_len,
                                         standardize="lower_and_strip_punctuation", name="vectorize")
    vectorize.adapt(tf.constant(train_texts))

    text_in = keras.Input(shape=(1,), dtype="string", name="text")
    x = vectorize(text_in)
    x = layers.Embedding(vocab_size, embed_dim, mask_zero=True, name="embed")(x)
    x = layers.SpatialDropout1D(0.2)(x)
    x = layers.Bidirectional(layers.GRU(units, return_sequences=True), name="bigru")(x)
    pooled = layers.Concatenate()([layers.GlobalMaxPooling1D()(x), layers.GlobalAveragePooling1D()(x)])
    shared = layers.Dropout(dropout)(layers.Dense(64, activation="relu")(pooled))
    sentiment = layers.Dense(len(SENTIMENTS), activation="softmax", name="sentiment")(shared)
    route = layers.Dense(len(ROUTES), activation="softmax", name="route")(shared)

    model = keras.Model(text_in, {"sentiment": sentiment, "route": route}, name="jetway_multitask")
    model.compile(optimizer=keras.optimizers.Adam(2e-3),
                  loss={"sentiment": "sparse_categorical_crossentropy", "route": "sparse_categorical_crossentropy"},
                  loss_weights={"sentiment": 0.5, "route": 1.0},
                  metrics={"sentiment": ["accuracy"], "route": ["accuracy"]})
    return model


def class_weights(y: np.ndarray, n: int) -> np.ndarray:
    """Per-sample weights so rare routes (booking, baggage) aren't drowned out."""
    counts = np.bincount(y, minlength=n).astype(float)
    w = len(y) / (n * np.maximum(counts, 1))
    return np.sqrt(w)[y]  # square root: helps rare classes without wrecking calibration
