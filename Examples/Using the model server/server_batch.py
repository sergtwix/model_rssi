# server_pairs.py
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from fastapi import FastAPI, Body, HTTPException
from typing import List, Tuple
import uvicorn


# =========================================================
# 1. EXACT COPIES OF FUNCTIONS FROM YOUR TRAINING SCRIPT
# =========================================================

def features_3ch_filtered(x):
    """
    Feature generation for filtered signal (3 channels).
    """
    x_prev = tf.roll(x, shift=1, axis=1)
    delta = x - x_prev
    mask = tf.concat([tf.zeros_like(x[:, :1, :]), tf.ones_like(x[:, 1:, :])], axis=1)
    delta = delta * mask
    local_std = tf.math.reduce_std(x, axis=1, keepdims=True)
    local_std_seq = tf.tile(local_std, [1, tf.shape(x)[1], 1])
    return tf.concat([x, delta, local_std_seq], axis=-1)


def features_4ch_raw(x):
    """
    Feature generation for raw signal (4 channels: Raw, Delta, Std, Z-score).
    """
    eps = 1e-6
    x_prev = tf.roll(x, shift=1, axis=1)
    delta = x - x_prev
    mask = tf.concat([tf.zeros_like(x[:, :1, :]), tf.ones_like(x[:, 1:, :])], axis=1)
    delta = delta * mask
    local_mean = tf.reduce_mean(x, axis=1, keepdims=True)
    local_std = tf.math.reduce_std(x, axis=1, keepdims=True)
    local_std_seq = tf.tile(local_std, [1, tf.shape(x)[1], 1])
    z_score = (x - local_mean) / (local_std_seq + eps)
    return tf.concat([x, delta, local_std_seq, z_score], axis=-1)


class Attention1D(keras.layers.Layer):
    def build(self, input_shape):
        feat = int(input_shape[-1])
        self.W = self.add_weight(name="W", shape=(feat, feat), initializer="glorot_uniform", trainable=True)
        self.b = self.add_weight(name="b", shape=(feat,), initializer="zeros", trainable=True)
        self.u = self.add_weight(name="u", shape=(feat,), initializer="glorot_uniform", trainable=True)
        super().build(input_shape)

    def call(self, inputs):
        proj = tf.tensordot(inputs, self.W, axes=1) + self.b
        score = tf.tensordot(tf.tanh(proj), self.u, axes=1)
        alphas = tf.nn.softmax(score, axis=1)
        context = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), axis=1)
        return context


class TimeStepDropout(layers.Layer):
    def __init__(self, drop_prob=0.2, **kwargs):
        super().__init__(**kwargs)
        self.drop_prob = float(drop_prob)

    def call(self, x, training=None):
        if training and self.drop_prob > 0.0:
            keep_prob = 1.0 - self.drop_prob
            noise_shape = (tf.shape(x)[0], tf.shape(x)[1], 1)
            mask = tf.nn.dropout(tf.ones_like(x), rate=self.drop_prob, noise_shape=noise_shape)
            return x * mask / keep_prob
        return x


# =========================================================
# 2. CUSTOM_OBJECTS DICTIONARY (Must contain EVERYTHING used during training)
# =========================================================
CUSTOM_OBJECTS = {
    "Attention1D": Attention1D,
    "TimeStepDropout": TimeStepDropout,
    # Both functions are included so load_model can find the correct one
    "features_4ch_raw": features_4ch_raw,
    "features_3ch_filtered": features_3ch_filtered,
}

# ==== Parameters (Check if they match training) ====
BIN_START_M = 0.00
BIN_STEP_M = 0.25
MODEL_PATH = "model_Filter-NONE_G0.05_A1.keras"  # Ensure this path is correct

# ==== Load Model ====
print(f"Loading model from {MODEL_PATH}...")
try:
    model = keras.models.load_model(MODEL_PATH, custom_objects=CUSTOM_OBJECTS, compile=False)
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit(1)

num_classes = int(model.output_shape[-1])
bin_centers_m = BIN_START_M + BIN_STEP_M * np.arange(num_classes, dtype=np.float32)

# ==== FastAPI App ====
app = FastAPI(title="Distance Predictor (Keyed Pairs)")


def _validate_pairs(pairs: List[Tuple[str, List[float]]]) -> None:
    if not isinstance(pairs, list) or len(pairs) == 0:
        raise HTTPException(status_code=400, detail="Expected a non-empty list of pairs.")
    for i, item in enumerate(pairs):
        if not (isinstance(item, (list, tuple)) and len(item) == 2):
            raise HTTPException(status_code=400, detail=f"Item #{i} must be [key, [5 values]].")
        key, arr = item[0], item[1]
        if not isinstance(key, str):
            raise HTTPException(status_code=400, detail=f"Item #{i}: key must be a string.")
        if not (isinstance(arr, (list, tuple)) and len(arr) == 5):
            raise HTTPException(status_code=400, detail=f"Item #{i}: expected exactly 5 RSSI values.")
        for j, v in enumerate(arr):
            try:
                float(v)
            except Exception:
                raise HTTPException(status_code=400, detail=f"Item #{i}: value #{j} is not a number.")


@app.post("/predict_pairs")
def predict_pairs(
        pairs: List[Tuple[str, List[float]]] = Body(
            ...,
            example=[
                ["Beacon-01", [-63, -64, -66, -63, -61]],
                ["Beacon-02", [-83, -61, -82, -66, -61]],
            ],
            description="Array of pairs: [ [\"ID\", [rssi1..rssi5]], ... ]"
        )
):
    """
    Receives:
    [
      ["Beacon-01", [-60, -60, -60, -60, -60]],
      ["Beacon-02", [-80, -80, -80, -80, -80]]
    ]
    Returns:
    [
      ["Beacon-01", 1.57],
      ["Beacon-02", 5.23]
    ]
    """
    _validate_pairs(pairs)

    # 1) Extract RSSI values
    rssi_only = np.asarray([vals for (_, vals) in pairs], dtype=np.float32)  # (N, 5)

    # 2) Reshape to (N, 5, 1)
    x = rssi_only.reshape(rssi_only.shape[0], 5, 1)

    # 3) Predict probabilities
    probs = model.predict(x, verbose=0).astype(np.float32)  # (N, Num_Classes)

    # 4) Calculate weighted average distance (more precise than just argmax)
    num_cls = probs.shape[1]
    all_dists = BIN_START_M + (np.arange(num_cls) * BIN_STEP_M)

    # Dot product of probabilities and distances for each row
    expected = np.sum(probs * all_dists, axis=1)

    # 5) Formulate response
    out: List[Tuple[str, float]] = []
    for (key, _), dist in zip(pairs, expected):
        out.append([key, float(dist)])

    return out


if __name__ == "__main__":
    HOST = "0.0.0.0"
    PORT = 8000
    print(f"🚀 Starting server at http://{HOST}:{PORT}")
    uvicorn.run(app, host=HOST, port=PORT)