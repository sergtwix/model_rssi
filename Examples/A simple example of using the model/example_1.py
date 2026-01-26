import numpy as np
import tensorflow as tf
from tensorflow import keras


# =========================================================
# 1. COPY FUNCTIONS (Must match the training script)
# =========================================================

def features_3ch_filtered(x):
    """
    Feature generation for filtered input.
    """
    x_prev = tf.roll(x, shift=1, axis=1)
    delta = x - x_prev
    # Mask the first element because roll wraps around
    mask = tf.concat([tf.zeros_like(x[:, :1, :]), tf.ones_like(x[:, 1:, :])], axis=1)
    delta = delta * mask

    local_std = tf.math.reduce_std(x, axis=1, keepdims=True)
    local_std_seq = tf.tile(local_std, [1, tf.shape(x)[1], 1])

    return tf.concat([x, delta, local_std_seq], axis=-1)


def features_4ch_raw(x):
    """
    Feature generation for raw input (includes Z-score).
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
    """
    Custom Attention Layer.
    """

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


# =========================================================
# 2. CONFIGURATION (Must match training settings!)
# =========================================================
BIN_START_M = 0.00  # Starting distance (Class 0)
BIN_STEP_M = 0.25  # Distance step per class

# Specify the path to your saved model file
model_path = "model_Filter-NONE_G0.05_A1.keras"

# Load the model
print(f"Loading model from {model_path}...")
try:
    model = keras.models.load_model(model_path, custom_objects={
        'features_4ch_raw': features_4ch_raw,
        'features_3ch_filtered': features_3ch_filtered,
        'Attention1D': Attention1D
    })
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit()

# =========================================================
# 3. PREPARE DATA (Example for RSSI)
# =========================================================
# Enter 5 RSSI values here
my_data = [-57, -66, -51, -57, -50]

# Reshape to match model input: (Batch_Size, Time_Steps, Features) -> (1, 5, 1)
X_test = np.array(my_data, dtype=np.float32).reshape(1, 5, 1)

# =========================================================
# 4. PREDICTION AND DISTANCE CALCULATION
# =========================================================
# Get probabilities for all classes
probs = model.predict(X_test, verbose=0)

# Find the index of the class with the highest probability
pred_class = np.argmax(probs)

# --- OPTION A: Discrete Grid Distance ---
# Simple mapping: Class ID * Step + Start
distance_grid_m = BIN_START_M + (pred_class * BIN_STEP_M)

# --- OPTION B: Weighted Distance (More Precise) ---
# Calculates the weighted average based on probabilities.
# Example: If model is 50% sure it's 1.0m and 50% sure it's 1.25m, result is 1.125m.
num_classes = probs.shape[1]
all_distances = BIN_START_M + (np.arange(num_classes) * BIN_STEP_M)
distance_weighted_m = np.sum(probs * all_distances)

# =========================================================
# 5. OUTPUT RESULTS
# =========================================================
print("\n" + "=" * 45)
print(f"📊 INPUT DATA (RSSI):     {my_data}")
print("-" * 45)
print(f"Most Probable Class (ID): {pred_class}")
print(f"Model Confidence:         {probs[0][pred_class] * 100:.2f}%")
print("-" * 45)
print(f"📍 DISTANCE (Grid):       {distance_grid_m:.2f} m")
print(f"🎯 DISTANCE (Weighted):   {distance_weighted_m:.2f} m")
print("=" * 45)