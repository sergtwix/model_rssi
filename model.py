# -*- coding: utf-8 -*-
# =========================================================
# Distance Classification based on RSSI with Attention + Adaptive Features + Filters
# =========================================================

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.layers import (
    Conv1D, Dense, Input, Dropout, BatchNormalization,
    Bidirectional, GRU, LayerNormalization, MultiHeadAttention
)

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils import class_weight

# =========================================================
#  FILTERING SETTINGS
# =========================================================
# Options: 'NONE' (Baseline), 'KALMAN', 'PARTICLE'
FILTER_METHOD = 'NONE'

# Kalman Settings
KALMAN_R = 10.0  # Measurement Noise
KALMAN_Q = 0.1   # Process Noise

# Particle Settings
PARTICLE_N = 200     # Number of particles
PARTICLE_SIGMA = 2.0 # Expected RSSI noise

# -------------------- Cycle Switches --------------------
CYCLE_REG = False
CYCLE_NOISE = False
CYCLE_AUG = False
CYCLE_CONV = False
CYCLE_LSTM = False
CYCLE_GRU = False
CYCLE_DENSE = False


USE_MHA = False              # Multi-Head Self-Attention (before Bahdanau attention)
USE_MINMAX_SCALER = False    # External MinMax (not needed — normalized within the model)
USE_TIME_DROPOUT = False     # Zeroing out random timesteps (disabled by default)
TIME_DROPOUT_P = 0.0

# -------------------- Fixed Hyperparameters --------------------
FIXED_REG = 0.00005
FIXED_NOISE = 0.05
FIXED_AUG = 1
FIXED_CONV = 64
FIXED_LSTM = 64
FIXED_GRU = 64
FIXED_DENSE = 64

# -------------------- Lists for Iteration --------------------
reg_list = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001] if CYCLE_REG else [FIXED_REG]
noise_list = np.arange(0.05, 0.31, 0.05) if CYCLE_NOISE else [FIXED_NOISE]
aug_list = range(1, 6) if CYCLE_AUG else [FIXED_AUG]
conv_list = [16, 32, 64] if CYCLE_CONV else [FIXED_CONV]
lstm_list = [16, 32, 64] if CYCLE_LSTM else [FIXED_LSTM]
gru_list = [16, 32, 64] if CYCLE_GRU else [FIXED_GRU]
dense_list = [32, 64, 128] if CYCLE_DENSE else [FIXED_DENSE]

# -------------------- Data Parameters (UPDATED) --------------------
# 1. Get the absolute path to the directory where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


DATA_FOLDER_NAME = "DATA"  # <--- CHANGE THIS to your folder name

# 3. Construct the full path
INPUT_DIR = os.path.join(BASE_DIR, DATA_FOLDER_NAME)
RESULT_FILE = os.path.join(BASE_DIR, 'rezult.txt')

# Check if the directory exists
if not os.path.exists(INPUT_DIR):
    print(f"\n⚠️  CRITICAL WARNING: Data directory not found!")
    print(f"    Expected path: {INPUT_DIR}")
    print(f"    Please create a folder named '{DATA_FOLDER_NAME}' next to the script.\n")
# -------------------------------------------------------------------

COUNT = 5  # Number of points in the window (x1..x5)
INITIAL_D = 1

USE_CLASS_WEIGHTS = False  # True: Use weights, False: Trim data

if USE_CLASS_WEIGHTS:
    COUNTSAME = False
else:
    COUNTSAME = True

AUGUM = True
NUM_TESTS = 1

BIN_START_M = 0.00
BIN_STEP_M = 0.25


# =========================================================
#  FILTER CLASSES (NumPy implementation)
# =========================================================

class KalmanFilter1D:
    def __init__(self, R=10.0, Q=0.1, initial_value=0.0):
        self.R = R
        self.Q = Q
        self.x = initial_value
        self.P = 1.0

    def filter(self, measurements):
        results = []
        for z in measurements:
            self.P = self.P + self.Q
            K = self.P / (self.P + self.R)
            self.x = self.x + K * (z - self.x)
            self.P = (1 - K) * self.P
            results.append(self.x)
        return np.array(results)


class ParticleFilter1D:
    def __init__(self, num_particles=200, sigma_obs=2.0, process_std=0.5):
        self.N = num_particles
        self.sigma_obs = sigma_obs
        self.process_std = process_std
        self.particles = np.zeros(self.N)
        self.weights = np.ones(self.N) / self.N

    def filter(self, measurements):
        results = []
        if len(measurements) > 0:
            self.particles = np.random.normal(measurements[0], self.sigma_obs, self.N)

        for z in measurements:
            self.particles += np.random.normal(0, self.process_std, self.N)
            dist = self.particles - z
            self.weights = np.exp(-0.5 * (dist / self.sigma_obs) ** 2)
            self.weights += 1.e-300
            self.weights /= np.sum(self.weights)
            estimate = np.sum(self.particles * self.weights)
            results.append(estimate)
            cumulative_sum = np.cumsum(self.weights)
            cumulative_sum[-1] = 1.0
            indexes = np.searchsorted(cumulative_sum, np.random.rand(self.N))
            self.particles = self.particles[indexes]
            self.weights.fill(1.0 / self.N)

        return np.array(results)


def apply_selected_filter(df, method):
    if method == 'NONE':
        return df

    df_filtered = df.copy()
    columns_to_filter = [f'x{i + 1}' for i in range(COUNT)]

    print(f"   -> Applying filter: {method}...")

    for col in columns_to_filter:
        data_series = df[col].values

        if method == 'KALMAN':
            kf = KalmanFilter1D(R=KALMAN_R, Q=KALMAN_Q, initial_value=data_series[0])
            filtered_data = kf.filter(data_series)

        elif method == 'PARTICLE':
            pf = ParticleFilter1D(num_particles=PARTICLE_N, sigma_obs=PARTICLE_SIGMA)
            filtered_data = pf.filter(data_series)

        else:
            filtered_data = data_series

        df_filtered[col] = filtered_data

    return df_filtered


# =========================================================
#  FEATURE ENGINEERING (TensorFlow Layers)
# =========================================================

def features_3ch_filtered(x):
    x_prev = tf.roll(x, shift=1, axis=1)
    delta = x - x_prev
    mask = tf.concat([tf.zeros_like(x[:, :1, :]), tf.ones_like(x[:, 1:, :])], axis=1)
    delta = delta * mask
    local_std = tf.math.reduce_std(x, axis=1, keepdims=True)
    local_std_seq = tf.tile(local_std, [1, tf.shape(x)[1], 1])
    return tf.concat([x, delta, local_std_seq], axis=-1)


def features_4ch_raw(x):
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


# =========================================================
#  MODEL
# =========================================================

class Attention1D(layers.Layer):
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


def build_model(conv_units, lstm_units_unused, gru_units, dense_units, reg_strength, use_filtered_features):
    inp = Input(shape=(COUNT, 1), name="rssi_input")
    if use_filtered_features:
        x = layers.Lambda(features_3ch_filtered, name="feats_3ch")(inp)
    else:
        x = layers.Lambda(features_4ch_raw, name="feats_4ch")(inp)

    if USE_TIME_DROPOUT and TIME_DROPOUT_P > 0.0:
        x = TimeStepDropout(TIME_DROPOUT_P)(x)

    x = Conv1D(conv_units, kernel_size=3, padding='same', activation='relu',
               kernel_regularizer=l2(reg_strength))(x)
    x = BatchNormalization()(x)
    x = Bidirectional(GRU(gru_units, return_sequences=True))(x)
    x = Dropout(0.3)(x)

    if USE_MHA:
        attn_out = MultiHeadAttention(num_heads=4, key_dim=max(8, gru_units // 2))(x, x)
        x = LayerNormalization()(x + attn_out)

    x = Attention1D()(x)
    x = Dropout(0.3)(x)
    x = Dense(dense_units, activation='relu')(x)
    x = Dropout(0.4)(x)
    out = Dense(int(num_classes), activation='softmax')(x)
    model = keras.Model(inputs=inp, outputs=out, name="CNN-BiGRU-Att")
    return model


# =========================================================
#  MAIN SCRIPT
# =========================================================

# 1. Log file preparation
with open(RESULT_FILE, 'w', encoding='utf-8') as f:
    f.write(f"Filter:{FILTER_METHOD},Test_No,Gaus_Noise,Aug_Factor,Conv,LSTM,GRU,Dense,Reg,Accuracy,Loss,MAE_m\n")

# 2. Loading and filtering data
all_data = []
d_values = []
min_len = float('inf')

# Check if directory is valid before searching
if not os.path.exists(INPUT_DIR):
    # This check prevents glob from running on a non-existent path
    print(f"Error: Directory {INPUT_DIR} does not exist.")
    exit(1)

search_pattern = os.path.join(INPUT_DIR, 'distance_neiron_*.csv')
files = sorted(glob.glob(search_pattern))

if not files:
    print(f"⚠️ No files found matching pattern: {search_pattern}")

if COUNTSAME:
    for filepath in files:
        try:
            df = pd.read_csv(filepath)
            min_len = min(min_len, len(df))
        except Exception:
            continue

for filepath in files:
    try:
        d_str = os.path.basename(filepath).split('_')[-1].split('.')[0]
        if not d_str.isdigit(): continue
        d = int(d_str)

        df = pd.read_csv(filepath)
        if COUNTSAME and min_len != float('inf'):
            df = df.iloc[:min_len]

        print(f"Read {len(df)} rows: {os.path.basename(filepath)}")

        df = apply_selected_filter(df, FILTER_METHOD)
        X = df[[f'x{i + 1}' for i in range(COUNT)]].values.astype(np.float32)

        if USE_MINMAX_SCALER:
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler(feature_range=(0, 1))
            X = scaler.fit_transform(X)

        y = np.full(len(df), d - INITIAL_D, dtype=int)
        all_data.append((X, y))
        d_values.append(d)
    except Exception as e:
        print(f"Skipped {filepath} due to error: {e}")
        continue

if not all_data:
    raise RuntimeError("No data loaded. Check your folder path and CSV filenames.")

d_values = sorted(set(d_values))

# 3. Train/Val/Test Split
train_parts, test_parts = [], []
for X, y in all_data:
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y, shuffle=True)
    train_parts.append((X_tr, y_tr))
    test_parts.append((X_te, y_te))

X_train_base = np.vstack([X for X, _ in train_parts]).astype(np.float32)
y_train_base = np.concatenate([y for _, y in train_parts]).astype(int)
X_test = np.vstack([X for X, _ in test_parts]).astype(np.float32)
y_test = np.concatenate([y for _, y in test_parts]).astype(int)

num_classes = max(y_train_base.max(), y_test.max()) + 1
print("Classes:", d_values, "| Num classes:", num_classes)

X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_base, y_train_base, test_size=0.15, random_state=42, stratify=y_train_base
)

# === STATISTICS DISPLAY ===
print("\n" + "=" * 60)
print("📊 DATA DISTRIBUTION STATISTICS BY CLASS")
print("=" * 60)
print(f"Total classes found: {num_classes}")
print(f"{'Class (ID)':<10} | {'Distance':<10} | {'Total':<8} | {'Train':<8} | {'Val':<8} | {'Test':<8}")
print("-" * 65)

all_y_concat = np.concatenate([y for _, y in all_data])
unique_classes, total_counts = np.unique(all_y_concat, return_counts=True)
train_counts = np.bincount(y_tr, minlength=num_classes)
val_counts = np.bincount(y_val, minlength=num_classes)
test_counts = np.bincount(y_test, minlength=num_classes)

total_records = 0
for cls_idx in unique_classes:
    dist_val = BIN_START_M + cls_idx * BIN_STEP_M
    tc = total_counts[cls_idx]
    tr = train_counts[cls_idx]
    val = val_counts[cls_idx]
    te = test_counts[cls_idx]
    total_records += tc
    print(f"{cls_idx:<10} | {dist_val:.2f}m      | {tc:<8} | {tr:<8} | {val:<8} | {te:<8}")

print("-" * 65)
print(f"{'TOTAL':<23} | {total_records:<8} | {np.sum(train_counts):<8} | {np.sum(val_counts):<8} | {np.sum(test_counts):<8}")
print("=" * 60 + "\n")

X_tr_seq = X_tr.reshape(-1, COUNT, 1)
X_val_seq = X_val.reshape(-1, COUNT, 1)
X_test_seq = X_test.reshape(-1, COUNT, 1)

def gaussian_soft_targets(y_int, num_classes, sigma=0.5):
    idx = np.arange(num_classes)[None, :]
    centers = y_int[:, None]
    probs = np.exp(-0.5 * ((idx - centers) / sigma) ** 2)
    probs /= probs.sum(axis=1, keepdims=True)
    return probs.astype('float32')

y_tr_soft = gaussian_soft_targets(y_tr, num_classes, sigma=0.5)
y_val_soft = gaussian_soft_targets(y_val, num_classes, sigma=0.5)

if USE_CLASS_WEIGHTS:
    unique_classes_train = np.unique(y_tr)
    class_weights_vals = class_weight.compute_class_weight(
        class_weight='balanced', classes=unique_classes_train, y=y_tr
    )
    class_weights_dict = dict(zip(unique_classes_train, class_weights_vals))
    print("\n⚖️  AUTOMATIC CLASS WEIGHTS:")
    for cls, weight in class_weights_dict.items():
        print(f"   Class {cls}: {weight:.4f}")
else:
    class_weights_dict = None

# 4. Training Loop
for reg_strength in reg_list:
    for Gaus_Noise in noise_list:
        aug_pipe = keras.Sequential([layers.GaussianNoise(stddev=Gaus_Noise)])

        for augmentation_factor in aug_list:
            for conv_units in conv_list:
                for lstm_units in lstm_list:
                    for gru_units in gru_list:
                        for dense_units in dense_list:
                            for test_round in range(1, NUM_TESTS + 1):
                                print(f"\n=== Test {test_round} [FILTER: {FILTER_METHOD}] ===")

                                X_train_seq = np.copy(X_tr_seq)
                                y_train_soft = np.copy(y_tr_soft)

                                if AUGUM and augmentation_factor > 1:
                                    X_aug = [X_train_seq]; Y_aug = [y_train_soft]
                                    for _ in range(augmentation_factor - 1):
                                        noisy = aug_pipe(X_tr_seq, training=True).numpy()
                                        X_aug.append(noisy)
                                        Y_aug.append(y_tr_soft)
                                    X_train_seq = np.vstack(X_aug)
                                    y_train_soft = np.vstack(Y_aug)

                                is_filtered_mode = (FILTER_METHOD in ['KALMAN', 'PARTICLE'])

                                model = build_model(
                                    conv_units, lstm_units, gru_units, dense_units,
                                    reg_strength, use_filtered_features=is_filtered_mode
                                )

                                model.compile(optimizer=Adam(3e-4), loss='categorical_crossentropy', metrics=['accuracy'])

                                history = model.fit(
                                    X_train_seq, y_train_soft,
                                    validation_data=(X_val_seq, y_val_soft),
                                    epochs=150, batch_size=32,
                                    callbacks=[ReduceLROnPlateau(patience=5), EarlyStopping(patience=10, restore_best_weights=True)],
                                    class_weight=class_weights_dict, verbose=1
                                )

                                loss, acc = model.evaluate(X_test_seq, keras.utils.to_categorical(y_test, num_classes), verbose=0)
                                y_pred_prob = model.predict(X_test_seq, verbose=0)
                                y_pred = np.argmax(y_pred_prob, axis=1)

                                bin_centers_m = BIN_START_M + BIN_STEP_M * np.arange(num_classes)
                                y_pred_m = (y_pred_prob * bin_centers_m[None, :]).sum(axis=1)
                                y_true_m = bin_centers_m[y_test]
                                mae_m = float(np.mean(np.abs(y_pred_m - y_true_m)))

                                print(f"   -> Acc: {acc:.4f} | MAE: {mae_m:.3f} m")

                                name = f"Filter-{FILTER_METHOD}_G{Gaus_Noise}_A{augmentation_factor}"
                                cm = confusion_matrix(y_test, y_pred)
                                tick_labels = [f"{BIN_START_M + BIN_STEP_M * i:.2f}" for i in range(num_classes)]

                                plt.figure(figsize=(10, 8))
                                plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
                                plt.title(f'Confusion Matrix: {name}')
                                plt.colorbar()
                                ticks = np.arange(num_classes)
                                plt.xticks(ticks, tick_labels, rotation=45, ha='right')
                                plt.yticks(ticks, tick_labels)
                                plt.ylabel('True distance (m)')
                                plt.xlabel('Predicted distance (m)')

                                thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
                                for i in range(cm.shape[0]):
                                    for j in range(cm.shape[1]):
                                        plt.text(j, i, format(cm[i, j], 'd'), ha='center', va='center',
                                                 color='white' if cm[i, j] > thresh else 'black')

                                plt.text(0.5, -0.15, f"Test acc: {acc:.4f} | loss: {loss:.4f} | MAE: {mae_m:.3f}m",
                                         ha='center', va='center', transform=plt.gca().transAxes, fontsize=11)
                                plt.tight_layout()
                                # Save inside the BASE_DIR (root) to avoid losing them in data folder
                                cm_path = os.path.join(BASE_DIR, f'confusion_{name}.png')
                                plt.savefig(cm_path, dpi=150)
                                plt.close()
                                print(f"   -> Saved CM to: confusion_{name}.png")

                                model_path = os.path.join(BASE_DIR, f'model_{name}.keras')
                                model.save(model_path)
                                print(f"   -> Saved Model to: model_{name}.keras")

                                with open(RESULT_FILE, 'a', encoding='utf-8') as f:
                                    f.write(f"{FILTER_METHOD},{test_round},{Gaus_Noise},{augmentation_factor},{conv_units},{lstm_units},{gru_units},{dense_units},{reg_strength},{acc:.4f},{loss:.4f},{mae_m:.3f}\n")

print(f"\n✅ Completed. Filter used: {FILTER_METHOD}")