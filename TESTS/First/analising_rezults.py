#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FULL PIPELINE (MERGED VERSION):
1) Parsing raw log -> CSV (with calibration).
2) Inference via Keras -> time_matrix_ffill.csv.
3) Stage 1: Pair Aggregation (time_matrix_averaged).
4) Stage 2: Forecast Classification (time_matrix_averaged_class).
5) Stage 3: Reference (Ground Truth) Processing (correct_result_avg).
6) Stage 4: Final Comparison and Report (With Total cells checked).
"""

import csv
import json
import numpy as np
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from copy import deepcopy
import sys
import os
import warnings
import pandas as pd

# Keras / Tensorflow imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import load_model

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterator, **kwargs):
        return iterator

warnings.filterwarnings("ignore", message="X does not have valid feature names")

# =============================================================================
#                              CONFIGURATION
# =============================================================================

# ---------- Global Folder Settings ----------
WORK_DIR = Path("1")  # Working directory
CREATE_WORK_DIR = True  # Create if it doesn't exist
INPUTS_RELATIVE_TO_WORK_DIR = True  # Look for input files inside WORK_DIR

# ---------- Input Files (Inference) ----------
USE_RAW_INPUT = True
RAW_LOG_PATH = "input_data_corect.log"  # Raw log file
CALIBRATION_FILE = "calibration.log"  # Calibration file (row-col,val)
CSV_PATH = "output.log"  # Intermediate CSV after parsing
KERAS_MODEL_PATH = "model_Filter-NONE_G0.05_A1.keras"
ALT_H5_PATH = "model_CNN-BiLSTM-BiGRU-Attention-5points.h5"

# ---------- Inference Parameters ----------
RAW_ROWS = 30
RAW_COLS = 30
ROWS_TO_READ = None
MODEL_COUNT = 5
WINDOW_MODE = "blocks"
INCLUDE_INCOMPLETE = False
OUT_PREFIX = "time_matrix_ffill"  # Inference result

# Mapping strategy for raw model output (before aggregation)
MAP_STRATEGY = "coarse_exp"
BIN_START_M = 0.0
BIN_STEP_M = 0.25

# ---------- RESULTS PROCESSING STAGES (NEW LOGIC) ----------
ENABLE_COMPARISON = True  # Perform Stages 1-4
ACTUAL_CSV = "correct.csv"  # Input reference file (correct data)

# Stage 1 Settings (Aggregation)
# 1 - Arithmetic Mean, 2 - Minimum
MERGE_STRATEGY = 1

# Stage 2 Settings (FORECAST Classification)
THRESHOLDS_PREDICTED = [
    (0.0, 0),   # <= 0        -> 0
    (0.10, 1),  # (0, 0.1]    -> 1
    (0.55, 2),  # (0.1, 0.55] -> 2
    (1.00, 3),  # (0.55, 1.0] -> 3
    (3.00, 4),  # (1.0, 3.0]  -> 4
    (6.00, 5),  # (3.00, 6.0] -> 5
    (float('inf'), 6)  # > 6.0 -> 6
]

# Stage 3 Settings (REFERENCE Classification)
THRESHOLDS_CORRECT = [
    (0.10, 1),
    (0.55, 2),
    (1.00, 3),
    (3.00, 4),
    (6.00, 5),
    (float('inf'), 6)
]

# Output filenames (inside WORK_DIR)
FILE_INTERMEDIATE_AVG = 'time_matrix_averaged.csv'
FILE_INTERMEDIATE_CLASS = 'time_matrix_averaged_class.csv'
FILE_INTERMEDIATE_CORRECT_AVG = 'correct_result_avg.csv'
FILE_FINAL_RESULT = 'final_comparison_result.csv'
FILE_FINAL_REPORT = 'result_avg.txt'


# =============================================================================
#                                PATH HELPERS
# =============================================================================
def _ensure_work_dir():
    if CREATE_WORK_DIR:
        WORK_DIR.mkdir(parents=True, exist_ok=True)


def _resolve_input(path_like: str) -> Path:
    p = Path(path_like)
    if p.is_absolute(): return p
    if INPUTS_RELATIVE_TO_WORK_DIR: return (WORK_DIR / p)
    p1 = WORK_DIR / p
    return p1 if p1.exists() else p


def _resolve_output(path_like: str) -> Path:
    p = Path(path_like)
    return p if p.is_absolute() else (WORK_DIR / p)


def _resolve_output_prefix(prefix: str) -> Path:
    p = Path(prefix)
    return p if p.is_absolute() else (WORK_DIR / p)


# =============================================================================
#                       BLOCK 1: INFERENCE (KERAS MODEL)
# =============================================================================

try:
    register = keras.saving.register_keras_serializable
except Exception:
    from keras.utils import register_keras_serializable as register


@register(package="custom", name="features_3ch_filtered")
def features_3ch_filtered(x):
    x_prev = tf.roll(x, shift=1, axis=1)
    delta = x - x_prev
    mask = tf.concat([tf.zeros_like(x[:, :1, :]), tf.ones_like(x[:, 1:, :])], axis=1)
    delta = delta * mask
    local_std = tf.math.reduce_std(x, axis=1, keepdims=True)
    local_std_seq = tf.tile(local_std, [1, tf.shape(x)[1], 1])
    return tf.concat([x, delta, local_std_seq], axis=-1)


@register(package="custom", name="features_4ch_raw")
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


@register(package="custom", name="Attention1D")
class Attention1D(layers.Layer):
    def build(self, input_shape):
        feat = int(input_shape[-1])
        self.W = self.add_weight(name="W", shape=(feat, feat), initializer="glorot_uniform", trainable=True)
        self.b = self.add_weight(name="b", shape=(feat,), initializer="zeros", trainable=True)
        self.u = self.add_weight(name="u", shape=(feat,), initializer="glorot_uniform", trainable=True)
        super().build(input_shape)

    def call(self, inputs):
        score = tf.tensordot(tf.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b), self.u, axes=1)
        alphas = tf.nn.softmax(score, axis=1)
        context = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), axis=1)
        return context


@register(package="custom", name="TimeStepDropout")
class TimeStepDropout(layers.Layer):
    def __init__(self, drop_prob=0.2, **kwargs):
        super().__init__(**kwargs)
        self.drop_prob = float(drop_prob)

    def call(self, x, training=None):
        if training:
            keep_prob = 1.0 - self.drop_prob
            noise_shape = (tf.shape(x)[0], tf.shape(x)[1], 1)
            rnd = keep_prob + tf.random.uniform(noise_shape, 0, 1)
            mask = tf.floor(rnd)
            x = x / keep_prob * mask
        return x


def load_calibration_map(calib_path: Path) -> dict:
    calibration = {}
    if not calib_path.exists():
        print(f"[WARN] Calibration file {calib_path} not found. Skipping.")
        return calibration
    print(f"[CALIB] Reading calibration from {calib_path.name}...")
    try:
        with calib_path.open('r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                parts = line.split(',')
                if len(parts) == 2:
                    try:
                        calibration[parts[0].strip()] = float(parts[1].strip())
                    except ValueError:
                        pass
    except Exception as e:
        print(f"[ERR] Error reading calibration: {e}")
    return calibration


def parse_dense_log_to_csv(input_path: Path, output_path: Path, rows=30, cols=30, calibration_map=None):
    if calibration_map is None: calibration_map = {}
    expected = rows * cols
    written = 0
    print(f"[PARSE] Processing {input_path.name}...")

    with input_path.open('r', encoding='utf-8') as infile, \
            output_path.open('w', encoding='utf-8', newline="") as outfile:
        w = csv.writer(outfile)
        w.writerow(['datetime', 'row', 'col', 'value'])

        for line in infile:
            line = line.strip()
            if not line: continue
            parts = line.split(';')
            if len(parts) < 1 + expected: continue

            dt = parts[0]
            try:
                values = list(map(float, parts[1:1 + expected]))
            except ValueError:
                continue

            for idx, raw_val in enumerate(values):
                if raw_val < 0:
                    r = idx // cols
                    c = idx % cols
                    offset = calibration_map.get(f"{r}-{c}", 0.0)
                    final_val = int(raw_val + offset)
                    w.writerow([dt, r, c, final_val])
                    written += 1
    print(f"[PARSE] Points written: {written}. CSV -> {output_path.name}")


def _load_model_and_features():
    my_custom_objects = {
        "Attention1D": Attention1D,
        "TimeStepDropout": TimeStepDropout,
        "features_3ch_filtered": features_3ch_filtered,
        "features_4ch_raw": features_4ch_raw,
    }
    model_path = _resolve_input(KERAS_MODEL_PATH)
    if not model_path.exists():
        model_path = _resolve_input(ALT_H5_PATH)

    try:
        model = load_model(model_path, custom_objects=my_custom_objects, compile=False)
        num_classes = int(model.output_shape[-1])
        classes = list(range(1, num_classes + 1))
        return "keras", model, classes
    except Exception as e:
        sys.exit(f"Error loading model: {e}")


def coarse_from_proba(proba_vec, bin_start=BIN_START_M, bin_step=BIN_STEP_M):
    centers = bin_start + bin_step * np.arange(len(proba_vec), dtype=np.float32)
    exp_m = float(np.dot(proba_vec, centers))
    return 0, 0.0, exp_m


def run_inference_logic():
    _ensure_work_dir()
    raw_log_path = _resolve_input(RAW_LOG_PATH)
    csv_path = _resolve_input(CSV_PATH)
    calib_path = _resolve_input(CALIBRATION_FILE)

    # 1. Parsing
    if USE_RAW_INPUT:
        if not raw_log_path.exists(): sys.exit(f"Log not found {raw_log_path}")
        calib_map = load_calibration_map(calib_path)
        parse_dense_log_to_csv(raw_log_path, csv_path, rows=RAW_ROWS, cols=RAW_COLS, calibration_map=calib_map)

    # 2. Loading Data
    print("[INFERENCE] Reading CSV...")
    data_dict = defaultdict(list)
    with csv_path.open("r", encoding="utf-8") as f:
        f.seek(0)
        csv_reader = csv.reader(f)
        next(csv_reader, None)  # skip header
        for row in csv_reader:
            if len(row) < 4: continue
            try:
                dt_obj = datetime.strptime(row[0], "%m/%d/%Y %I:%M:%S %p")
                dt_str = dt_obj.strftime("%H:%M")
            except:
                dt_str = row[0]
            key = f"{row[1]}-{row[2]}"
            val = float(row[3])
            data_dict[key].append([dt_str, val])

    # 3. Windows
    windows = {}
    for key, items in data_dict.items():
        w_list = []
        if WINDOW_MODE == "blocks":
            for i in range(0, len(items), MODEL_COUNT):
                chunk = items[i:i + MODEL_COUNT]
                if len(chunk) == MODEL_COUNT:
                    start_t = chunk[-1][0]
                    vals = [x[1] for x in chunk]
                    w_list.append([start_t, vals])
        else:  # sliding
            for i in range(len(items) - MODEL_COUNT + 1):
                chunk = items[i:i + MODEL_COUNT]
                start_t = chunk[-1][0]
                vals = [x[1] for x in chunk]
                w_list.append([start_t, vals])
        if w_list: windows[key] = w_list

    # 4. Predict
    kind, model, classes = _load_model_and_features()

    X_batch = []
    meta = []
    for key, w_items in windows.items():
        for t, vals in w_items:
            X_batch.append(vals)
            meta.append((key, t))

    if not X_batch:
        print("[WARN] No data for prediction.")
        return None

    print(f"[INFERENCE] Predicting {len(X_batch)} windows...")
    X_arr = np.array(X_batch, dtype=np.float32).reshape(-1, MODEL_COUNT, 1)
    probas = model.predict(X_arr, batch_size=2048, verbose=1)

    preds_by_time = defaultdict(dict)

    for i, (key, t) in enumerate(meta):
        if MAP_STRATEGY == "coarse_exp":
            _, _, val = coarse_from_proba(probas[i])
            val = round(val, 3)
        else:
            idx = np.argmax(probas[i])
            val = int(classes[idx])

        preds_by_time[t][key] = val

    # 5. Build Matrix & Save
    times = sorted(preds_by_time.keys(), key=lambda x: x)
    all_keys = sorted(data_dict.keys(), key=lambda x: tuple(map(int, x.split('-'))))

    out_prefix = _resolve_output_prefix(OUT_PREFIX)
    out_csv = out_prefix.with_suffix(".csv")

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["time"] + all_keys)
        last_row = {k: 0 for k in all_keys}

        for t in times:
            current_data = preds_by_time[t]
            row_to_write = []
            for k in all_keys:
                if k in current_data:
                    last_row[k] = current_data[k]
                row_to_write.append(last_row[k])
            w.writerow([t] + row_to_write)

    print(f"[INFERENCE] Result saved: {out_csv}")
    return out_csv


# =============================================================================
#                       BLOCK 2: NEW PROCESSING LOGIC (4 STAGES)
# =============================================================================

def get_class_from_value(val, thresholds):
    """Universal classification function."""
    if pd.isna(val): return val
    for threshold, label in thresholds:
        if val <= threshold: return label
    return thresholds[-1][1]


def print_step_header(step_num, title):
    print("\n" + "=" * 60)
    print(f"STAGE {step_num}: {title}")
    print("=" * 60)


# --- STAGE 1 ---
def step_1_process_pairs(input_path, output_path):
    print_step_header(1, f"Aggregating pairs from {input_path}")
    if not input_path.exists():
        print(f"Error: File {input_path} not found.")
        return False
    try:
        df = pd.read_csv(input_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return False

    df.columns = df.columns.str.strip()
    if 'time' not in df.columns:
        print("Error: Column 'time' not found.")
        return False

    result_data = {'time': df['time']}
    processed_pairs = set()
    count_pairs = 0

    print(f"Merge Strategy: {'Average' if MERGE_STRATEGY == 1 else 'Minimum'}")

    # Collect all columns
    all_cols = [c for c in df.columns if c != 'time']

    for col in all_cols:
        try:
            parts = col.split('-')
            if len(parts) != 2: continue
            u, v = int(parts[0]), int(parts[1])
            if u == v: continue

            pair_key = tuple(sorted((u, v)))
            if pair_key in processed_pairs: continue
            processed_pairs.add(pair_key)

            idx1, idx2 = pair_key[0], pair_key[1]
            col_name_fw = f"{idx1}-{idx2}"
            col_name_rv = f"{idx2}-{idx1}"

            vals_1 = df[col_name_fw] if col_name_fw in df.columns else pd.Series(0, index=df.index)
            vals_2 = df[col_name_rv] if col_name_rv in df.columns else pd.Series(0, index=df.index)

            final_vals = None
            if MERGE_STRATEGY == 1:  # Mean
                sum_vals = vals_1 + vals_2
                count_non_zero = (vals_1 != 0).astype(int) + (vals_2 != 0).astype(int)
                final_vals = np.where(count_non_zero > 0, sum_vals / count_non_zero, 0)
            elif MERGE_STRATEGY == 2:  # Min
                v1_temp = vals_1.replace(0, np.inf)
                v2_temp = vals_2.replace(0, np.inf)
                min_vals = np.minimum(v1_temp, v2_temp)
                final_vals = np.where(min_vals == np.inf, 0, min_vals)

            result_data[col_name_fw] = np.round(final_vals, 3)
            count_pairs += 1
        except ValueError:
            continue

    res_df = pd.DataFrame(result_data)
    cols = sorted([c for c in res_df.columns if c != 'time'], key=lambda x: tuple(map(int, x.split('-'))))
    res_df = res_df[['time'] + cols]

    res_df.to_csv(output_path, index=False)
    print(f"[DONE] Pairs processed: {count_pairs}. Saved: {output_path.name}")
    return True


# --- STAGE 2 ---
def step_2_classify_predicted(input_path, output_path):
    print_step_header(2, f"Classifying data {input_path.name}")
    if not input_path.exists(): return False

    df = pd.read_csv(input_path)
    df.columns = df.columns.str.strip()

    data_columns = [c for c in df.columns if c != 'time']

    filtered_cols = []
    for col in data_columns:
        try:
            parts = col.split('-')
            if int(parts[0]) < int(parts[1]): filtered_cols.append(col)
        except:
            pass

    df_filtered = df[['time'] + filtered_cols].copy()

    print("Applying classes...")
    for col in filtered_cols:
        df_filtered[col] = df_filtered[col].apply(lambda x: get_class_from_value(x, THRESHOLDS_PREDICTED))
        try:
            df_filtered[col] = df_filtered[col].astype('Int64')
        except:
            pass

    df_filtered.to_csv(output_path, index=False)
    print(f"[DONE] Classified data saved: {output_path.name}")
    return True


# --- STAGE 3 ---
def step_3_process_correct(input_path, output_path):
    print_step_header(3, f"Processing reference file {input_path.name}")
    if not input_path.exists():
        print(f"Error: File {input_path} not found.")
        return False

    df = pd.read_csv(input_path)
    df.columns = df.columns.str.strip()
    if 'time' not in df.columns: return False

    pairs_map = {}
    for col in df.columns:
        if col == 'time': continue
        try:
            parts = col.split('-')
            if len(parts) == 2:
                n1, n2 = int(parts[0]), int(parts[1])
                pair_key = tuple(sorted((n1, n2)))
                if pair_key not in pairs_map: pairs_map[pair_key] = []
                pairs_map[pair_key].append(col)
        except ValueError:
            continue

    df_processed = pd.DataFrame()
    df_processed['time'] = df['time']

    for (n1, n2), cols in pairs_map.items():
        new_col_name = f"{n1}-{n2}"
        df_processed[new_col_name] = df[cols].mean(axis=1)

    data_columns = [c for c in df_processed.columns if c != 'time']
    data_columns.sort(key=lambda x: (int(x.split('-')[0]), int(x.split('-')[1])))
    df_final = df_processed[['time'] + data_columns].copy()

    print("Applying classes for reference...")
    for col in data_columns:
        df_final[col] = df_final[col].apply(lambda x: get_class_from_value(x, THRESHOLDS_CORRECT))
        try:
            df_final[col] = df_final[col].astype('Int64')
        except:
            pass

    df_final.to_csv(output_path, index=False)
    print(f"[DONE] Reference data saved: {output_path.name}")
    return True


# --- STAGE 4 ---
def step_4_compare(file_predicted, file_correct, out_csv, out_txt):
    print_step_header(4, "Comparing files and generating report")
    if not file_predicted.exists() or not file_correct.exists():
        print("Error: Missing files for comparison.")
        return

    df1 = pd.read_csv(file_predicted)
    df2 = pd.read_csv(file_correct)

    df1.columns = df1.columns.str.strip()
    df2.columns = df2.columns.str.strip()

    if 'time' in df1.columns: df1.set_index('time', inplace=True)
    if 'time' in df2.columns: df2.set_index('time', inplace=True)

    common_cols = df1.columns.intersection(df2.columns)
    common_index = df1.index.intersection(df2.index)
    print(f"Common columns: {len(common_cols)}, rows: {len(common_index)}")

    d1 = df1.loc[common_index, common_cols]  # Predicted
    d2 = df2.loc[common_index, common_cols]  # Correct

    # COMPARISON LOGIC
    cond_has_nine = (d1 == 9) | (d2 == 9)
    cond_has_zero = (d1 == 0) | (d2 == 0)
    cond_equal = (d1 == d2)

    conditions = [cond_has_nine, cond_has_zero, cond_equal]
    choices = [9, 0, 1]
    final_data = np.select(conditions, choices, default=2)

    result_df = pd.DataFrame(final_data, index=common_index, columns=common_cols)
    result_df.reset_index(inplace=True)

    # REPORT
    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append(f"DETAILED REPORT ({len(common_cols)} pairs)")
    report_lines.append(f"{'Pair':<10} | {'Total':<8} | {'Correct (1)':<12} | {'Accuracy %':<10}")
    report_lines.append("-" * 60)

    sorted_cols = sorted(common_cols, key=lambda x: tuple(map(int, x.split('-'))) if '-' in x else x)

    for col in sorted_cols:
        series = result_df[col]
        total = len(series)
        correct = (series == 1).sum()
        pct = (correct / total * 100) if total > 0 else 0.0
        report_lines.append(f"{col:<10} | {total:<8} | {correct:<12} | {pct:.2f}%")

    report_lines.append("\n" + "=" * 60)
    report_lines.append("TOP ERRORS (Only where result = 2)")
    report_lines.append("Format: Correct -> Predicted")
    report_lines.append("-" * 60)

    mismatch_mask = (final_data == 2)
    correct_vals = d2.values[mismatch_mask]
    predicted_vals = d1.values[mismatch_mask]

    if len(correct_vals) > 0:
        error_df = pd.DataFrame({'correct': correct_vals, 'predicted': predicted_vals})
        err_counts = error_df.value_counts().reset_index(name='count')
        err_counts.sort_values(by='count', ascending=False, inplace=True)
        total_errs = err_counts['count'].sum()
        for _, row in err_counts.iterrows():
            c, p, cnt = row['correct'], row['predicted'], row['count']
            pct = (cnt / total_errs * 100)
            report_lines.append(f"{c} -> {p:<15} | {cnt:<10} | {pct:.2f}%")
    else:
        report_lines.append("No type '2' errors found.")

    # Statistics
    numeric_vals = result_df.drop(columns=['time'], errors='ignore').values.flatten()
    unique, counts = np.unique(numeric_vals, return_counts=True)
    stats = dict(zip(unique, counts))
    total_cells = len(numeric_vals)

    report_lines.append("\n" + "=" * 60)
    report_lines.append("GENERAL STATISTICS")

    # -----------------------------------------------
    # Total check
    # -----------------------------------------------
    line_total = f"Total cells checked: {total_cells}"
    report_lines.append(line_total)
    print(line_total)

    print("\n--- SUMMARY ---")

    # Count "Pure Matches" (1)
    cnt_1 = stats.get(1, 0)
    pct_1 = (cnt_1 / total_cells * 100) if total_cells > 0 else 0

    # Count "Mismatches" (2 + 0 + 9)
    cnt_mismatch = stats.get(2, 0) + stats.get(0, 0) + stats.get(9, 0)
    pct_mismatch = (cnt_mismatch / total_cells * 100) if total_cells > 0 else 0

    # Output 1 (Matches)
    line1 = f"{'Matches (1)':<25}: {cnt_1:<6} ({pct_1:.2f}%)"
    report_lines.append(line1)
    print(line1)

    # Output 2 (Mismatches, including 0 and 9)
    line2 = f"{'Mismatches (All)':<25}: {cnt_mismatch:<6} ({pct_mismatch:.2f}%)"
    report_lines.append(line2)
    print(line2)

    try:
        with out_txt.open('w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        result_df.to_csv(out_csv, index=False)
        print(f"\n[SUCCESS] Report saved: {out_txt.name}")
        print(f"[SUCCESS] Comparison CSV: {out_csv.name}")
    except Exception as e:
        print(f"Error writing files: {e}")


# =============================================================================
#                                    MAIN
# =============================================================================
def main():
    print("=== START PART A: INFERENCE ===")

    # 1. Run inference (creates time_matrix_ffill.csv)
    pred_csv_path = run_inference_logic()

    if not pred_csv_path:
        print("[CRITICAL] Inference did not return a file. Stopping.")
        return

    if not ENABLE_COMPARISON:
        print("Comparison mode disabled.")
        return

    print("\n=== START PART B: PROCESSING AND COMPARISON ===")

    # Define paths
    path_avg = _resolve_output(FILE_INTERMEDIATE_AVG)
    path_class = _resolve_output(FILE_INTERMEDIATE_CLASS)
    path_correct_in = _resolve_input(ACTUAL_CSV)
    path_correct_out = _resolve_output(FILE_INTERMEDIATE_CORRECT_AVG)
    path_final_csv = _resolve_output(FILE_FINAL_RESULT)
    path_final_txt = _resolve_output(FILE_FINAL_REPORT)

    # 2. Stage 1: Forecast Aggregation
    if not step_1_process_pairs(pred_csv_path, path_avg):
        return

    # 3. Stage 2: Forecast Classification
    if not step_2_classify_predicted(path_avg, path_class):
        return

    # 4. Stage 3: Reference Processing
    if not step_3_process_correct(path_correct_in, path_correct_out):
        print(f"[SKIP] Reference file {ACTUAL_CSV} not found or error in processing.")
        return

    # 5. Stage 4: Comparison
    step_4_compare(path_class, path_correct_out, path_final_csv, path_final_txt)

    print("\n=== WORK COMPLETED ===")


if __name__ == "__main__":
    main()