from pathlib import Path
import numpy as np
import wfdb


# Map MIT-BIH annotation symbols into 5 classes 
LABEL_MAP = {
    "N": 0, "L": 0, "R": 0, "e": 0, "j": 0,  # Normal
    "A": 1, "a": 1, "J": 1, "S": 1,          # SVEB
    "V": 2, "E": 2,                          # VEB
    "F": 3,                                  # Fusion
    "/": 4, "f": 4, "Q": 4                   # Unclassifiable
}

CLASS_NAMES = {
    0: "Normal",
    1: "SVEB",
    2: "VEB",
    3: "Fusion",
    4: "Unclassifiable"
}


def load_record(record_path: str):
  
    record = wfdb.rdrecord(record_path)
    annotation = wfdb.rdann(record_path, "atr")
    return record, annotation


def extract_beats(record, annotation, window_size=200, channel=0):
  
        #X -> shape (num_samples, signal_length)
        #y -> shape (num_samples,)
    
    signal = record.p_signal[:, channel]
    beat_samples = annotation.sample
    beat_symbols = annotation.symbol

    X = []
    y = []

    for idx, symbol in zip(beat_samples, beat_symbols):
        if symbol not in LABEL_MAP:
            continue

        start = idx - window_size
        end = idx + window_size

        if start < 0 or end > len(signal):
            continue

        beat_segment = signal[start:end]
        label = LABEL_MAP[symbol]

        X.append(beat_segment)
        y.append(label)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


def normalize_beats(X: np.ndarray):
    
    #Normalize each beat independently to zero mean / unit variance.
    
    mean = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True) + 1e-8
    return (X - mean) / std


def load_and_process_record(record_path: str, window_size=200, channel=0):
    record, annotation = load_record(record_path)
    X, y = extract_beats(record, annotation, window_size=window_size, channel=channel)
    X = normalize_beats(X)
    return X, y


def load_multiple_records(record_paths, window_size=200, channel=0):
 
        #X_all -> shape (total_samples, signal_length)
        #y_all -> shape (total_samples,)
    
    X_list = []
    y_list = []

    for record_path in record_paths:
        print(f"Loading record: {record_path}")
        X, y = load_and_process_record(record_path, window_size=window_size, channel=channel)

        if len(X) == 0:
            print(f"Skipping {record_path}: no valid beats extracted")
            continue

        print(f"  Extracted {len(X)} beats")
        X_list.append(X)
        y_list.append(y)

    if not X_list:
        raise ValueError("No valid data was loaded from the provided records.")

    X_all = np.concatenate(X_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)

    return X_all, y_all


def print_class_distribution(y):
    unique, counts = np.unique(y, return_counts=True)
    print("Class distribution:")
    for cls, count in zip(unique, counts):
        class_name = CLASS_NAMES.get(int(cls), f"Class {cls}")
        print(f"  {int(cls)} ({class_name}): {count}")
