import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

FEATURES = ["Consumption", "Production", "Nuclear", "Wind", "Hydroelectric", "Oil and Gas", "Coal", "Solar", "Biomass"]
TARGET = "Consumption"
N_STEPS = 24
N_PRED = 5
DEFAULT_CSV = "data/processed/electricity_utc.csv"

def make_windows(part, x_scaler, y_scaler):
    features = x_scaler.transform(part[FEATURES]).astype("float32")
    target = y_scaler.transform(part[[TARGET]]).astype("float32")

    X, y = [], []
    for middle in range(N_STEPS, len(part) - N_PRED + 1):
        X.append(features[middle - N_STEPS:middle])
        y.append(target[middle:middle + N_PRED])
    return np.asarray(X), np.asarray(y)

def preprocess(csv_path=DEFAULT_CSV):
    df = pd.read_csv(csv_path, usecols=["DateTime", *FEATURES])
    df["DateTime"] = pd.to_datetime(df["DateTime"], utc=True)
    df = df.sort_values("DateTime").set_index("DateTime")
    gaps = df.index.to_series().diff().iloc[1:]
    if df.index.hasnans or not gaps.eq(pd.Timedelta(hours=1)).all():
        raise ValueError("Cần dữ liệu UTC liên tục theo giờ từ notebook EDA.")
    if not np.isfinite(df[FEATURES].to_numpy(dtype=float)).all():
        raise ValueError("Có giá trị thiếu hoặc vô hạn trong các cột đầu vào.")

    df["Wind"] = df["Wind"].clip(lower=0)

    train_end = int(len(df) * 0.70)
    val_end = int(len(df) * 0.85)
    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]
    if min(len(train), len(val), len(test)) < N_STEPS + N_PRED:
        raise ValueError("Mỗi tập cần ít nhất 29 giờ để tạo một mẫu.")

    x_scaler = StandardScaler().fit(train[FEATURES])
    y_scaler = StandardScaler().fit(train[[TARGET]])

    X_train, y_train = make_windows(train, x_scaler, y_scaler)
    X_val, y_val = make_windows(val, x_scaler, y_scaler)
    X_test, y_test = make_windows(test, x_scaler, y_scaler)
    return X_train, y_train, X_val, y_val, X_test, y_test, x_scaler, y_scaler


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", nargs="?", default=DEFAULT_CSV)
    args = parser.parse_args()
    X_train, y_train, X_val, y_val, X_test, y_test, _, _ = preprocess(args.csv_path)
    print("Train:", X_train.shape, y_train.shape)
    print("Validation:", X_val.shape, y_val.shape)
    print("Test:", X_test.shape, y_test.shape)