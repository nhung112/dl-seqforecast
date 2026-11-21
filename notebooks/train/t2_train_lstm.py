import argparse

import numpy as np
import tensorflow as tf
from sklearn.metrics import r2_score
from tensorflow.keras.callbacks import EarlyStopping

from notebooks.models.t2_lstm import build_model
from notebooks.t2_preprocessing import DEFAULT_CSV, FEATURES, N_PRED, TARGET, preprocess


def inverse_target(values, scaler):
    shape = values.shape
    return scaler.inverse_transform(values.reshape(-1, 1)).reshape(shape)


def print_metrics(title, actual, predicted):
    print(f"\n{title}")
    print(f"{'Step':<9} {'R²':>10} {'NSE':>10} {'MAE (MWh)':>12} {'RMSE (MWh)':>12}")
    for step in range(actual.shape[1]):
        print_metric_row(f"+{step + 1}", actual[:, step], predicted[:, step])
    print_metric_row("Overall", actual.ravel(), predicted.ravel())


def print_metric_row(label, actual, predicted):
    errors = actual - predicted
    denominator = np.sum((actual - actual.mean()) ** 2)
    nse = 1 - np.sum(errors ** 2) / denominator if denominator > 0 else np.nan
    r2 = r2_score(actual, predicted, force_finite=False) if denominator > 0 else np.nan
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors ** 2))
    print(f"{label:<9} {r2:>10.4f} {nse:>10.4f} {mae:>12.2f} {rmse:>12.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", nargs="?", default=DEFAULT_CSV)
    parser.add_argument("--no-attention", action="store_true")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    tf.keras.utils.set_random_seed(args.seed)
    X_train, y_train, X_val, y_val, X_test, y_test, x_scaler, y_scaler = preprocess(
        args.csv_path
    )

    model = build_model(X_train.shape[1:], use_attention=not args.no_attention)
    model.summary()
    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True)
        ],
    )

    y_pred = model.predict(X_test, batch_size=args.batch_size, verbose=0)
    actual = inverse_target(y_test, y_scaler).squeeze(axis=-1)
    predicted = inverse_target(y_pred, y_scaler).squeeze(axis=-1)

    # Với cửa sổ 24 giờ, 5 vị trí đầu khớp 5 giờ dự báo của ngày trước.
    target_index = FEATURES.index(TARGET)
    yesterday = X_test[:, :N_PRED, target_index]
    yesterday = yesterday * x_scaler.scale_[target_index] + x_scaler.mean_[target_index]

    print_metrics("Seasonal naive (24h)", actual, yesterday)
    title = "LSTM Encoder–Decoder"
    if not args.no_attention:
        title += " + Multi-Head Attention"
    print_metrics(title, actual, predicted)


if __name__ == "__main__":
    main()
