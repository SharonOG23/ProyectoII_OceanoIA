"""
pipeline_rnn_oleaje.py

Pipeline completo (reemplaza a preparar_ventanas_rnn.py + entrenar_rnn.py,
ahora en un solo archivo): desde el CSV crudo hasta el modelo LSTM
entrenado y evaluado.

Convención: colócalo en proyecto/eda/pipeline_rnn_oleaje.py, junto a
build_dataset_unificado.py. Run igual que los demás scripts.

PARTE A - Preprocesamiento
  1. Carga dataset_final_horario.csv.
  2. Imputa nulos reales (sea_surface_temperature, ocean_current_velocity/
     direction) por interpolación temporal lineal, por región.
  3. Codifica wave_direction y ocean_current_direction (ángulos 0-360°)
     como pares seno/coseno, tanto en features como en targets. Esto
     evita el problema de "salto" en 0°/360° (ej. 359° y 2° son casi
     el mismo ángulo pero como número crudo la diferencia es 357) que
     dañaba tanto el entrenamiento como la métrica de tolerancia.
  4. Agrega tiempo cíclico (hora, día del año) y codifica Region/PhaseName.
  5. Arma ventanas de 72h de entrada -> salidas a +24h/+48h/+72h.
  6. Split 70/15/15 POR REGIÓN (no global): Pacífico Sur solo tiene
     datos desde 2025-07, así que un corte de fecha único la dejaría
     sin datos en train.
  7. Escala con StandardScaler ajustado SOLO con train.

PARTE B - Entrenamiento
  8. LSTM(64)->LSTM(32)->Dense(32)->3 cabezas (una por horizonte).
  9. Pérdida Huber (más robusta a los picos de Precipitacion que MSE).
  10. Evalúa en test con MAE/RMSE/R2 y una métrica de "accuracy con
      tolerancia" AJUSTADA POR TIPO DE VARIABLE:
        - Variables angulares (wave_direction, ocean_current_direction):
          distancia circular <= 15°.
        - Precipitacion: tolerancia absoluta ±2 mm (una tolerancia
          relativa no tiene sentido en una variable que puede valer 0).
        - Resto (wave_height, sea_surface_temperature,
          ocean_current_velocity, Temp_*, Vel_Viento): dentro de ±10%
          del valor real.

SOBRE EL 90% DE ACCURACY: es un objetivo razonable para las variables
lentas/estables (temperaturas, sea_surface_temperature) y, con la
codificación circular, para las direcciones. Para wave_height,
ocean_current_velocity y sobre todo Precipitacion (variables ruidosas
hora a hora) un 90% bajo estas tolerancias sigue siendo optimista pese
a las mejoras; el resumen final lo deja explícito por variable en vez
de prometerlo de forma pareja para todas.
"""

import os
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
DATA_PATH = "../../data/processed/dataset_unificado/dataset_final_horario.csv"  # relativo a eda/
OUT_DIR = "../../models/rnn_oleaje"

INPUT_HOURS = 72
HORIZONS = [24, 48, 72]
MAX_HORIZON = max(HORIZONS)

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15

COLS_TO_INTERPOLATE = [
    "sea_surface_temperature", "ocean_current_velocity", "ocean_current_direction",
]
CIRCULAR_COLS = ["wave_direction", "ocean_current_direction"]
LINEAR_REG_TARGETS = [
    "wave_height", "sea_surface_temperature", "ocean_current_velocity",
    "Temp_Promedio", "Temp_Minima", "Temp_Maxima", "Precipitacion", "Vel_Viento",
]
# columnas de regresión finales = lineales + sin/cos de cada circular
REG_TARGET_COLS = LINEAR_REG_TARGETS + [f"{c}_{t}" for c in CIRCULAR_COLS for t in ("sin", "cos")]

# tolerancias para la métrica de "accuracy"
TOLERANCE_REL = 0.10          # +/-10% para variables normales
PRECIP_ABS_TOL_MM = 2.0       # +/-2 mm para Precipitacion (tolerancia relativa no aplica cerca de 0)
ANGLE_ABS_TOL_DEG = 15.0      # +/-15 grados (distancia circular) para direcciones

EPOCHS = 80
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
RANDOM_SEED = 42

os.makedirs(OUT_DIR, exist_ok=True)
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ----------------------------------------------------------------------
# PARTE A: PREPROCESAMIENTO
# ----------------------------------------------------------------------
def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["datetime"])
    df = df.sort_values(["Region", "datetime"]).reset_index(drop=True)
    return df


def impute_nulls(df):
    before = df[COLS_TO_INTERPOLATE].isnull().sum().sum()
    pieces = []
    for region, group in df.groupby("Region", sort=False):
        group = group.sort_values("datetime").set_index("datetime")
        group[COLS_TO_INTERPOLATE] = (
            group[COLS_TO_INTERPOLATE].interpolate(method="time", limit_direction="both")
        )
        pieces.append(group.reset_index())
    df = pd.concat(pieces, ignore_index=True)
    df = df.sort_values(["Region", "datetime"]).reset_index(drop=True)
    after = df[COLS_TO_INTERPOLATE].isnull().sum().sum()
    print(f"Nulos en {COLS_TO_INTERPOLATE}: {before} -> {after} tras interpolación.")
    return df


def add_circular_encoding(df):
    for col in CIRCULAR_COLS:
        rad = np.deg2rad(df[col].values)
        df[f"{col}_sin"] = np.sin(rad)
        df[f"{col}_cos"] = np.cos(rad)
    return df


def add_time_features(df):
    df["hour_sin"] = np.sin(2 * np.pi * df["datetime"].dt.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["datetime"].dt.hour / 24)
    doy = df["datetime"].dt.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    return df


def encode_categoricals(df):
    from sklearn.preprocessing import OneHotEncoder
    region_ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    phase_ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    region_ohe.fit(df[["Region"]])
    phase_ohe.fit(df[["PhaseName"]])

    region_df = pd.DataFrame(
        region_ohe.transform(df[["Region"]]),
        columns=region_ohe.get_feature_names_out(["Region"]), index=df.index,
    )
    phase_df = pd.DataFrame(
        phase_ohe.transform(df[["PhaseName"]]),
        columns=phase_ohe.get_feature_names_out(["PhaseName"]), index=df.index,
    )
    df = pd.concat([df, region_df, phase_df], axis=1)
    return df, region_ohe, phase_ohe


def contiguous_segments(sub_df):
    dt = sub_df["datetime"].values
    gap = np.diff(dt) != np.timedelta64(1, "h")
    return np.concatenate(([0], np.cumsum(gap)))


def compute_date_cutoffs_per_region(df):
    cutoffs = {}
    for region, sub in df.groupby("Region"):
        dmin, dmax = sub["datetime"].min(), sub["datetime"].max()
        total_h = (dmax - dmin) / pd.Timedelta(hours=1)
        train_end = dmin + pd.Timedelta(hours=total_h * TRAIN_FRAC)
        val_end = dmin + pd.Timedelta(hours=total_h * (TRAIN_FRAC + VAL_FRAC))
        cutoffs[region] = (train_end, val_end)
    return cutoffs


def build_windows(df, feature_cols, cutoffs):
    splits = {"train": [], "val": [], "test": []}
    for region, sub in df.groupby("Region"):
        train_end, val_end = cutoffs[region]
        sub = sub.reset_index(drop=True)
        sub = sub.copy()
        sub["_seg"] = contiguous_segments(sub)

        for _, seg in sub.groupby("_seg"):
            seg = seg.reset_index(drop=True)
            n = len(seg)
            span = INPUT_HOURS + MAX_HORIZON
            if n < span:
                continue

            feats = seg[feature_cols].values.astype("float32")
            reg_vals = seg[REG_TARGET_COLS].values.astype("float32")

            for i in range(0, n - span + 1):
                last_idx = i + INPUT_HOURS - 1
                window_end_dt = seg["datetime"].iloc[last_idx]
                split = "train" if window_end_dt <= train_end else (
                    "val" if window_end_dt <= val_end else "test"
                )
                X = feats[i:i + INPUT_HOURS]
                y_reg = {h: reg_vals[last_idx + h] for h in HORIZONS}
                splits[split].append((X, y_reg, region, window_end_dt))
    return splits


def preprocess():
    from sklearn.preprocessing import StandardScaler

    df = load_data()
    df = impute_nulls(df)
    df = add_circular_encoding(df)
    df = add_time_features(df)
    df, region_ohe, phase_ohe = encode_categoricals(df)

    cutoffs = compute_date_cutoffs_per_region(df)

    feature_cols = (
        LINEAR_REG_TARGETS
        + [f"{c}_{t}" for c in CIRCULAR_COLS for t in ("sin", "cos")]
        + ["hour_sin", "hour_cos", "doy_sin", "doy_cos"]
        + list(region_ohe.get_feature_names_out(["Region"]))
        + list(phase_ohe.get_feature_names_out(["PhaseName"]))
    )

    splits = build_windows(df, feature_cols, cutoffs)

    x_scaler = StandardScaler().fit(np.concatenate([w[0] for w in splits["train"]], axis=0))

    n_reg = len(REG_TARGET_COLS)
    y_scaler = StandardScaler().fit(
        np.array([[w[1][h] for h in HORIZONS] for w in splits["train"]]).reshape(-1, n_reg)
    )

    data = {}
    for split_name, windows in splits.items():
        X = np.stack([w[0] for w in windows])
        shp = X.shape
        X = x_scaler.transform(X.reshape(-1, shp[-1])).reshape(shp).astype("float32")
        y = {}
        for h in HORIZONS:
            y_h = np.stack([w[1][h] for w in windows]).astype("float32")
            y[h] = y_scaler.transform(y_h).astype("float32")
        data[split_name] = (X, y)
        print(f"{split_name}: {len(windows)} ventanas "
              f"({pd.Series([w[2] for w in windows]).value_counts().to_dict()})")

    joblib.dump(x_scaler, os.path.join(OUT_DIR, "x_scaler.pkl"))
    joblib.dump(y_scaler, os.path.join(OUT_DIR, "y_scaler.pkl"))
    joblib.dump(region_ohe, os.path.join(OUT_DIR, "region_onehot.pkl"))
    joblib.dump(phase_ohe, os.path.join(OUT_DIR, "phase_onehot.pkl"))

    return data, y_scaler, feature_cols, cutoffs


# ----------------------------------------------------------------------
# PARTE B: MODELO Y ENTRENAMIENTO
# ----------------------------------------------------------------------
def build_model(n_timesteps, n_features, n_targets):
    inputs = keras.Input(shape=(n_timesteps, n_features), name="ventana_entrada")
    x = layers.LSTM(64, return_sequences=True)(inputs)
    x = layers.Dropout(0.2)(x)
    x = layers.LSTM(32)(x)
    x = layers.Dropout(0.2)(x)
    tronco = layers.Dense(32, activation="relu")(x)

    outputs = {f"y_reg_{h}h": layers.Dense(n_targets, name=f"y_reg_{h}h")(tronco) for h in HORIZONS}
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss={f"y_reg_{h}h": keras.losses.Huber() for h in HORIZONS},
        metrics={f"y_reg_{h}h": ["mae"] for h in HORIZONS},
    )
    return model


def circular_error_deg(true_deg, pred_deg):
    diff = np.abs(true_deg - pred_deg) % 360
    return np.minimum(diff, 360 - diff)


def evaluate(model, X_test, y_test_scaled, y_scaler):
    preds = model.predict(X_test, batch_size=BATCH_SIZE, verbose=0)
    lines = []
    achieved_90 = []
    below_90 = []

    for h in HORIZONS:
        y_true = y_scaler.inverse_transform(y_test_scaled[h])
        y_pred = y_scaler.inverse_transform(preds[f"y_reg_{h}h"])
        lines.append(f"--- Horizonte +{h}h (n={len(y_true)}) ---")

        col = {name: j for j, name in enumerate(REG_TARGET_COLS)}

        for var in LINEAR_REG_TARGETS:
            j = col[var]
            yt, yp = y_true[:, j], y_pred[:, j]
            mae = np.mean(np.abs(yt - yp))
            rmse = np.sqrt(np.mean((yt - yp) ** 2))
            r2 = 1 - np.sum((yt - yp) ** 2) / max(np.sum((yt - yt.mean()) ** 2), 1e-9)

            if var == "Precipitacion":
                acc = np.mean(np.abs(yt - yp) <= PRECIP_ABS_TOL_MM) * 100
                tol_desc = f"±{PRECIP_ABS_TOL_MM}mm"
            else:
                tol = np.maximum(np.abs(yt) * TOLERANCE_REL, 1e-3)
                acc = np.mean(np.abs(yt - yp) <= tol) * 100
                tol_desc = f"±{int(TOLERANCE_REL*100)}%"

            lines.append(f"  {var:26s} MAE={mae:8.4f} RMSE={rmse:8.4f} R2={r2:6.3f} "
                         f"accuracy({tol_desc})={acc:5.1f}%")
            (achieved_90 if acc >= 90 else below_90).append(f"{var}@+{h}h ({acc:.1f}%)")

        for cvar in CIRCULAR_COLS:
            sin_j, cos_j = col[f"{cvar}_sin"], col[f"{cvar}_cos"]
            true_deg = (np.degrees(np.arctan2(y_true[:, sin_j], y_true[:, cos_j])) + 360) % 360
            pred_deg = (np.degrees(np.arctan2(y_pred[:, sin_j], y_pred[:, cos_j])) + 360) % 360
            cerr = circular_error_deg(true_deg, pred_deg)
            mae_circ = cerr.mean()
            acc = np.mean(cerr <= ANGLE_ABS_TOL_DEG) * 100
            lines.append(f"  {cvar:26s} MAE_circular={mae_circ:8.4f}° "
                         f"accuracy(±{int(ANGLE_ABS_TOL_DEG)}° circular)={acc:5.1f}%")
            (achieved_90 if acc >= 90 else below_90).append(f"{cvar}@+{h}h ({acc:.1f}%)")

    return "\n".join(lines), achieved_90, below_90


def main():
    data, y_scaler, feature_cols, cutoffs = preprocess()
    X_train, y_train = data["train"]
    X_val, y_val = data["val"]
    X_test, y_test = data["test"]

    n_timesteps, n_features = X_train.shape[1], X_train.shape[2]
    n_targets = len(REG_TARGET_COLS)

    model = build_model(n_timesteps, n_features, n_targets)
    model.summary()

    y_train_dict = {f"y_reg_{h}h": y_train[h] for h in HORIZONS}
    y_val_dict = {f"y_reg_{h}h": y_val[h] for h in HORIZONS}

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6),
    ]

    history = model.fit(
        X_train, y_train_dict,
        validation_data=(X_val, y_val_dict),
        epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks, verbose=2,
    )

    model_path = os.path.join(OUT_DIR, "modelo_lstm.keras")
    model.save(model_path)
    pd.DataFrame(history.history).to_csv(os.path.join(OUT_DIR, "historial_entrenamiento.csv"), index=False)

    eval_text, achieved_90, below_90 = evaluate(model, X_test, y_test, y_scaler)

    summary = [
        f"Modelo guardado en: {model_path}",
        f"Epochs corridas: {len(history.history['loss'])} (máx {EPOCHS}, early stopping)",
        f"Mejor val_loss: {min(history.history['val_loss']):.4f}",
        "",
        "Variables/horizontes que SÍ alcanzaron accuracy >= 90% (con la",
        "tolerancia correspondiente a cada tipo de variable):",
        ("  " + ", ".join(achieved_90)) if achieved_90 else "  (ninguna)",
        "",
        "Variables/horizontes por DEBAJO de 90% (esperado para variables",
        "ruidosas hora a hora como oleaje, corrientes y precipitación,",
        "incluso con un modelo bien entrenado):",
        ("  " + ", ".join(below_90)) if below_90 else "  (ninguna)",
        "",
        "Evaluación completa en test (escala original):",
        eval_text,
    ]

    with open(os.path.join(OUT_DIR, "resumen_entrenamiento.txt"), "w") as f:
        f.write("\n".join(summary))

    print("\n".join(summary))


if __name__ == "__main__":
    main()
