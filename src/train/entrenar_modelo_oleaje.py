"""
entrenar_modelo_oleaje.py

Entrena un modelo LSTM univariante para pronosticar wave_height (oleaje),
uno por cada region costera. Sigue la misma estructura del notebook guia
visto en clase (RNN-Forecast_Temperatura), con las siguientes anotaciones
de cambios respecto a ese ejemplo:

  [CAMBIO 1] Fuente de datos: en vez de descargar el dataset Jena Climate,
             se usa el dataset unificado propio (outputs/dataset_unificado/
             dataset_final_horario.csv), generado por build_dataset_unificado.py.

  [CAMBIO 2] Variable objetivo: "wave_height" (oleaje) en vez de "T (degC)".
             Se eligio por tener 0% de nulos y ser una medicion horaria nativa
             (no imputada), a diferencia de otras variables del dataset.

  [CAMBIO 3] Remuestreo: el dataset base es horario; aqui se agrega a
             promedio DIARIO por region antes de aplicar la ventana, para que
             "batches de 30 dias" sea literal (30 pasos = 30 dias), igual de
             simple que el notebook guia (que usa datos mensuales, ventana=12).
             Si mas adelante se quiere mayor resolucion (pronostico en horas),
             se puede repetir el mismo proceso sin el resample diario.

  [CAMBIO 4] Loop por region: el notebook guia entrena un unico modelo. Aqui
             se repite el proceso completo (train/test, scaler, modelo,
             evaluacion, forecast) para cada una de las 3 regiones, ya que
             cada zona costera tiene su propia dinamica de oleaje.

  [CAMBIO 5] TimeseriesGenerator sigue disponible en esta version de
             TensorFlow/Keras (con warning de deprecacion), se mantiene tal
             cual el notebook guia para no alejarse del ejemplo visto en
             clase. Si en el futuro se retira, el reemplazo directo es
             tf.keras.utils.timeseries_dataset_from_array.

Todo lo demas (arquitectura LSTM(150) + Dense, EarlyStopping, escalado con
MinMaxScaler ajustado solo en train, reentrenamiento final con todos los
datos para el forecast futuro) sigue el mismo patron del notebook guia.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # para poder correr sin entorno grafico y guardar .png
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.callbacks import EarlyStopping

# ------------------------------------------------------------------
# Configuracion
# ------------------------------------------------------------------
DATA_PATH = "../../data/processed/dataset_unificado/dataset_final_horario.csv"
OUTPUT_DIR = "../../models/modelo_oleaje"

VARIABLE_OBJETIVO = "wave_height"
REGIONES = ["Caribe", "Pacifico_Central", "Pacifico_Sur"]

LONGITUD_VENTANA = 30        # 30 dias de historia por batch [CAMBIO 3]
TAMANO_BATCH = 1
EPOCHS = 20
PACIENCIA_EARLY_STOP = 4
DIAS_FORECAST = 30           # dias a futuro a pronosticar
DIAS_TEST = 60               # tamano del conjunto de test (dias)


def cargar_serie_diaria(df: pd.DataFrame, region: str) -> pd.DataFrame:
    """Filtra una region y remuestrea la variable objetivo a promedio diario."""
    serie = df[df["Region"] == region].copy()
    serie["datetime"] = pd.to_datetime(serie["datetime"])
    serie = serie.set_index("datetime")[VARIABLE_OBJETIVO]
    serie_diaria = serie.resample("D").mean().to_frame(name=VARIABLE_OBJETIVO)
    serie_diaria = serie_diaria.dropna()  # por si algun dia queda sin registros
    return serie_diaria


def entrenar_region(df_diario: pd.DataFrame, region: str) -> dict:
    print(f"\n{'=' * 60}\nRegion: {region}\n{'=' * 60}")
    out_region = os.path.join(OUTPUT_DIR, region)
    os.makedirs(out_region, exist_ok=True)

    if len(df_diario) < LONGITUD_VENTANA + DIAS_TEST + 10:
        print(f"  Datos insuficientes para {region} ({len(df_diario)} dias). Se omite.")
        return {"region": region, "omitido": True, "motivo": "datos insuficientes"}

    # --- 4. Train/Test split (igual que el notebook guia) ---
    test_ind = len(df_diario) - DIAS_TEST
    train = df_diario.iloc[:test_ind]
    test = df_diario.iloc[test_ind:]

    # --- 5. Escalado (fit SOLO con train) ---
    scaler = MinMaxScaler()
    scaler.fit(train)
    scaled_train = scaler.transform(train)
    scaled_test = scaler.transform(test)

    # --- 6. Generador de series temporales ---
    generador = TimeseriesGenerator(
        scaled_train, scaled_train, length=LONGITUD_VENTANA, batch_size=TAMANO_BATCH
    )
    val_generador = TimeseriesGenerator(
        scaled_test, scaled_test, length=LONGITUD_VENTANA, batch_size=TAMANO_BATCH
    )

    # --- 7. Modelo ---
    n_variables = 1
    model = Sequential()
    model.add(LSTM(150, activation="relu", input_shape=(LONGITUD_VENTANA, n_variables)))
    model.add(Dense(n_variables))
    model.compile(optimizer="adam", loss="mse")

    # --- 8. Entrenamiento ---
    early_stop = EarlyStopping(monitor="val_loss", patience=PACIENCIA_EARLY_STOP)
    model.fit(
        generador,
        epochs=EPOCHS,
        validation_data=val_generador,
        callbacks=[early_stop],
        verbose=1,
    )

    losses = pd.DataFrame(model.history.history)
    plt.figure(figsize=(8, 5))
    losses.plot(ax=plt.gca())
    plt.title(f"Curva de perdida - {region}")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.savefig(os.path.join(out_region, "curva_perdida.png"))
    plt.close()

    # --- 9.1 Evaluacion sobre el conjunto de test ---
    test_predictions = []
    primer_batch = scaled_train[-LONGITUD_VENTANA:]
    batch_actual = primer_batch.reshape((1, LONGITUD_VENTANA, n_variables))

    for _ in range(len(test)):
        pred_actual = model.predict(batch_actual, verbose=0)[0]
        test_predictions.append(pred_actual)
        batch_actual = np.append(batch_actual[:, 1:, :], [[pred_actual]], axis=1)

    true_predictions = scaler.inverse_transform(test_predictions)
    test = test.copy()
    test["Prediccion"] = true_predictions

    rmse = np.sqrt(mean_squared_error(test[VARIABLE_OBJETIVO], test["Prediccion"]))
    print(f"  RMSE en test ({region}): {rmse:.4f} m")

    plt.figure(figsize=(12, 6))
    test[[VARIABLE_OBJETIVO, "Prediccion"]].plot(ax=plt.gca())
    plt.title(f"{VARIABLE_OBJETIVO} real vs prediccion - {region} (test)")
    plt.ylabel("metros")
    plt.savefig(os.path.join(out_region, "prediccion_vs_real_test.png"))
    plt.close()

    # --- 9.2 Reentrenamiento con TODOS los datos, para forecast futuro ---
    full_scaler = MinMaxScaler()
    scaled_full = full_scaler.fit_transform(df_diario)

    generador_full = TimeseriesGenerator(
        scaled_full, scaled_full, length=LONGITUD_VENTANA, batch_size=TAMANO_BATCH
    )

    model_full = Sequential()
    model_full.add(LSTM(150, activation="relu", input_shape=(LONGITUD_VENTANA, n_variables)))
    model_full.add(Dense(n_variables))
    model_full.compile(optimizer="adam", loss="mse")
    model_full.fit(generador_full, epochs=EPOCHS, verbose=1)

    forecast = []
    primer_batch = scaled_full[-LONGITUD_VENTANA:]
    batch_actual = primer_batch.reshape((1, LONGITUD_VENTANA, n_variables))
    for _ in range(DIAS_FORECAST):
        pred_actual = model_full.predict(batch_actual, verbose=0)[0]
        forecast.append(pred_actual)
        batch_actual = np.append(batch_actual[:, 1:, :], [[pred_actual]], axis=1)

    forecast = full_scaler.inverse_transform(forecast)
    forecast_index = pd.date_range(
        start=df_diario.index[-1] + pd.Timedelta(days=1), periods=DIAS_FORECAST, freq="D"
    )
    forecast_df = pd.DataFrame(data=forecast, index=forecast_index, columns=["Forecast"])

    plt.figure(figsize=(12, 6))
    ax = df_diario[VARIABLE_OBJETIVO].plot(label="Historico")
    forecast_df["Forecast"].plot(ax=ax, label="Forecast")
    plt.title(f"Forecast {DIAS_FORECAST} dias - {region}")
    plt.ylabel("metros")
    plt.legend()
    plt.savefig(os.path.join(out_region, "forecast_futuro.png"))
    plt.close()

    forecast_df.to_csv(os.path.join(out_region, "forecast_futuro.csv"))
    model_full.save(os.path.join(out_region, "modelo_lstm.keras"))

    return {"region": region, "omitido": False, "rmse_test": float(rmse),
            "dias_entrenamiento": len(df_diario)}


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = pd.read_csv(DATA_PATH)

    resultados = []
    for region in REGIONES:
        df_diario = cargar_serie_diaria(df, region)
        resultado = entrenar_region(df_diario, region)
        resultados.append(resultado)

    resumen_path = os.path.join(OUTPUT_DIR, "resumen_resultados.json")
    with open(resumen_path, "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False)

    print(f"\nResumen guardado en: {resumen_path}")
    for r in resultados:
        print(r)


if __name__ == "__main__":
    main()
