# ============================================================
# MODELO RNN/LSTM MULTISALIDA PARA SERIES TEMPORALES HORARIAS
# ============================================================
# Flujo:
# 1. Carga de datos
# 2. Eliminación automática de variables con alta nulidad
# 3. Limpieza y preprocesamiento
# 4. Escalado MinMax
# 5. Creación de secuencias de 30 días (720 horas)
# 6. Entrenamiento LSTM multisalida
# 7. Evaluación mediante MAPE, MAE, RMSE y R²
# 8. Reentrenamiento con el 100% de los datos
# 9. Forecast futuro multivariable
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM,
    Dense,
    Dropout
)

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau
)

# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

ARCHIVO_CSV = "../../data/processed/dataset_unificado/dataset_final_horario.csv"

UMBRAL_NULIDAD = 40          # %
VENTANA_DIAS = 30
HORAS_DIA = 24
LONGITUD_SECUENCIA = VENTANA_DIAS * HORAS_DIA

PORCENTAJE_TEST = 0.20

EPOCHS = 50
BATCH_SIZE = 32

# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def calcular_mape(y_real, y_pred):
    """
    Cálculo del MAPE evitando divisiones entre cero.
    """

    mascara = y_real != 0

    return np.mean(
        np.abs(
            (y_real[mascara] - y_pred[mascara])
            / y_real[mascara]
        )
    ) * 100


def crear_secuencias(datos, longitud_secuencia):
    """
    Convierte una serie temporal multivariable
    en secuencias para LSTM.
    """

    X = []
    y = []

    for i in range(longitud_secuencia, len(datos)):

        X.append(datos[i - longitud_secuencia:i])

        y.append(datos[i])

    return np.array(X), np.array(y)


# ============================================================
# 1. CARGA DE DATOS
# ============================================================

print("\n==============================")
print("Cargando dataset...")
print("==============================")

df = pd.read_csv(ARCHIVO_CSV)

df["datetime"] = pd.to_datetime(df["datetime"])

df = df.sort_values("datetime")

df = df.set_index("datetime")

print(df.head())

# ============================================================
# 2. ANÁLISIS DE NULIDAD
# ============================================================

print("\n==============================")
print("ANÁLISIS DE NULIDAD")
print("==============================")

porcentaje_nulos = (
    df.isnull().sum()
    / len(df)
) * 100

print(
    porcentaje_nulos.sort_values(
        ascending=False
    )
)

variables_eliminadas = porcentaje_nulos[
    porcentaje_nulos > UMBRAL_NULIDAD
].index.tolist()

print("\nVariables eliminadas:")

for variable in variables_eliminadas:
    print(f" - {variable}")

df = df.drop(
    columns=variables_eliminadas,
    errors="ignore"
)

# ============================================================
# 3. PREPROCESAMIENTO
# ============================================================

print("\n==============================")
print("PREPROCESAMIENTO")
print("==============================")

# Eliminamos columnas de texto
columnas_texto = [
    "Region",
    "PhaseName"
]

for columna in columnas_texto:

    if columna in df.columns:
        df.drop(
            columns=columna,
            inplace=True
        )

# Convertir a numérico por seguridad

for columna in df.columns:

    df[columna] = pd.to_numeric(
        df[columna],
        errors="coerce"
    )

# Imputación mediante mediana

df = df.fillna(df.median())

print("\nVariables finales utilizadas:")

for columna in df.columns:
    print(columna)

# ============================================================
# 4. ESCALADO
# ============================================================

print("\n==============================")
print("ESCALADO")
print("==============================")

cantidad_registros_test = int(
    len(df) * PORCENTAJE_TEST
)

indice_corte = len(df) - cantidad_registros_test

datos_train = df.iloc[:indice_corte]
datos_test = df.iloc[indice_corte:]

scaler = MinMaxScaler()

scaler.fit(datos_train)

train_escalado = scaler.transform(
    datos_train
)

test_escalado = scaler.transform(
    datos_test
)

# ============================================================
# 5. CREACIÓN DE SECUENCIAS
# ============================================================

print("\n==============================")
print("CREANDO SECUENCIAS")
print("==============================")

X_train, y_train = crear_secuencias(
    train_escalado,
    LONGITUD_SECUENCIA
)

X_test, y_test = crear_secuencias(
    test_escalado,
    LONGITUD_SECUENCIA
)

n_variables = df.shape[1]

print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("X_test :", X_test.shape)
print("y_test :", y_test.shape)

# ============================================================
# 6. CREACIÓN DEL MODELO LSTM
# ============================================================

print("\n==============================")
print("CREANDO MODELO")
print("==============================")

modelo = Sequential()

modelo.add(
    LSTM(
        128,
        return_sequences=True,
        input_shape=(
            LONGITUD_SECUENCIA,
            n_variables
        )
    )
)

modelo.add(
    Dropout(0.20)
)

modelo.add(
    LSTM(
        64,
        return_sequences=False
    )
)

modelo.add(
    Dropout(0.20)
)

modelo.add(
    Dense(
        32,
        activation="relu"
    )
)

modelo.add(
    Dense(
        n_variables
    )
)

modelo.compile(
    optimizer="adam",
    loss="mse"
)

modelo.summary()

# ============================================================
# 7. ENTRENAMIENTO
# ============================================================

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

reduccion_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=5,
    verbose=1
)

historial = modelo.fit(
    X_train,
    y_train,
    validation_data=(
        X_test,
        y_test
    ),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[
        early_stopping,
        reduccion_lr
    ],
    verbose=1
)

# ============================================================
# 8. EVALUACIÓN
# ============================================================

print("\n==============================")
print("EVALUACIÓN")
print("==============================")

predicciones_escaladas = modelo.predict(
    X_test
)

predicciones = scaler.inverse_transform(
    predicciones_escaladas
)

y_real = scaler.inverse_transform(
    y_test
)

mape_global = calcular_mape(
    y_real,
    predicciones
)

mae_global = mean_absolute_error(
    y_real,
    predicciones
)

rmse_global = np.sqrt(
    mean_squared_error(
        y_real,
        predicciones
    )
)

r2_global = r2_score(
    y_real,
    predicciones,
    multioutput="uniform_average"
)

print(f"\nMAPE : {mape_global:.2f}%")
print(f"MAE  : {mae_global:.4f}")
print(f"RMSE : {rmse_global:.4f}")
print(f"R²   : {r2_global:.4f}")

# ============================================================
# INTERPRETACIÓN DEL RESULTADO
# ============================================================

print("\n==============================")
print("INTERPRETACIÓN")
print("==============================")

if mape_global <= 10:
    print(
        "✅ El modelo cumple el objetivo "
        "(MAPE <= 10%)."
    )
else:
    print(
        "⚠️ El modelo NO cumple el objetivo "
        "(MAPE <= 10%)."
    )

    print(
        "\nRecomendaciones:"
    )

    print(
        "- Probar ventanas de 15, 20 y 45 días."
    )

    print(
        "- Probar arquitectura GRU."
    )

    print(
        "- Incorporar variables temporales."
    )

    print(
        "- Aumentar historial de entrenamiento."
    )

# ============================================================
# CURVA DE APRENDIZAJE
# ============================================================

plt.figure(figsize=(10, 5))

plt.plot(
    historial.history["loss"],
    label="Train"
)

plt.plot(
    historial.history["val_loss"],
    label="Validation"
)

plt.title("Loss del entrenamiento")
plt.legend()
plt.show()

# ============================================================
# 9. REENTRENAMIENTO CON TODOS LOS DATOS
# ============================================================

print("\n==============================")
print("REENTRENAMIENTO FINAL")
print("==============================")

scaler_final = MinMaxScaler()

datos_escalados = scaler_final.fit_transform(
    df
)

X_total, y_total = crear_secuencias(
    datos_escalados,
    LONGITUD_SECUENCIA
)

modelo_final = Sequential()

modelo_final.add(
    LSTM(
        128,
        return_sequences=True,
        input_shape=(
            LONGITUD_SECUENCIA,
            n_variables
        )
    )
)

modelo_final.add(
    Dropout(0.20)
)

modelo_final.add(
    LSTM(64)
)

modelo_final.add(
    Dropout(0.20)
)

modelo_final.add(
    Dense(
        32,
        activation="relu"
    )
)

modelo_final.add(
    Dense(
        n_variables
    )
)

modelo_final.compile(
    optimizer="adam",
    loss="mse"
)

modelo_final.fit(
    X_total,
    y_total,
    epochs=max(
        10,
        len(historial.history["loss"])
    ),
    batch_size=BATCH_SIZE,
    verbose=1
)

# ============================================================
# 10. FORECAST FUTURO
# ============================================================

HORAS_A_PREDECIR = 720

forecast = []

ventana_actual = datos_escalados[
    -LONGITUD_SECUENCIA:
]

for _ in range(HORAS_A_PREDECIR):

    entrada = ventana_actual.reshape(
        1,
        LONGITUD_SECUENCIA,
        n_variables
    )

    prediccion = modelo_final.predict(
        entrada,
        verbose=0
    )

    forecast.append(
        prediccion[0]
    )

    ventana_actual = np.vstack(
        [
            ventana_actual[1:],
            prediccion
        ]
    )

forecast = np.array(forecast)

forecast = scaler_final.inverse_transform(
    forecast
)

indice_forecast = pd.date_range(
    start=df.index[-1] + pd.Timedelta(hours=1),
    periods=HORAS_A_PREDECIR,
    freq="H"
)

forecast_df = pd.DataFrame(
    forecast,
    columns=df.columns,
    index=indice_forecast
)

print("\nForecast generado:")
print(forecast_df.head())

# ============================================================
# GUARDADO DEL MODELO
# ============================================================

modelo_final.save(
    "./../models/RNN_Marino.keras"
)

forecast_df.to_csv(
    "forecast_multisalida.csv"
)

print("\nProceso completado correctamente.")