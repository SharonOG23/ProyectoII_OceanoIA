#Importación librerías

import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

# MODELO
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.callbacks import EarlyStopping

from zarr.codecs import transpose

#Datos
# df = pd.read_csv("../..data/raw/RNN/datos_marinos_diarios.csv")
# df.head()



# 1. Obtiene la carpeta raíz del proyecto de forma dinámica
BASE_DIR = Path(__file__).resolve().parent.parent

# 2. Construye la ruta exacta hacia el archivo
ruta_csv = BASE_DIR / "data" / "processed" / "dataset_diario_regional.csv"

# 3. Carga los datos
df = pd.read_csv(ruta_csv, delimiter=",", decimal=".", index_col=0)

print("EL set de datos tiene:\n",df.shape[0],"filas y",df.shape[1], "columnas")

print("\nSe muestran las primeras 5 filas del set de datos \n",df.head(),"\n")

print("Datos del set de datos:\n", df.describe().transpose())


# PROCESADO

# df.plot(figsize=(12,8))
# plt.show()

# TRAN TEST Split

train_size = int(0.8 * len(df))
test_size = len(df) - train_size
train_df, test_df = df[:train_size], df[train_size:]

print("\nDistribución de los datos en set de entrenamiento y test\n",train_df.shape, test_df.shape)


# Escalado

scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(train_df)

MinMaxScaler(feature_range=(0, 1))

scaled_train = scaler.transform(train_df)
scaled_test = scaler.transform(test_df)

# minimo = scaled_train.min()
# print(minimo)

#CREADOR DE SERIE TEMPORAL

# Suponemos coger una ventana de 12 meses atrás para predecir el siguiente mes (1 muestra)
longitud = 12
tamaño_batch = 1
generador = TimeseriesGenerator(data=scaled_train, targets=scaled_train, length=longitud, batch_size=tamaño_batch)

# ¿Qué obtenemos del primer batch (coge 12 primeras muestras de train para predecir la muestra 13)?
X,y = generador[0]

print(f'Dado el array de entrada: \n{X.flatten()}')
print(f'Predecimos: \n {y}')

# CREACION  DEL MODELO

# Definimos el número de variables de salida a predecir
n_variables = scaled_train.shape[1]

# Definición del modelo

model = Sequential()
model.add(LSTM(150, activation='relu', input_shape=(longitud, n_variables))) # Podemos aumentar el número de neuronas LSTM para intentar conseguir mayor precisión ~* El 150 anterior es el número de neuronas
model.add(Dense(n_variables)) #Solo 1 variable de salida, si hubiera múltiples variables indicaríamos tantas neuronas como variables
model.compile(optimizer='adam', loss='mse') # ~* Error cuadrático médio
model.summary()


"""
model = Sequential()
model.add(LSTM(units=64, return_sequences=True, input_shape=(longitud, 1)))
model.add(LSTM(units=64))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()
"""

# ENTRENAMIENTO DEL MODELO

"""
?????
NOTA: El tamaño del conjunto de test debe ser superior a la "longitud" elegida para los batches.
"""

early_stop = EarlyStopping(monitor='val_loss',patience=4)

#Definimos generador para el conjunto de test de tal manera que nos sirva para la validación del modelo
val_generador = TimeseriesGenerator(scaled_test,scaled_test, length=longitud, batch_size=tamaño_batch)

model.fit(generador,
          epochs=20,
          validation_data=val_generador,
          callbacks=[early_stop])

losses = pd.DataFrame(model.history.history)

losses.plot()
plt.show()

# VALIDACIÓN PREVIA

print("Validacion Previa\n")

print("Train escalado:\n",scaled_train[-longitud:].shape)
print("Numero de variables:\n",n_variables)


# EVALUACIÓN DEL MODELO
## Predicción sobre el conjunto de test

# primer_batch = scaled_train[-longitud:]
# print(primer_batch.shape)
# primer_batch = primer_batch.reshape((1, longitud, n_variables))
# print(primer_batch.shape)
#
# pred = model.predict(primer_batch)
# print(pred)

#-------------------------

primer_batch = scaled_train[-longitud:]
print("Shape inicial:", primer_batch.shape)

primer_batch = primer_batch.reshape((1, longitud, n_variables))
print("Shape reshape:", primer_batch.shape)

pred = model.predict(primer_batch)
print("Predicción:", pred)


"""
Lógica en un bucle FOR para predecir en todo el conjunto de test
"""

predTest = test_predictions = []

primer_batch = scaled_train[-longitud:]
batch_actual = primer_batch.reshape((1, longitud, n_variables))

for i in range(len(test_df)):
    # Hacemos la predicción 1 time stamp (el siguiente mes). Indicar [0] para coger el valor de la predicción en lugar del array
    pred_actual = model.predict(batch_actual)[0]

    # Guardamos la predicción en la lista test_predictions
    test_predictions.append(pred_actual)

    # Actualizamos el batch para descartar el primer valor e incluir la nueva predicción (desplazamiento)
    batch_actual = np.append(batch_actual[:, 1:, :], [[pred_actual]], axis=1)

print(predTest)
