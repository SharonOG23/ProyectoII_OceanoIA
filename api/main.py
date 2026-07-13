"""
main.py
=======
API REST OceanoIA con FastAPI.

Endpoints:
  - POST /predict/especie  (imagen del pez)
  - POST /predict/oceano   (serie temporal de 30 días)

Ejecutar:
    uvicorn api.main:app --reload
"""

from __future__ import annotations # Nos deja usar dict[int, str] como tipo de datos para que
# Python no llegue a tener errores, es más para tener compatibilidad.

import io # Libreria que trabaja con imágenes que la guarda en la memoria de la compu.
import json # Libreria que sirve para leer y escribir archivos con formato JSON.
import sys # Libreria que nos ayuda a encontrar las carpetas del proyecto.
from pathlib import Path # Libreria que nos ayuda a manejar rutas de carpetas/archivos.
from typing import List # Libreria que nos ayuda para decir "esto es una lista de tal cosa".

# Permite a Python importar módulos desde la carpeta raíz del proyecto.
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np # Librería para trabajar con números y matrices.
# Libreria que nos importa para crear la API, los endpoint que esperan un archivo, devuelve errores
# y para subir la imagenes de las especies.
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel # Libreria que nos permite definirle algún molde de cuáles datos va a recibir.

app = FastAPI(
    title="OceanoIA API",
    description="API REST para identificación de especies, pronóstico oceánico y recomendación de pesca.",
    version="1.0.0",
) # Aquí creamos la API.

# ===================== Carga perezosa de modelos =====================
_models = {} # Aquí guardamos el modelo.
_clases_cache = None # Aquí se guarda la lista de especies ya calculados, para no tener que volver a recalcular otra ves cada que se pide una predicción.

# IMPORTANTE: debe coincidir EXACTAMENTE con IMG_SIZE del script de entrenamiento
CNN_IMG_SIZE = (160, 160)

# Rutas de los artefactos del modelo CNN (ajusta si guardas en otro lugar)
CNN_MODEL_PATH = "models/clasificador_peces_final 3.keras" # 0.98
CNN_CLASES_PATH = "models/clases.json"


def _load(key: str, path: str):
    """Carga modelos genéricos como el RNN sin problemas de anidamiento."""
    if key not in _models: # Si el modelo no está ya cargado en memoria entonces
        import tensorflow as tf # Importa TensorFlow solo aquí (para no ralentizar el arranque).
        try:
            _models[key] = tf.keras.models.load_model(path, compile=False)
            # Carga el modelo desde el archivo .keras (compile=False = solo para predecir, no reentrenar).
        except Exception as e:
            raise HTTPException(  # Si falla la carga, devuelve un error claro (503).
                status_code=503,
                detail=f"Modelo no disponible: {path}. Entrénalo primero. ({e})",
            )
    return _models[key] # Devuelve el modelo (recién cargado o ya guardado antes).

# Creamos una función, en la cual estará la ruta donde está guardado el modelo, y también
# la cantidad de clases que el modelo puede reconocer.
def _load_cnn(path: str, num_classes: int):
    if "cnn" not in _models: # Si el modelo CNN no está, lo carga, y si ya existe lo reutiliza.
        import tempfile # Importa una herramienta para crear carpetas temporales.
        import zipfile # Importamos esto ya que un .keras es en realidad un ZIP por dentro.

        import tensorflow as tf # Importa TensorFlow, la biblioteca para crear y usar redes neuronales.
        from tensorflow.keras import layers, models # Importa las capas y herramientas para construir el modelo.
        from tensorflow.keras.applications import MobileNetV2 # Importa la arquitectura MobileNetV2, una red neuronal ya diseñada para procesar imágenes.

        try: # Ejecuta el código, pero, si hay un error pasa a la sección de except.
            base_model = MobileNetV2( # Crea la base del modelo utilizando MobileNetV2.
                input_shape=CNN_IMG_SIZE + (3,), # Define un tamaño de las imágenes y el 3 representa los canales de color RGB.
                include_top=False, # Aquí no se utiliza la última capa de MobileNetV2, entonces nos permite agregar nuestras propias capas.
                weights=None,  # Los pesos reales vienen del .keras, no de imagenet.
            )
            model = models.Sequential([ # Crea un modelo donde las capas se ejecutan una después de otra.
                base_model, # Agrega MobileNetV2 como primera parte del modelo.
                layers.GlobalAveragePooling2D(), # Reduce la info de las imágenes para así obtener solo las características detectadas.
                layers.Dense(128, activation="relu"), # Agregamos una capa de 128 neuronas y usamos la función ReLU que le ayuda al modelo a aprender patrones.
                layers.Dropout(0.3), # Apaga el 30% de las neuronas mientras está el entrenamiento y evita un sobreajuste.
                layers.Dense(num_classes, activation="softmax"), # Esta es la capa de salida, y se utiliza softmax que nos devuelve la probabilidad de cada clase.
            ])

            with zipfile.ZipFile(path, "r") as z: # Abre el archivo .keras como un archivo ZIP en modo lectura.
                with tempfile.TemporaryDirectory() as tmp: # Crea una carpeta temporal para trabajar con los archivos.
                    z.extract("model.weights.h5", tmp) # Extrae del archivo .keras el archivo que contiene los pesos del modelo.
                    model.load_weights(f"{tmp}/model.weights.h5") # Carga esos pesos al modelo para que pueda realizar predicciones.

            _models["cnn"] = model # Guarda el modelo en memoria para no volver a cargarlo después.
        except Exception as e: # Si ocurre un error
            raise HTTPException( # Genera un error que será enviado al cliente de la API.
                status_code=503, # Devuelve el código 503, indicando que el servicio no está disponible.
                detail=f"Modelo no disponible: {path}. Entrénalo primero. ({e})",
            ) # Muestra un mensaje indicando que el modelo no se encontró o hubo un problema al cargarlo.
    return _models["cnn"] # Modelo CNN cargado.


def _load_clases() -> dict[int, str]: # Carga la lista de nombres de las clases.
    global _clases_cache
    if _clases_cache is None: # Guarda esa lista en memoria para no tener que crearla nuevamente.
        clases_ordenadas = sorted([ # Ordena los nombres alfabéticamente.
            "Atun_aleta_amarilla",
            "Camaron",
            "Corvina",
            "Corvina_reina",
            "Dorada",
            "Espadín_del_mar_negro",
            "Jurel",
            "Marlin_pez_vela",
            "Parg_rojo",
            "Pargo_mancha",
            "Pez_dorado",
            "Salmonete",
            "Salmonete_de_fango",
            "Tiburon_martillo",
            "Tortuga",
            "Trucha",
        ])
        _clases_cache = dict(enumerate(clases_ordenadas)) # Asigna un número a cada clase (por ejemplo, 0 - Atun_aleta_amarilla.
    return _clases_cache # Devuelve el diccionario con los números y nombres de las clases.


# Especies protegidas / de veda.
ESPECIES_PROTEGIDAS = {"Marlin_pez_vela", "Tortuga", "Tiburon_martillo"}

# ===================== Health =====================
@app.get("/") # Crea un endpoint que responde a las solicitudes GET.
def root(): # Define la función root, que se ejecuta cuando alguien entra a la ruta principal.
    return { # Devuelve la información en formato JSON.
        "name": "OceanoIA API",
        "endpoints": ["/predict/especie", "/predict/oceano"],
        "status": "ok",
    }


# ===================== CNN: especie =====================
@app.post("/predict/especie") # Crea un endpoint POST para predecir la especie de una imagen.
async def predict_especie(file: UploadFile = File(...)): # Define una función que recibe una imagen enviada por el usuario.
    """Recibe una imagen y devuelve la especie con probabilidad."""
    try: # Intenta ejecutar el código.
        from PIL import Image, UnidentifiedImageError # Importa herramientas para abrir imágenes y detectar si son inválidas.

        idx_to_clase = _load_clases() # Carga los nombres de las especies que reconoce el modelo.
        model = _load_cnn(CNN_MODEL_PATH, num_classes=len(idx_to_clase)) # Carga el modelo entrenado.

        img_bytes = await file.read() # Lee la imagen enviada por el usuario.
        if not img_bytes: # Verifica si el archivo está vacío.
            raise HTTPException(status_code=400, detail="Archivo vacío.") # Devuelve un error si no se envió una imagen.

        try:
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB").resize(CNN_IMG_SIZE) # Abre la imagen, la convierte a color RGB y la ajusta al tamaño que necesita el modelo.
        except UnidentifiedImageError: # Detecta si el archivo no es una imagen válida.
            raise HTTPException(status_code=400, detail="El archivo no es una imagen válida.") # Devuelve un error indicando que el archivo no es una imagen.

        arr = np.expand_dims(np.array(img) / 255.0, 0).astype("float32") # Convierte la imagen en un arreglo de números, normaliza sus valores entre 0 y 1 y la prepara para el modelo.

        preds = model.predict(arr, verbose=0)[0] # El modelo analiza la imagen y calcula la probabilidad de cada especie.
        idx = int(np.argmax(preds)) # Obtiene el índice de la especie con mayor probabilidad.
        especie = idx_to_clase[idx] # Convierte ese índice en el nombre de la especie.

        return { # Devuelve el resultado en formato JSON.
            "especie":   especie,
            "confianza": float(preds[idx]),
            "protegida": especie in ESPECIES_PROTEGIDAS,
            "todas": {
                idx_to_clase[i]: float(p) for i, p in enumerate(preds) # Devuelve la probabilidad obtenida para todas las especies.
            },
        }
    except HTTPException: # Si el error ya es una excepción HTTP, la vuelve a enviar.
        raise # Reenvía el mismo error sin modificarlo.
    except Exception as e: # Captura cualquier otro error inesperado.
        raise HTTPException(status_code=500, detail=str(e)) # Devuelve un error 500 indicando que ocurrió un problema interno en el servidor.


# ===================== RNN: pronóstico =====================
# Carga del modelo.
RNN_MODEL_PATH = "models/rnn_oleaje/modelo_lstm.keras"
RNN_X_SCALER_PATH = "models/rnn_oleaje/x_scaler.pkl"
RNN_Y_SCALER_PATH = "models/rnn_oleaje/y_scaler.pkl"
RNN_PHASE_ENCODER_PATH = "models/rnn_oleaje/phase_onehot.pkl"
RNN_REGION_ENCODER_PATH = "models/rnn_oleaje/region_onehot.pkl"

# Nombres de las 12 salidas del modelo, en el mismo orden que las primeras
# 12 columnas de entrada (ver tabla arriba).
RNN_SALIDAS = [
    "wave_height", "sea_surface_temperature", "ocean_current_velocity",
    "Temp_Promedio", "Temp_Minima", "Temp_Maxima", "Precipitacion",
    "Vel_Viento", "wave_direction_sin", "wave_direction_cos",
    "ocean_current_direction_sin", "ocean_current_direction_cos",
]


def _load_rnn():
    """
    Carga el modelo LSTM y los archivos necesarios para poder realizar
    las predicciones. Solo se cargan una vez para que la API sea más rápida.
    """
    if "rnn" not in _models:
        import joblib
        import tensorflow as tf
        try:
            _models["rnn"] = {
                "model": tf.keras.models.load_model(RNN_MODEL_PATH, compile=False),
                "x_scaler": joblib.load(RNN_X_SCALER_PATH),
                "y_scaler": joblib.load(RNN_Y_SCALER_PATH),
                "phase_encoder": joblib.load(RNN_PHASE_ENCODER_PATH),
                "region_encoder": joblib.load(RNN_REGION_ENCODER_PATH),
            }
        except Exception as e:
            raise HTTPException(
                status_code=503,
                detail=f"Modelo RNN no disponible: {RNN_MODEL_PATH}. ({e})",
            )
    return _models["rnn"]


class RegistroClima(BaseModel):
    """
    Representa la información del clima de una sola hora.
    Estos datos son los que recibe la API para luego hacer la predicción.
    """
    datetime: str  # formato ISO, ej: "2026-07-12T14:00:00"
    wave_height: float
    wave_direction: float  # en grados (0-360)
    sea_surface_temperature: float
    ocean_current_velocity: float
    ocean_current_direction: float  # en grados (0-360)
    Temp_Promedio: float
    Temp_Minima: float
    Temp_Maxima: float
    Precipitacion: float
    Vel_Viento: float
    Region: str  # "Caribe", "Pacifico_Central" o "Pacifico_Sur"
    PhaseName: str  # ej: "Luna llena", "Luna nueva", etc.


class OceanoSeriesIn(BaseModel):
    """
    Representa la lista completa de datos que recibe la API.
    Deben enviarse exactamente 72 registros, uno por cada hora.
    """
    serie: List[RegistroClima]


def _armar_fila_27(registro: RegistroClima, region_encoder, phase_encoder) -> list[float]:
    """
    Convierte un registro del clima en el formato que necesita el modelo.
    También transforma algunos datos, como la región, la fase lunar,
    la hora y la dirección del viento, para que el modelo pueda entenderlos.
    """
    from datetime import datetime as dt

    fecha = dt.fromisoformat(registro.datetime)
    hora = fecha.hour  # 0-23
    dia_del_anio = fecha.timetuple().tm_yday  # 1-365/366

    # Codificación cíclica: convierte un número "circular" (como la hora,
    # donde 23 está pegado a 0) en un par (seno, coseno), para que el
    # modelo entienda esa cercanía en vez de ver 0 y 23 como opuestos.
    hour_sin = np.sin(2 * np.pi * hora / 24)
    hour_cos = np.cos(2 * np.pi * hora / 24)
    doy_sin = np.sin(2 * np.pi * dia_del_anio / 365)
    doy_cos = np.cos(2 * np.pi * dia_del_anio / 365)

    wave_dir_rad = np.radians(registro.wave_direction)
    curr_dir_rad = np.radians(registro.ocean_current_direction)

    # One-hot de región y fase lunar, usando los MISMOS encoders que se
    # usaron al entrenar (para garantizar el mismo orden de columnas).
    region_1hot = region_encoder.transform([[registro.Region]])[0]
    phase_1hot = phase_encoder.transform([[registro.PhaseName]])[0]

    fila = [
        registro.wave_height,
        registro.sea_surface_temperature,
        registro.ocean_current_velocity,
        registro.Temp_Promedio,
        registro.Temp_Minima,
        registro.Temp_Maxima,
        registro.Precipitacion,
        registro.Vel_Viento,
        np.sin(wave_dir_rad),
        np.cos(wave_dir_rad),
        np.sin(curr_dir_rad),
        np.cos(curr_dir_rad),
        hour_sin,
        hour_cos,
        doy_sin,
        doy_cos,
        *region_1hot,  # 3 columnas
        *phase_1hot,  # 8 columnas
    ]
    return fila

# ===================== RNN: oceano =====================
@app.get("/predict/oceano")
def oceano_demo(zona: str):
    """
        Recibe la zona seleccionada por el usuario y devuelve un pronóstico.
        """
    # Datos para cada zona costera.
    datos = {
        "Caribe": (1.1, 29.0),
        "Pacifico_Central": (2.5, 28.2),
        "Pacifico_Sur": (3.1, 27.8),
    }

    # Busca los datos de la zona seleccionada.
    # Si la zona no existe, utiliza valores por defecto.
    oleaje, sst = datos.get(zona, (2.0, 28.0))

    # Devuelve la información en formato JSON para que
    # Streamlit y el bot puedan mostrarla al usuario.
    return {
        "metricas": {
            "oleaje_max": oleaje,
            "sst": sst,
            "periodo": 6.5
        },
        "wave_height": [ # Datos para el gráfico del oleaje.
            oleaje-0.3,
            oleaje-0.2,
            oleaje,
            oleaje+0.1,
            oleaje,
            oleaje-0.1,
            oleaje,
            oleaje+0.2
        ],
        "sea_surface_temperature": [ # Datos para el gráfico de temperatura del mar.
            sst-0.2,
            sst-0.1,
            sst,
            sst+0.1,
            sst,
            sst-0.1,
            sst,
            sst+0.2
        ]
    }
