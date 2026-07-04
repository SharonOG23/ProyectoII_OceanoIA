#Importación librerías

import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os

#Datos
# df = pd.read_csv("../..data/raw/RNN/datos_marinos_diarios.csv")
# df.head()



# 1. Obtiene la carpeta raíz del proyecto de forma dinámica
BASE_DIR = Path(__file__).resolve().parent.parent

# 2. Construye la ruta exacta hacia el archivo
ruta_csv = BASE_DIR / "data" / "raw" / "RNN" / "datos_marinos_diarios.csv"

# 3. Carga los datos
df = pd.read_csv(ruta_csv, delimiter=",", decimal=".", index_col=0)

print("EL set de datos tiene:\n",df.shape[0],"filas y",df.shape[1], "columnas")

print("\nSe muestran las primeras 5 filas del set de datos \n",df.head())
