#Importación librerías

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os

#Datos
df = pd.read_csv(".data/raw/RNN/datos_marinos_diarios.csv")
df.head()