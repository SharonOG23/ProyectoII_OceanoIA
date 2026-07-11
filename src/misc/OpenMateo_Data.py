import openmeteo_requests
import pandas as pd
import numpy as np
import requests_cache
import os

from retry_requests import retry
from datetime import date

# =====================================================
# CONFIGURACIÓN OPEN-METEO
# =====================================================

cache_session = requests_cache.CachedSession(
    ".cache",
    expire_after=3600
)

retry_session = retry(
    cache_session,
    retries=5,
    backoff_factor=0.2
)

openmeteo = openmeteo_requests.Client(
    session=retry_session
)

url = "https://marine-api.open-meteo.com/v1/marine"

# =====================================================
# RANGO HISTÓRICO
# =====================================================

START_DATE = "2022-01-01"
END_DATE = date.today().strftime("%Y-%m-%d")

# =====================================================
# MALLA COSTERA COSTA RICA
#
# Pacífico:
#   lat 8.0 - 11.0
#   lon -86.5 - -84.0
#
# Caribe:
#   lat 9.0 - 11.5
#   lon -83.5 - -81.0
# =====================================================

pacifico_lat = np.arange(8.0, 11.1, 0.25)
pacifico_lon = np.arange(-86.5, -83.9, 0.25)

caribe_lat = np.arange(9.0, 11.6, 0.25)
caribe_lon = np.arange(-83.5, -80.9, 0.25)

puntos = []

for lat in pacifico_lat:
    for lon in pacifico_lon:
        puntos.append((round(lat, 4), round(lon, 4)))

for lat in caribe_lat:
    for lon in caribe_lon:
        puntos.append((round(lat, 4), round(lon, 4)))

print(f"Puntos a evaluar: {len(puntos)}")

# =====================================================
# CONSULTA DE DATOS
# =====================================================

dataset_completo = []

for i, (lat, lon) in enumerate(puntos):

    print(
        f"[{i+1}/{len(puntos)}] "
        f"Consultando {lat}, {lon}"
    )

    try:

        params = {

            "latitude": lat,
            "longitude": lon,

            "start_date": START_DATE,
            "end_date": END_DATE,

            "hourly": [
                "wave_height",
                "wave_direction",
                "sea_surface_temperature",
                "ocean_current_velocity",
                "ocean_current_direction"
            ],

            "timezone": "auto"
        }

        response = openmeteo.weather_api(
            url,
            params=params
        )[0]

        hourly = response.Hourly()

        wave_height = hourly.Variables(0).ValuesAsNumpy()
        wave_direction = hourly.Variables(1).ValuesAsNumpy()
        sea_surface_temperature = hourly.Variables(2).ValuesAsNumpy()
        ocean_current_velocity = hourly.Variables(3).ValuesAsNumpy()
        ocean_current_direction = hourly.Variables(4).ValuesAsNumpy()

        # Si todo es NaN se descarta el punto
        if np.all(np.isnan(wave_height)):
            continue

        fechas = pd.date_range(
            start=pd.to_datetime(
                hourly.Time(),
                unit="s",
                utc=True
            ),
            end=pd.to_datetime(
                hourly.TimeEnd(),
                unit="s",
                utc=True
            ),
            freq=pd.Timedelta(
                seconds=hourly.Interval()
            ),
            inclusive="left"
        )

        df = pd.DataFrame({

            "datetime": fechas,

            "latitude": lat,
            "longitude": lon,

            "wave_height": wave_height,
            "wave_direction": wave_direction,

            "sea_surface_temperature":
                sea_surface_temperature,

            "ocean_current_velocity":
                ocean_current_velocity,

            "ocean_current_direction":
                ocean_current_direction

        })

        # Se conserva únicamente si existe
        # al menos una fila válida.

        df = df.dropna(
            subset=[
                "wave_height",
                "sea_surface_temperature"
            ],
            how="all"
        )

        if len(df) > 0:

            dataset_completo.append(df)

            print(
                f"   -> {len(df):,} registros válidos"
            )

    except Exception as e:

        print(
            f"Error en {lat},{lon}: {e}"
        )


# =====================================================
# CREAR CSV FINAL
# =====================================================

if len(dataset_completo) > 0:

    resultado = pd.concat(
        dataset_completo,
        ignore_index=True
    )

    resultado.sort_values(
        by=[
            "latitude",
            "longitude",
            "datetime"
        ],
        inplace=True
    )

    # Ruta fija donde se guardará el CSV
    carpeta_destino = r"C:\Users\fab_t\Documents\Codigo\IA_Aplicada\Proyecto_II\ProyectoII_OceanoIA\data\raw\RNN"

    # Crear la carpeta si no existe
    os.makedirs(carpeta_destino, exist_ok=True)

    # Manteniendo el nombre de archivo original
    archivo_salida = os.path.join(
        carpeta_destino,
        "CostaRica_Marine_OpenMateo_2022_Hoy.csv"
    )

    resultado.to_csv(
        archivo_salida,
        index=False,
        encoding="utf-8"
    )

    print("\n================================")
    print("CSV generado correctamente")
    print(f"Archivo: {archivo_salida}")
    print(f"Filas: {len(resultado):,}")
    print("================================")

else:

    print(
        "No se encontraron puntos válidos."
    )