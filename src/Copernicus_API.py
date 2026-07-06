from pathlib import Path
import sys

try:
    import copernicusmarine
except ImportError as error:
    print("Error al importar 'copernicusmarine'.")
    print("El problema parece estar relacionado con una dependencia, probablemente 'urllib3'.")
    print()
    print("Solución recomendada:")
    print("1. Verifica que no exista un archivo o carpeta llamado 'urllib3.py', 'urllib3/',")
    print("   'pystac.py' o 'copernicusmarine.py' dentro de tu proyecto.")
    print("2. Reinstala las dependencias con:")
    print("   python -m pip uninstall -y urllib3 pystac copernicusmarine")
    print("   python -m pip install --no-cache-dir urllib3 pystac copernicusmarine")
    print()
    print(f"Detalle técnico: {error}")
    sys.exit(1)

import xarray as xr
import pandas as pd
import folium

# ---------------------------------------------------------------------------
# CONFIGURACIÓN DE RUTAS
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
CARPETA_DATOS = BASE_DIR / "data" / "raw" / "RNN"
CARPETA_DATOS.mkdir(parents=True, exist_ok=True)

ARCHIVO_FISICA_NC = CARPETA_DATOS / "costa_rica_fisica.nc"
ARCHIVO_VIENTO_NC = CARPETA_DATOS / "costa_rica_viento.nc"
ARCHIVO_DATOS_CSV = CARPETA_DATOS / "Copernicus_data.csv"
ARCHIVO_MAPA_HTML = BASE_DIR / "mapa_costa_rica.html"

# ---------------------------------------------------------------------------
# CONFIGURACIÓN DE PARÁMETROS
# ---------------------------------------------------------------------------

COORDENADAS_CR = {
    "minimum_longitude": -86.0,
    "maximum_longitude": -82.5,
    "minimum_latitude": 8.0,
    "maximum_latitude": 11.5,
}

FECHA_INICIO = "2020-01-01T00:00:00"

# Recomendación:
# No uses una fecha futura si el dataset todavía no contiene datos hasta esa fecha.
# Puedes ajustarla según la disponibilidad real del producto en Copernicus.
FECHA_FIN = "2026-07-5T23:59:59"

# ---------------------------------------------------------------------------
# PASO 1: DESCARGA DE DATOS DESDE COPERNICUS MARINE
# ---------------------------------------------------------------------------

print("Iniciando descarga de datos desde Copernicus...")

# Esto guarda tus credenciales en el entorno de ejecución para que no se detenga a pedirlas
copernicusmarine.login(username="fbrenes", password="4HQN6J58-i8ig")

print("-> Descargando datos físicos: temperatura superficial y altura del mar...")

try:
    copernicusmarine.subset(
        dataset_id="cmems_mod_glo_phy_anfc_0.083deg_P1D-m",
        variables=["tob"],
        start_datetime=FECHA_INICIO,
        end_datetime=FECHA_FIN,
        output_filename=str(ARCHIVO_FISICA_NC),
        **COORDENADAS_CR,
    )
except Exception as e:
    import traceback
    traceback.print_exc()

# print("-> Descargando datos de viento...")
#
# try:
#     copernicusmarine.subset(
#         dataset_id="cmems_obs-wind_glo_phy_my_l4_0.25deg_PT1D",
#         variables=["wind_speed"],
#         start_datetime=FECHA_INICIO,
#         end_datetime=FECHA_FIN,
#         output_filename=str(ARCHIVO_VIENTO_NC),
#         **COORDENADAS_CR,
#     )
# except Exception as e:
#     print(f"Error descargando viendo: {e}")

print("¡Descargas de archivos NetCDF finalizadas con éxito!")

# ---------------------------------------------------------------------------
# PASO 2: PROCESAMIENTO Y CONVERSIÓN A FORMATO CSV
# ---------------------------------------------------------------------------

print("\nProcesando archivos NetCDF y convirtiendo a CSV...")

with xr.open_dataset(ARCHIVO_FISICA_NC) as ds_fisica, xr.open_dataset(ARCHIVO_VIENTO_NC) as ds_viento:
    df_fisica = ds_fisica.to_dataframe().reset_index()
    df_viento = ds_viento.to_dataframe().reset_index()

if "depth" in df_fisica.columns:
    profundidad_minima = df_fisica["depth"].min()
    df_fisica = df_fisica[df_fisica["depth"] == profundidad_minima]

df_fisica = df_fisica.rename(
    columns={
        "latitude": "Latitud",
        "longitude": "Longitud",
        "time": "Fecha_Hora",
        "thetao": "Temperatura_Mar_C",
        "zos": "Altura_Mar_Mareas_m",
    }
)

df_viento = df_viento.rename(
    columns={
        "latitude": "Latitud",
        "longitude": "Longitud",
        "time": "Fecha_Hora",
        "wind_speed": "Velocidad_Viento_ms",
    }
)

df_final = pd.concat([df_fisica, df_viento], ignore_index=True)

df_final.to_csv(ARCHIVO_DATOS_CSV, index=False)

print(f"¡Archivo CSV guardado exitosamente como: '{ARCHIVO_DATOS_CSV}'!")

# ---------------------------------------------------------------------------
# PASO 3: CREACIÓN DEL MAPA INTERACTIVO CON FOLIUM
# ---------------------------------------------------------------------------

print("\nGenerando mapa interactivo de Costa Rica...")

mapa_cr = folium.Map(
    location=[9.7489, -83.7534],
    zoom_start=8,
    tiles="OpenStreetMap",
)

df_mapa = df_fisica.dropna(subset=["Temperatura_Mar_C"])

if df_mapa.empty:
    print("No hay datos válidos de temperatura para generar marcadores en el mapa.")
else:
    df_mapa = df_mapa.sample(n=min(500, len(df_mapa)), random_state=42)

    for _, fila in df_mapa.iterrows():
        texto_popup = f"""
        <b>Coordenadas:</b> {fila['Latitud']:.2f}, {fila['Longitud']:.2f}<br>
        <b>Temp. Mar:</b> {fila['Temperatura_Mar_C']:.2f} °C<br>
        <b>Anomalía Marea SSH:</b> {fila['Altura_Mar_Mareas_m']:.2f} m
        """

        folium.CircleMarker(
            location=[fila["Latitud"], fila["Longitud"]],
            radius=4,
            popup=folium.Popup(texto_popup, max_width=300),
            color="blue" if fila["Temperatura_Mar_C"] < 25 else "red",
            fill=True,
            fill_opacity=0.7,
        ).add_to(mapa_cr)

mapa_cr.save(ARCHIVO_MAPA_HTML)

print(
    f"¡Mapa interactivo generado! "
    f"Abre el archivo '{ARCHIVO_MAPA_HTML}' en cualquier navegador web para verlo."
)
