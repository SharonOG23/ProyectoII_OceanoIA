import argparse
import os
import sys
from pathlib import Path


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

BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Carpeta por defecto (relativa a la raíz del proyecto, BASE_DIR) donde se
# guardarán el .nc, el .csv y el mapa .html. Se puede sobreescribir al
# ejecutar el script, por ejemplo:
#   python Copernicus_API.py --carpeta data/raw/viento
#   python Copernicus_API.py --carpeta "C:\ruta\absoluta\a\otra\carpeta"
CARPETA_DATOS_POR_DEFECTO = "data/raw/RNN"

parser = argparse.ArgumentParser(
    description="Descarga datos de Copernicus Marine y los guarda en la carpeta indicada."
)
parser.add_argument(
    "--carpeta", "-c",
    type=str,
    default=CARPETA_DATOS_POR_DEFECTO,
    help=(
        "Carpeta donde se guardarán los archivos generados (.nc, .csv, .html). "
        "Puede ser una ruta relativa a la raíz del proyecto "
        f"(por defecto: '{CARPETA_DATOS_POR_DEFECTO}') o una ruta absoluta."
    ),
)
argumentos = parser.parse_args()

ruta_carpeta = Path(argumentos.carpeta)
# Si la ruta no es absoluta, se interpreta relativa a la raíz del proyecto.
CARPETA_DATOS = ruta_carpeta if ruta_carpeta.is_absolute() else BASE_DIR / ruta_carpeta
CARPETA_DATOS.mkdir(parents=True, exist_ok=True)

print(f"Los archivos se guardarán en: {CARPETA_DATOS}")

ARCHIVO_DESCARGADO = CARPETA_DATOS / "datos_Copernicus.nc"
ARCHIVO_DATOS_CSV = CARPETA_DATOS / "datos_Copernicus.csv"
ARCHIVO_MAPA_HTML = CARPETA_DATOS / "mapa_costa_rica.html"

# ---------------------------------------------------------------------------
# CONFIGURACIÓN DE PARÁMETROS
# ---------------------------------------------------------------------------

COORDENADAS_CR = {
    "minimum_longitude": -87.12067190856095,
    "maximum_longitude": -81.90192083351737,
    "minimum_latitude": 7.523523292816108,
    "maximum_latitude": 11.251202632132944,
}

FECHA_INICIO = "2022-05-09T00:00:00"

# Recomendación:
# No uses una fecha futura si el dataset todavía no contiene datos hasta esa fecha.
# Ajusta según la disponibilidad real del producto en Copernicus.
FECHA_FIN = "2026-07-09T00:00:00"

# ---------------------------------------------------------------------------
# PASO 1: DESCARGA DE DATOS DESDE COPERNICUS MARINE
# ---------------------------------------------------------------------------

print("Iniciando descarga de datos desde Copernicus...")

# IMPORTANTE: nunca escribas tu usuario/contraseña directamente en el código.
# Defínelos como variables de entorno antes de ejecutar el script, por ejemplo en
# la terminal:
#   export COPERNICUSMARINE_SERVICE_USERNAME="tu_usuario"
#   export COPERNICUSMARINE_SERVICE_PASSWORD="tu_password"
# copernicusmarine detecta estas variables automáticamente y no es necesario
# llamar a login() manualmente cada vez. Si prefieres hacerlo interactivo,
# usa copernicusmarine.login() sin argumentos y te pedirá los datos una sola vez,
# guardándolos de forma segura en tu configuración local.

usuario = "fbrenes" #os.environ.get("COPERNICUSMARINE_SERVICE_USERNAME")
password = "4HQN6J58-i8ig" #os.environ.get("COPERNICUSMARINE_SERVICE_PASSWORD")

if usuario and password:
    copernicusmarine.login(username=usuario, password=password, force_overwrite=True)
else:
    # Si ya iniciaste sesión antes, copernicusmarine reutiliza las credenciales
    # guardadas y no volverá a pedirlas.
    print("No se encontraron credenciales en variables de entorno; "
          "se usará la sesión guardada localmente (si existe).")

print("-> Descargando datos de viento (dispersómetro)...")

try:
    resultado_descarga = copernicusmarine.subset(
        dataset_id="cmems_obs-wind_glo_phy_nrt_l3-fy3e-windrad-asc-0.25deg_P1D-i",
        variables=["measurement_time", "wind_speed", "wind_to_dir", "model_wind_to_dir"],
        minimum_longitude=COORDENADAS_CR["minimum_longitude"],
        maximum_longitude=COORDENADAS_CR["maximum_longitude"],
        minimum_latitude=COORDENADAS_CR["minimum_latitude"],
        maximum_latitude=COORDENADAS_CR["maximum_latitude"],
        start_datetime=FECHA_INICIO,
        end_datetime=FECHA_FIN,
        output_directory=str(CARPETA_DATOS),
        output_filename=ARCHIVO_DESCARGADO.name,
        disable_progress_bar=True,
        overwrite=True,
    )
except Exception:
    import traceback
    traceback.print_exc()
    sys.exit(1)

def _resolver_archivo_nc_descargado(ruta_esperada: Path, resultado) -> Path:
    """
    Distintas versiones de copernicusmarine se comportan distinto con
    output_filename/output_directory: a veces crean el .nc directamente,
    a veces crean una carpeta con ese nombre y el .nc real queda adentro.
    Esta función normaliza ambos casos.
    """
    # Caso normal: ya es el archivo .nc esperado.
    if ruta_esperada.is_file():
        return ruta_esperada

    # Caso "carpeta en vez de archivo": buscar el .nc dentro.
    if ruta_esperada.is_dir():
        candidatos = sorted(ruta_esperada.rglob("*.nc"))
        if candidatos:
            return candidatos[0]

    # Respaldo: usar la ruta que el propio objeto de resultado reporta.
    ruta_reportada = getattr(resultado, "file_path", None) or getattr(
        resultado, "output_file", None
    )
    if ruta_reportada:
        ruta_reportada = Path(ruta_reportada)
        if ruta_reportada.is_file():
            return ruta_reportada
        if ruta_reportada.is_dir():
            candidatos = sorted(ruta_reportada.rglob("*.nc"))
            if candidatos:
                return candidatos[0]

    print(f"Error: no se encontró ningún archivo .nc en '{ruta_esperada}'.")
    print(f"Resultado devuelto por copernicusmarine.subset(): {resultado}")
    sys.exit(1)


ARCHIVO_DESCARGADO = _resolver_archivo_nc_descargado(ARCHIVO_DESCARGADO, resultado_descarga)
print(f"Archivo NetCDF localizado en: {ARCHIVO_DESCARGADO}")

print("¡Descarga del archivo NetCDF finalizada con éxito!")

# ---------------------------------------------------------------------------
# NOTA IMPORTANTE SOBRE CREDENCIALES
# ---------------------------------------------------------------------------
# Este script todavía tiene el usuario/contraseña escritos directamente en el
# código (líneas más arriba). Esto es un riesgo de seguridad, sobre todo si
# vas a compartir o subir este archivo (por ejemplo a un repositorio Git).
# Se recomienda moverlos a variables de entorno como indica el comentario
# original del script.

# ---------------------------------------------------------------------------
# PASO 2: PROCESAMIENTO Y CONVERSIÓN A FORMATO CSV
# ---------------------------------------------------------------------------

print("\nProcesando archivo NetCDF y convirtiendo a CSV...")

with xr.open_dataset(ARCHIVO_DESCARGADO) as ds_fisica:
    df_fisica = ds_fisica.to_dataframe().reset_index()

if "depth" in df_fisica.columns:
    profundidad_minima = df_fisica["depth"].min()
    df_fisica = df_fisica[df_fisica["depth"] == profundidad_minima]

# Los nombres de columnas tras to_dataframe() son los nombres de variable
# originales (sin unidades entre corchetes), así que renombramos así:
df_fisica = df_fisica.rename(
    columns={
        "measurement_time": "Tiempo_Medicion",
        "wind_speed": "Vel_Viento",
        "wind_to_dir": "Direccion_Viento",
        "model_wind_to_dir": "Direccion_Viento_Modelo",
        "latitude": "Latitud",
        "longitude": "Longitud",
    }
)

df_final = df_fisica

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

columnas_necesarias = ["Latitud", "Longitud", "Vel_Viento"]
df_mapa = df_final.dropna(subset=[c for c in columnas_necesarias if c in df_final.columns])

if df_mapa.empty:
    print("No hay datos válidos de viento para generar marcadores en el mapa.")
else:
    df_mapa = df_mapa.sample(n=min(500, len(df_mapa)), random_state=42)

    for _, fila in df_mapa.iterrows():
        dir_viento = fila["Direccion_Viento"] if "Direccion_Viento" in fila and pd.notna(fila["Direccion_Viento"]) else None
        texto_popup = f"""
        <b>Coordenadas:</b> {fila['Latitud']:.2f}, {fila['Longitud']:.2f}<br>
        <b>Vel. viento:</b> {fila['Vel_Viento']:.2f} m/s<br>
        <b>Dirección viento:</b> {f"{dir_viento:.2f}°" if dir_viento is not None else "N/D"}
        """

        folium.CircleMarker(
            location=[fila["Latitud"], fila["Longitud"]],
            radius=4,
            popup=folium.Popup(texto_popup, max_width=300),
            color="blue" if fila["Vel_Viento"] < 8 else "red",
            fill=True,
            fill_opacity=0.7,
        ).add_to(mapa_cr)

mapa_cr.save(ARCHIVO_MAPA_HTML)

print(
    f"¡Mapa interactivo generado! "
    f"Abre el archivo '{ARCHIVO_MAPA_HTML}' en cualquier navegador web para verlo."
)