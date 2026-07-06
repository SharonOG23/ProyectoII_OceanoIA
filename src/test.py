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

BASE_DIR = Path(__file__).resolve().parent.parent
CARPETA_DATOS = BASE_DIR / "data" / "raw" / "RNN"
CARPETA_DATOS.mkdir(parents=True, exist_ok=True)

ARCHIVO_DESCARGADO = CARPETA_DATOS / "costa_rica.nc"
ARCHIVO_DATOS_CSV = CARPETA_DATOS / "costa_rica.csv"
ARCHIVO_MAPA_HTML = CARPETA_DATOS / "mapa_costa_rica.html"

# ---------------------------------------------------------------------------
# CONFIGURACIÓN DE PARÁMETROS
# ---------------------------------------------------------------------------

COORDENADAS_CR = {
    "minimum_longitude": -86.50039881274324,
    "maximum_longitude": -82.11922642494376,
    "minimum_latitude": 7.5707039686301245,
    "maximum_latitude": 11.269335816014534,
}

FECHA_INICIO = "2022-12-01T01:07:06"

# Recomendación:
# No uses una fecha futura si el dataset todavía no contiene datos hasta esa fecha.
# Ajusta según la disponibilidad real del producto en Copernicus.
FECHA_FIN = "2026-05-30T22:34:58"

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

usuario = os.environ.get("COPERNICUSMARINE_SERVICE_USERNAME")
password = os.environ.get("COPERNICUSMARINE_SERVICE_PASSWORD")

if usuario and password:
    copernicusmarine.login(username=usuario, password=password, force_overwrite=True)
else:
    # Si ya iniciaste sesión antes, copernicusmarine reutiliza las credenciales
    # guardadas y no volverá a pedirlas.
    print("No se encontraron credenciales en variables de entorno; "
          "se usará la sesión guardada localmente (si existe).")

print("-> Descargando datos de oleaje y viento...")

try:
    resultado_descarga = copernicusmarine.subset(
        dataset_id="cmems_obs-wave_glo_phy-swh_nrt_h2c-l3_PT1S",
        dataset_version="202211",
        variables=["VAVH", "VAVH_UNFILTERED", "WIND_SPEED"],
        minimum_longitude=COORDENADAS_CR["minimum_longitude"],
        maximum_longitude=COORDENADAS_CR["maximum_longitude"],
        minimum_latitude=COORDENADAS_CR["minimum_latitude"],
        maximum_latitude=COORDENADAS_CR["maximum_latitude"],
        start_datetime=FECHA_INICIO,
        end_datetime=FECHA_FIN,
        minimum_depth=0,
        maximum_depth=0,
        coordinates_selection_method="strict-inside",
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
        "VAVH": "Altura_m",
        "VAVH_UNFILTERED": "Altura_m_sin_filtrar",
        "WIND_SPEED": "Vel_Viento",
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

columnas_necesarias = ["Latitud", "Longitud", "Altura_m"]
df_mapa = df_final.dropna(subset=[c for c in columnas_necesarias if c in df_final.columns])

if df_mapa.empty:
    print("No hay datos válidos de oleaje para generar marcadores en el mapa.")
else:
    df_mapa = df_mapa.sample(n=min(500, len(df_mapa)), random_state=42)

    for _, fila in df_mapa.iterrows():
        vel_viento = fila["Vel_Viento"] if "Vel_Viento" in fila and pd.notna(fila["Vel_Viento"]) else None
        texto_popup = f"""
        <b>Coordenadas:</b> {fila['Latitud']:.2f}, {fila['Longitud']:.2f}<br>
        <b>Altura de ola:</b> {fila['Altura_m']:.2f} m<br>
        <b>Vel. viento:</b> {f"{vel_viento:.2f} m/s" if vel_viento is not None else "N/D"}
        """

        folium.CircleMarker(
            location=[fila["Latitud"], fila["Longitud"]],
            radius=4,
            popup=folium.Popup(texto_popup, max_width=300),
            color="blue" if fila["Altura_m"] < 2 else "red",
            fill=True,
            fill_opacity=0.7,
        ).add_to(mapa_cr)

mapa_cr.save(ARCHIVO_MAPA_HTML)

print(
    f"¡Mapa interactivo generado! "
    f"Abre el archivo '{ARCHIVO_MAPA_HTML}' en cualquier navegador web para verlo."
)