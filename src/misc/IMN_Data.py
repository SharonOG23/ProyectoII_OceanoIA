from datetime import date
import os
import pandas as pd
import meteostat as ms

# 1. Definir los puntos geográficos de manera estructurada (Nombre, Latitud, Longitud, Altitud)
# De esta forma tienes las coordenadas disponibles en todo momento.
REGIONES = {
    "Caribe": {"lat": 9.9913, "lon": -83.0415, "alt": 3},  # Limón
    "Pacifico_Norte": {"lat": 10.6333, "lon": -85.4333, "alt": 25},  # Liberia / Guanacaste
    "Pacifico_Central": {"lat": 9.6158, "lon": -84.6299, "alt": 5},  # Jacó / Puntarenas
    "Pacifico_Sur": {"lat": 8.6412, "lon": -83.6019, "alt": 10}  # Golfito / Osa
}

# Rango de tiempo
START = date(2022, 1, 1)
END = date(2026, 7, 9)

# Lista para almacenar los DataFrames de cada región
lista_dfs = []

print("Extrayendo e interpolando datos de Meteostat...")

for nombre_region, info in REGIONES.items():
    # Crear el objeto Point dinámicamente para Meteostat
    punto = ms.Point(info["lat"], info["lon"], info["alt"])

    # Obtener las 4 estaciones más cercanas al punto
    estaciones = ms.stations.nearby(punto, limit=4)

    # Consultar datos diarios
    datos_diarios = ms.daily(estaciones, START, END)

    # Interpolar para obtener los datos específicos del punto geográfico
    df_region = ms.interpolate(datos_diarios, punto).fetch()

    # Si se obtuvieron datos, añadir columnas identificadoras y guardar
    if not df_region.empty:
        df_region = df_region.reset_index()

        # Asignar nombre, latitud y longitud desde nuestro diccionario estructurado
        df_region["Region"] = nombre_region
        df_region["Latitud"] = info["lat"]
        df_region["Longitud"] = info["lon"]

        # Seleccionar y reordenar columnas clave
        df_region = df_region[["time", "Region", "Latitud", "Longitud", "temp", "tmin", "tmax", "prcp", "wspd"]]
        lista_dfs.append(df_region)
        print(f"✅ Datos obtenidos con éxito para: {nombre_region}")
    else:
        print(f"❌ No se pudieron obtener datos para: {nombre_region}")

# 2. Consolidar todos los datos en un único DataFrame
if lista_dfs:
    df_final = pd.concat(lista_dfs, ignore_index=True)

    # Renombrar columnas a español
    df_final.columns = [
        "Fecha", "Region", "Latitud", "Longitud",
        "Temp_Promedio", "Temp_Minima", "Temp_Maxima", "Precipitacion", "Vel_Viento"
    ]

    # --- CONFIGURACIÓN DE RUTAS ---
    # Obtiene la ruta de la carpeta donde está este script (proyecto\src\misc)
    ruta_actual = os.path.dirname(os.path.abspath(__file__))

    # Sube 3 niveles (llega a 'ProyectoII_OceanoIA') y luego entra a 'datos/raw/RNN'
    carpeta_destino = os.path.abspath(os.path.join(ruta_actual, "..", "..", "data", "raw", "RNN"))

    # Crea las carpetas si aún no existen
    os.makedirs(carpeta_destino, exist_ok=True)

    # Define el nombre del archivo y la ruta absoluta final
    nombre_archivo = "datos_IMN.csv"
    ruta_final_archivo = os.path.join(carpeta_destino, nombre_archivo)
    # ------------------------------

    # 3. Guardar el archivo como .CSV en la ruta especificada
    df_final.to_csv(ruta_final_archivo, index=False, encoding="utf-8-sig")
    print(f"\n¡Proceso finalizado! Archivo guardado con éxito en:\n👉 {ruta_final_archivo}")
else:
    print("\nError: No se logró recopilar información de ninguna región.")