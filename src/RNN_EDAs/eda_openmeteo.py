"""
eda_openmeteo.py
EDA del dataset: CostaRica_Marine_OpenMateo_2022_Hoy.csv
Variables: oleaje (altura/dirección), SST, corriente oceánica.
Resolución horaria, 20 puntos de coordenadas fijas (grilla oceánica).

Cómo correrlo en PyCharm:
1. Ajusta DATA_PATH y OUTPUT_DIR si tu estructura de carpetas es distinta.
2. Corre el script directamente (botón Run). No necesita argumentos.
"""

import pandas as pd
from eda_utils import (
    basic_info, print_missing_report, plot_missing_heatmap, numeric_summary,
    plot_distributions, plot_correlation, plot_missing_by_time,
    plot_time_series_by_group, write_text_report, ensure_dir
)

DATA_PATH = "../../data/raw/RNN/CostaRica_Marine_OpenMateo_2022_Hoy.csv"
OUTPUT_DIR = "../../data/processed/resultados_EDAs_RNN/eda_openmeteo"
NOMBRE = "OpenMeteo Marine"

NUMERIC_COLS = [
    "wave_height", "wave_direction", "sea_surface_temperature",
    "ocean_current_velocity", "ocean_current_direction",
]


def main():
    ensure_dir(OUTPUT_DIR)
    reporte = []

    df = pd.read_csv(DATA_PATH)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")

    basic_info(df, NOMBRE)
    reporte.append(f"Dataset: {NOMBRE}")
    reporte.append(f"Filas: {df.shape[0]:,} | Columnas: {df.shape[1]}")
    reporte.append(f"Rango de fechas: {df['datetime'].min()} -> {df['datetime'].max()}")

    # --- Cobertura espacial ---
    n_coords = df[["latitude", "longitude"]].drop_duplicates().shape[0]
    print(f"\nPuntos de coordenadas únicos (zonas de la grilla oceánica): {n_coords}")
    reporte.append(f"Puntos de coordenadas únicos: {n_coords}")
    print(df[["latitude", "longitude"]].drop_duplicates().to_string(index=False))

    # --- Faltantes ---
    rep_na = print_missing_report(df, NOMBRE)
    reporte.append("\nValores faltantes por columna (%):")
    reporte.append(rep_na.to_string())
    plot_missing_heatmap(df, OUTPUT_DIR)
    plot_missing_by_time(df, "datetime", NUMERIC_COLS, OUTPUT_DIR, freq="W")

    # --- Estadística descriptiva ---
    numeric_summary(df, NUMERIC_COLS)
    reporte.append("\nEstadística descriptiva:")
    reporte.append(df[NUMERIC_COLS].describe().to_string())

    # --- Distribuciones y outliers ---
    plot_distributions(df, NUMERIC_COLS, OUTPUT_DIR)

    # --- Correlación entre variables ---
    plot_correlation(df, NUMERIC_COLS, OUTPUT_DIR)

    # --- Series de tiempo por zona (coordenada) ---
    # Se crea un identificador de zona combinando lat/lon
    df["zona"] = df["latitude"].astype(str) + "_" + df["longitude"].astype(str)
    for var in ["wave_height", "sea_surface_temperature"]:
        plot_time_series_by_group(
            df, "datetime", var, "zona", OUTPUT_DIR,
            name=f"serie_tiempo_{var}_por_zona.png", freq="W"
        )

    # --- Chequeo de consistencia de la grilla: ¿todas las zonas tienen el mismo rango de fechas? ---
    rango_por_zona = df.groupby("zona")["datetime"].agg(["min", "max", "count"])
    print("\n--- Rango de fechas y conteo de registros por zona ---")
    print(rango_por_zona.to_string())
    reporte.append("\nRango de fechas y conteo por zona:")
    reporte.append(rango_por_zona.to_string())

    write_text_report(reporte, OUTPUT_DIR)
    print(f"\nListo. Revisa las figuras y el resumen en: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
