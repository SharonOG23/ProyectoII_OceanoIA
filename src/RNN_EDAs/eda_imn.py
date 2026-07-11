"""
eda_imn.py
EDA del dataset: datos_IMN.csv
Variables: temperatura (promedio/min/max), precipitación, viento.
Resolución diaria, 4 regiones (Caribe, Pacífico Norte, Pacífico Central, Pacífico Sur).
Este dataset probablemente defina la unidad espacial ("zona") del modelo,
ya que es el único con una regionalización explícita.

Cómo correrlo en PyCharm:
1. Ajusta DATA_PATH y OUTPUT_DIR si tu estructura de carpetas es distinta.
2. Corre el script directamente (botón Run).
"""

import pandas as pd
from eda_utils import (
    basic_info, print_missing_report, plot_missing_heatmap, numeric_summary,
    plot_distributions, plot_correlation, plot_missing_by_time,
    plot_time_series_by_group, write_text_report, ensure_dir
)

DATA_PATH = "../../data/raw/RNN/datos_IMN.csv"
OUTPUT_DIR = "../../data/processed/resultados_EDAs_RNN/eda_IMN"
NOMBRE = "IMN"

NUMERIC_COLS = ["Temp_Promedio", "Temp_Minima", "Temp_Maxima", "Precipitacion", "Vel_Viento"]


def main():
    ensure_dir(OUTPUT_DIR)
    reporte = []

    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce")

    basic_info(df, NOMBRE)
    reporte.append(f"Dataset: {NOMBRE}")
    reporte.append(f"Filas: {df.shape[0]:,} | Columnas: {df.shape[1]}")
    reporte.append(f"Rango de fechas: {df['Fecha'].min()} -> {df['Fecha'].max()}")

    # --- Regiones (posibles "zonas" del modelo) ---
    print("\nRegiones:", df["Region"].unique().tolist())
    reporte.append(f"Regiones: {df['Region'].unique().tolist()}")
    registros_por_region = df.groupby("Region")["Fecha"].agg(["min", "max", "count"])
    print("\n--- Rango de fechas y conteo por región ---")
    print(registros_por_region.to_string())
    reporte.append("\nRango de fechas y conteo por región:")
    reporte.append(registros_por_region.to_string())

    # --- Faltantes ---
    rep_na = print_missing_report(df, NOMBRE)
    reporte.append("\nValores faltantes por columna (%):")
    reporte.append(rep_na.to_string())
    plot_missing_heatmap(df, OUTPUT_DIR)
    plot_missing_by_time(df, "Fecha", NUMERIC_COLS, OUTPUT_DIR, freq="ME")

    # --- ¿Precipitación falta más en alguna región o época del año? ---
    pct_nulo_precip_region = df.groupby("Region")["Precipitacion"].apply(lambda s: s.isna().mean() * 100)
    print("\n--- % de nulos en Precipitacion por región ---")
    print(pct_nulo_precip_region.to_string())
    reporte.append("\n% nulos en Precipitacion por región:")
    reporte.append(pct_nulo_precip_region.to_string())

    # --- Estadística descriptiva ---
    numeric_summary(df, NUMERIC_COLS)
    reporte.append("\nEstadística descriptiva:")
    reporte.append(df[NUMERIC_COLS].describe().to_string())

    # --- Distribuciones ---
    plot_distributions(df, NUMERIC_COLS, OUTPUT_DIR)

    # --- Correlación ---
    plot_correlation(df, NUMERIC_COLS, OUTPUT_DIR)

    # --- Series de tiempo por región ---
    for var in ["Temp_Promedio", "Vel_Viento", "Precipitacion"]:
        plot_time_series_by_group(
            df, "Fecha", var, "Region", OUTPUT_DIR,
            name=f"serie_tiempo_{var}_por_region.png", freq="ME"
        )

    write_text_report(reporte, OUTPUT_DIR)
    print(f"\nListo. Revisa las figuras y el resumen en: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
