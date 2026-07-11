"""
eda_copernicus.py
EDA del dataset: datos_Copernicus.csv
Variables: viento (velocidad y dirección).
Resolución diaria, 300 puntos de coordenadas (grilla más densa), pero
cobertura temporal muy corta (~2 meses) y con muchos nulos: hay que
diagnosticar esto a fondo antes de decidir si se usa para el modelo.

Cómo correrlo en PyCharm:
1. Ajusta DATA_PATH y OUTPUT_DIR si tu estructura de carpetas es distinta.
2. Corre el script directamente (botón Run).
"""

import pandas as pd
from eda_utils import (
    basic_info, print_missing_report, plot_missing_heatmap, numeric_summary,
    plot_distributions, plot_correlation, plot_missing_by_time,
    write_text_report, ensure_dir
)

DATA_PATH = "../../data/raw/RNN/datos_Copernicus.csv"
OUTPUT_DIR = "../../data/processed/resultados_EDAs_RNN/eda_copernicus"
NOMBRE = "Copernicus (viento)"

NUMERIC_COLS = ["Vel_Viento", "Direccion_Viento", "Direccion_Viento_Modelo"]


def main():
    ensure_dir(OUTPUT_DIR)
    reporte = []

    df = pd.read_csv(DATA_PATH)
    df["time"] = pd.to_datetime(df["time"], errors="coerce")

    basic_info(df, NOMBRE)
    reporte.append(f"Dataset: {NOMBRE}")
    reporte.append(f"Filas: {df.shape[0]:,} | Columnas: {df.shape[1]}")
    reporte.append(f"Rango de fechas: {df['time'].min()} -> {df['time'].max()}")
    dias_cobertura = (df["time"].max() - df["time"].min()).days
    reporte.append(f"Días de cobertura temporal: {dias_cobertura}")
    print(f"\n⚠ Días de cobertura temporal: {dias_cobertura} (dataset muy corto vs. OpenMeteo/IMN)")

    # --- Cobertura espacial ---
    n_coords = df[["Latitud", "Longitud"]].drop_duplicates().shape[0]
    print(f"\nPuntos de coordenadas únicos: {n_coords}")
    reporte.append(f"Puntos de coordenadas únicos: {n_coords}")

    # --- Faltantes ---
    rep_na = print_missing_report(df, NOMBRE)
    reporte.append("\nValores faltantes por columna (%):")
    reporte.append(rep_na.to_string())
    plot_missing_heatmap(df, OUTPUT_DIR)
    plot_missing_by_time(df, "time", NUMERIC_COLS, OUTPUT_DIR, freq="D")

    # --- ¿El faltante depende de la zona (coordenada) o es parejo? ---
    df["zona"] = df["Latitud"].astype(str) + "_" + df["Longitud"].astype(str)
    pct_nulo_por_zona = df.groupby("zona")[NUMERIC_COLS].apply(lambda g: g.isna().mean() * 100)
    print("\n--- % de nulos de Vel_Viento por zona (primeras 15) ---")
    print(pct_nulo_por_zona.sort_values("Vel_Viento", ascending=False).head(15).to_string())
    reporte.append("\n% nulos de Vel_Viento por zona (top 15 con más nulos):")
    reporte.append(pct_nulo_por_zona.sort_values("Vel_Viento", ascending=False).head(15).to_string())

    # --- Estadística descriptiva (solo sobre lo no-nulo) ---
    numeric_summary(df, NUMERIC_COLS)
    reporte.append("\nEstadística descriptiva:")
    reporte.append(df[NUMERIC_COLS].describe().to_string())

    # --- Distribuciones ---
    plot_distributions(df, NUMERIC_COLS, OUTPUT_DIR)

    # --- Correlación (incluye ver si Direccion_Viento y Direccion_Viento_Modelo son redundantes) ---
    plot_correlation(df, NUMERIC_COLS, OUTPUT_DIR)

    write_text_report(reporte, OUTPUT_DIR)
    print(f"\nListo. Revisa las figuras y el resumen en: {OUTPUT_DIR}")
    print("\n⚠ RECORDATORIO: con ~2 meses de historia y ~78% de nulos en viento, evalúa "
          "si este dataset se usa solo como referencia cruzada frente al viento del IMN, "
          "en vez de como fuente principal para entrenar el RNN.")


if __name__ == "__main__":
    main()
