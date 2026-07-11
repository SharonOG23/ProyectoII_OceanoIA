"""
eda_moon_phases.py
EDA del dataset: moon_phases_UTC_2020-2050.csv
Variables: Area (fracción iluminada), Category (fase discreta 0-7), Phase (fracción 0-1 del ciclo).
Resolución diaria, sin coordenadas (aplica igual para todas las zonas del país).

Cómo correrlo en PyCharm:
1. Ajusta DATA_PATH y OUTPUT_DIR si tu estructura de carpetas es distinta.
2. Corre el script directamente (botón Run).
"""

import os
import pandas as pd
from eda_utils import (
    basic_info, print_missing_report, numeric_summary,
    plot_distributions, save_fig, write_text_report, ensure_dir
)
import matplotlib.pyplot as plt
import seaborn as sns

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "../../data/raw/RNN/moon_phases_UTC_2020-2050.csv")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "../../data/processed/resultados_EDAs_RNN/eda_moon_phases")
NOMBRE = "Fases lunares"

# El CSV real solo trae la fase discreta (Category, 0-7), no Area ni Phase
# continuas. Se deja Category como única "numérica" para las estadísticas
# descriptivas, aunque en realidad es una variable categórica ordinal.
NUMERIC_COLS = ["Category"]


def main():
    ensure_dir(OUTPUT_DIR)
    reporte = []

    df = pd.read_csv(DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    basic_info(df, NOMBRE)
    reporte.append(f"Dataset: {NOMBRE}")
    reporte.append(f"Filas: {df.shape[0]:,} | Columnas: {df.shape[1]}")
    reporte.append(f"Rango de fechas: {df['Date'].min()} -> {df['Date'].max()}")

    # --- Continuidad: ¿hay un registro por cada día del rango, sin huecos? ---
    dias_esperados = (df["Date"].max() - df["Date"].min()).days + 1
    print(f"\nDías esperados en el rango: {dias_esperados} | Filas reales: {df.shape[0]}")
    reporte.append(f"Días esperados: {dias_esperados} | Filas reales: {df.shape[0]}")
    if dias_esperados == df.shape[0]:
        print("✔ No hay huecos: un registro por cada día del calendario.")
        reporte.append("Sin huecos: cobertura diaria completa.")
    else:
        print("⚠ Hay huecos en la serie diaria, revisar.")
        reporte.append("⚠ Hay huecos en la serie diaria, revisar.")

    # --- Faltantes ---
    rep_na = print_missing_report(df, NOMBRE)
    reporte.append("\nValores faltantes por columna (%):")
    reporte.append(rep_na.to_string())

    # --- Distribución de la fase discreta (Category) ---
    print("\n--- Conteo por categoría de fase lunar ---")
    conteo_cat = df["Category"].value_counts().sort_index()
    print(conteo_cat.to_string())
    reporte.append("\nConteo por categoría de fase lunar:")
    reporte.append(conteo_cat.to_string())

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.countplot(x="Category", data=df, ax=ax, color="slateblue")
    ax.set_title("Frecuencia de cada categoría de fase lunar (0-7)")
    save_fig(fig, "categorias_fase_lunar.png", OUTPUT_DIR)

    # --- Estadística descriptiva ---
    numeric_summary(df, NUMERIC_COLS)
    reporte.append("\nEstadística descriptiva:")
    reporte.append(df[NUMERIC_COLS].describe().to_string())

    # --- Distribuciones ---
    plot_distributions(df, NUMERIC_COLS, OUTPUT_DIR)

    # Nota: no hay correlación que calcular con una sola columna numérica
    # (Area/Phase no existen en este CSV), así que se omite plot_correlation.

    # --- Ciclo completo: graficar un año como ejemplo, para validar visualmente el patrón ---
    ejemplo = df[(df["Date"] >= "2023-01-01") & (df["Date"] <= "2023-12-31")]
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(ejemplo["Date"], ejemplo["Category"], color="darkorange", marker="o", markersize=2)
    ax.set_title("Fase lunar (Category) durante 2023 - validación visual del ciclo")
    ax.set_ylabel("Category (0-7)")
    fig.tight_layout()
    save_fig(fig, "ciclo_lunar_2023.png", OUTPUT_DIR)

    write_text_report(reporte, OUTPUT_DIR)
    print(f"\nListo. Revisa las figuras y el resumen en: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()