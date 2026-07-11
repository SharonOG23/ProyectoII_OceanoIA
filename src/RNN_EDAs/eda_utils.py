"""
eda_utils.py
Funciones compartidas para los scripts de EDA del proyecto de predicción
de condiciones marino-costeras (RNN).

Este módulo NO se ejecuta solo; lo importan eda_openmeteo.py, eda_copernicus.py,
eda_imn.py y eda_moon_phases.py
"""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # backend sin GUI, seguro para correr desde PyCharm o consola
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def save_fig(fig, name, output_dir):
    ensure_dir(output_dir)
    path = os.path.join(output_dir, name)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  [figura guardada] {path}")


def basic_info(df, nombre_dataset):
    print(f"\n{'='*70}")
    print(f"INFO BÁSICA: {nombre_dataset}")
    print(f"{'='*70}")
    print(f"Filas: {df.shape[0]:,} | Columnas: {df.shape[1]}")
    print("\nTipos de dato:")
    print(df.dtypes)
    print("\nPrimeras filas:")
    print(df.head(5).to_string())


def missing_report(df):
    """Devuelve un DataFrame con conteo y % de nulos por columna, ordenado desc."""
    total = df.isna().sum()
    pct = (df.isna().mean() * 100).round(2)
    rep = pd.DataFrame({"nulos": total, "pct_nulos": pct})
    rep = rep.sort_values("pct_nulos", ascending=False)
    return rep


def print_missing_report(df, nombre_dataset):
    rep = missing_report(df)
    print(f"\n--- Reporte de valores faltantes: {nombre_dataset} ---")
    print(rep.to_string())
    return rep


def plot_missing_heatmap(df, output_dir, name="missing_heatmap.png", sample=5000):
    """Heatmap de nulos. Si el df es muy grande, se toma una muestra para que sea legible."""
    data = df if len(df) <= sample else df.sample(sample, random_state=42).sort_index()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(data.isna(), cbar=False, yticklabels=False, cmap="viridis", ax=ax)
    ax.set_title("Mapa de valores faltantes (blanco = nulo)")
    save_fig(fig, name, output_dir)


def numeric_summary(df, numeric_cols):
    print("\n--- Estadística descriptiva (variables numéricas) ---")
    print(df[numeric_cols].describe().to_string())


def plot_distributions(df, numeric_cols, output_dir, prefix=""):
    """Histograma + boxplot para cada variable numérica, para ver forma y outliers."""
    for col in numeric_cols:
        serie = df[col].dropna()
        if serie.empty:
            print(f"  [aviso] {col} no tiene datos no-nulos, se omite el gráfico.")
            continue
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        sns.histplot(serie, kde=True, ax=axes[0], color="steelblue")
        axes[0].set_title(f"Distribución de {col}")
        sns.boxplot(x=serie, ax=axes[1], color="salmon")
        axes[1].set_title(f"Boxplot de {col} (detección de outliers)")
        fig.tight_layout()
        save_fig(fig, f"{prefix}dist_{col}.png", output_dir)


def plot_correlation(df, numeric_cols, output_dir, name="correlacion.png"):
    cols = [c for c in numeric_cols if df[c].notna().sum() > 1]
    if len(cols) < 2:
        print("  [aviso] No hay suficientes columnas numéricas con datos para correlación.")
        return
    corr = df[cols].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Matriz de correlación")
    save_fig(fig, name, output_dir)


def plot_time_series_by_group(df, time_col, value_col, group_col, output_dir,
                               name=None, agg="mean", freq=None):
    """
    Grafica la evolución temporal de value_col, una línea por cada valor de group_col.
    Si freq se indica (ej. 'D', 'M'), se remuestrea antes de graficar (útil para series horarias largas).
    """
    tmp = df[[time_col, value_col, group_col]].dropna(subset=[value_col]).copy()
    tmp[time_col] = pd.to_datetime(tmp[time_col], errors="coerce", utc=True)
    tmp = tmp.dropna(subset=[time_col])

    fig, ax = plt.subplots(figsize=(12, 5))
    for grupo, sub in tmp.groupby(group_col):
        sub = sub.set_index(time_col).sort_index()
        if freq:
            sub = sub[[value_col]].resample(freq).agg(agg)
        ax.plot(sub.index, sub[value_col], label=str(grupo), linewidth=1)
    ax.set_title(f"Evolución temporal de {value_col} por {group_col}")
    ax.set_xlabel("Fecha")
    ax.set_ylabel(value_col)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    save_fig(fig, name or f"serie_tiempo_{value_col}.png", output_dir)


def plot_missing_by_time(df, time_col, cols, output_dir, freq="D", name="missing_por_fecha.png"):
    """Muestra si el % de nulos varía a lo largo del tiempo (patrón no aleatorio)."""
    tmp = df[[time_col] + cols].copy()
    tmp[time_col] = pd.to_datetime(tmp[time_col], errors="coerce", utc=True)
    tmp = tmp.dropna(subset=[time_col]).set_index(time_col)
    pct_nulos = tmp.isna().resample(freq).mean() * 100

    fig, ax = plt.subplots(figsize=(12, 5))
    for col in cols:
        ax.plot(pct_nulos.index, pct_nulos[col], label=col)
    ax.set_title(f"% de nulos a través del tiempo (agregado por '{freq}')")
    ax.set_ylabel("% nulos")
    ax.legend(fontsize=8)
    fig.tight_layout()
    save_fig(fig, name, output_dir)


def write_text_report(lines, output_dir, filename="resumen_eda.txt"):
    ensure_dir(output_dir)
    path = os.path.join(output_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(str(l) for l in lines))
    print(f"\n[reporte guardado] {path}")
