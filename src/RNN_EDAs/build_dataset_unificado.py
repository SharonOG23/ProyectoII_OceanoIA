"""
build_dataset_unificado.py

Construye el dataset horario unificado por región para entrenar el RNN.

Decisiones tomadas (ver discusión EDA):
- Se descarta Copernicus: solo 61 días de historia y ~78% de nulos en viento.
- Se descartan Marea y Presion: ninguna fuente las provee.
- Se excluye la región Pacifico_Norte: la grilla marina de OpenMeteo no tiene
  puntos geograficamente cercanos a esa estacion (banda de latitud 10.0 cubre
  solo el lado Caribe; banda 8.0 cubre Pacifico Central/Sur). Incluirla forzaria
  un proxy geograficamente incorrecto.
- El viento del dataset final proviene de IMN (diario), no de Copernicus.
- Precipitacion en Pacifico_Central (77.5% nulos) se imputa con interpolacion
  temporal corta (huecos <=3 dias) + climatologia mensual para el resto.
- Granularidad final: horaria. Las variables diarias (IMN, fases lunares) se
  propagan hacia adelante (forward-fill) dentro de cada dia.

IMPORTANTE: este script asume que se ejecuta con working directory = eda/
(igual que los scripts de EDA). Si tu Run configuration usa la raiz del
proyecto, quita el "../" inicial de DATA_PATH y OUTPUT_DIR.
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_PATH = Path("../../data/raw/RNN/")
OUTPUT_DIR = Path("../../data/processed/dataset_unificado")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REGIONES_VALIDAS = ["Caribe", "Pacifico_Central", "Pacifico_Sur"]

# Mapeo zona OpenMeteo (lat_lon) -> region IMN, obtenido por vecino mas
# cercano (haversine) contra las coordenadas de cada estacion IMN.
# Pacifico_Norte queda deliberadamente fuera: ningun punto de la grilla
# cae razonablemente cerca de esa estacion.
ZONA_A_REGION = {
    # Caribe (lat 10.0)
    "10.0_-81.0": "Caribe",
    "10.0_-81.25": "Caribe",
    "10.0_-81.5": "Caribe",
    "10.0_-81.75": "Caribe",
    "10.0_-82.0": "Caribe",
    "10.0_-82.25": "Caribe",
    "10.0_-82.5": "Caribe",
    "10.0_-82.75": "Caribe",
    "10.0_-83.0": "Caribe",
    # Pacifico_Sur (lat 8.0, lon -84.0 a -85.0)
    "8.0_-84.0": "Pacifico_Sur",
    "8.0_-84.25": "Pacifico_Sur",
    "8.0_-84.5": "Pacifico_Sur",
    "8.0_-84.75": "Pacifico_Sur",
    "8.0_-85.0": "Pacifico_Sur",
    # Pacifico_Central (lat 8.0, lon -85.25 a -86.5)
    "8.0_-85.25": "Pacifico_Central",
    "8.0_-85.5": "Pacifico_Central",
    "8.0_-85.75": "Pacifico_Central",
    "8.0_-86.0": "Pacifico_Central",
    "8.0_-86.25": "Pacifico_Central",
    "8.0_-86.5": "Pacifico_Central",
}


def cargar_openmeteo():
    df = pd.read_csv(
        DATA_PATH / "CostaRica_Marine_OpenMateo_2022_Hoy.csv",
        parse_dates=["datetime"],
    )
    df["zona"] = df["latitude"].astype(str) + "_" + df["longitude"].astype(str)
    df["Region"] = df["zona"].map(ZONA_A_REGION)
    df = df.dropna(subset=["Region"]).copy()

    # Promedio entre los puntos de grilla de cada region, por hora.
    df["datetime"] = df["datetime"].dt.tz_localize(None)
    agg = (
        df.groupby(["Region", "datetime"])[
            [
                "wave_height",
                "wave_direction",
                "sea_surface_temperature",
                "ocean_current_velocity",
                "ocean_current_direction",
            ]
        ]
        .mean()
        .reset_index()
    )
    return agg


def imputar_precipitacion(serie_region: pd.DataFrame) -> pd.Series:
    """Interpolacion temporal corta + climatologia mensual como respaldo."""
    s = serie_region.set_index("Fecha")["Precipitacion"]
    interp = s.interpolate(method="time", limit=3)
    mes = s.index.month
    climatologia = s.groupby(mes).mean()
    faltantes = interp.isna()
    if faltantes.any():
        interp.loc[faltantes] = pd.Series(mes, index=s.index)[faltantes].map(
            climatologia
        )
    return interp


def cargar_imn():
    df = pd.read_csv(DATA_PATH / "datos_IMN.csv", parse_dates=["Fecha"])
    df = df[df["Region"].isin(REGIONES_VALIDAS)].copy()

    partes = []
    for region, grupo in df.groupby("Region"):
        grupo = grupo.sort_values("Fecha").copy()
        if region == "Pacifico_Central":
            grupo["Precipitacion"] = imputar_precipitacion(
                grupo[["Fecha", "Precipitacion"]]
            ).values
        else:
            # Huecos residuales (<1%) en otras regiones: interpolacion simple.
            grupo["Precipitacion"] = grupo["Precipitacion"].interpolate(
                method="linear", limit_direction="both"
            )
        grupo["Temp_Maxima"] = grupo["Temp_Maxima"].interpolate(limit_direction="both")
        grupo["Temp_Minima"] = grupo["Temp_Minima"].interpolate(limit_direction="both")
        partes.append(grupo)

    return pd.concat(partes, ignore_index=True)


def cargar_fases_lunares():
    df = pd.read_csv(DATA_PATH / "moon_phases_UTC_2020-2050.csv", parse_dates=["Date"])
    return df[["Date", "Category", "PhaseName"]]


def expandir_a_horario(df_diario: pd.DataFrame, col_fecha: str, cols_valor: list) -> pd.DataFrame:
    """Convierte una serie diaria en horaria repitiendo el valor del dia
    en cada una de sus 24 horas (forward-fill dentro del dia)."""
    filas = []
    for _, fila in df_diario.iterrows():
        horas = pd.date_range(fila[col_fecha], periods=24, freq="h")
        base = {c: fila[c] for c in cols_valor}
        for h in horas:
            filas.append({**base, "datetime": h})
    return pd.DataFrame(filas)


def main():
    print("Cargando OpenMeteo (horario)...")
    openmeteo = cargar_openmeteo()

    print("Cargando IMN (diario) e imputando precipitacion...")
    imn = cargar_imn()

    print("Cargando fases lunares (diario)...")
    fases = cargar_fases_lunares()

    print("Expandiendo IMN a horario por region (forward-fill diario)...")
    imn_horario_partes = []
    for region, grupo in imn.groupby("Region"):
        cols = ["Region", "Temp_Promedio", "Temp_Minima", "Temp_Maxima",
                "Precipitacion", "Vel_Viento"]
        expandido = expandir_a_horario(grupo, "Fecha", cols)
        imn_horario_partes.append(expandido)
    imn_horario = pd.concat(imn_horario_partes, ignore_index=True)

    print("Expandiendo fases lunares a horario (sin zona, se une por fecha)...")
    fases_horario = expandir_a_horario(fases, "Date", ["Category", "PhaseName"])

    print("Uniendo OpenMeteo + IMN por Region y datetime...")
    dataset = pd.merge(
        openmeteo, imn_horario, on=["Region", "datetime"], how="inner"
    )

    print("Uniendo fases lunares por datetime...")
    dataset = pd.merge(dataset, fases_horario, on="datetime", how="left")

    dataset = dataset.sort_values(["Region", "datetime"]).reset_index(drop=True)

    out_path = OUTPUT_DIR / "dataset_final_horario.csv"
    dataset.to_csv(out_path, index=False)

    resumen = []
    resumen.append(f"Filas totales: {len(dataset):,}")
    resumen.append(f"Regiones incluidas: {sorted(dataset['Region'].unique())}")
    resumen.append(f"Rango de fechas: {dataset['datetime'].min()} -> {dataset['datetime'].max()}")
    resumen.append("\nFilas por region:")
    resumen.append(str(dataset["Region"].value_counts()))
    resumen.append("\n% nulos por columna:")
    resumen.append(str((dataset.isna().mean() * 100).round(2)))

    resumen_txt = "\n".join(resumen)
    print(resumen_txt)
    (OUTPUT_DIR / "resumen_dataset_final.txt").write_text(resumen_txt, encoding="utf-8")

    print(f"\nGuardado en: {out_path}")


if __name__ == "__main__":
    main()
