from pathlib import Path
import pandas as pd


class ProcesadorFasesLunares:
    NOMBRES_FASES = {
        0: "Luna nueva",
        1: "Luna creciente",
        2: "Cuarto creciente",
        3: "Luna gibosa creciente",
        4: "Luna llena",
        5: "Luna gibosa menguante",
        6: "Cuarto menguante",
        7: "Luna menguante",
    }

    EMOJIS_FASES = {
        0: "🌑",
        1: "🌒",
        2: "🌓",
        3: "🌔",
        4: "🌕",
        5: "🌖",
        6: "🌗",
        7: "🌘",
    }

    def __init__(self, archivo_entrada: str | Path):
        self.archivo_entrada = Path(archivo_entrada)
        self.datos = None

    def cargar_datos(self):
        """Carga el archivo CSV."""
        self.datos = pd.read_csv(self.archivo_entrada)
        return self

    def enriquecer_fases(self):
        """Agrega nombre y emoji de la fase lunar."""
        self.datos["NombreFase"] = (
            self.datos["Category"]
            .map(self.NOMBRES_FASES)
        )

        self.datos["EmojiFase"] = (
            self.datos["Category"]
            .map(self.EMOJIS_FASES)
        )

        return self

    def eliminar_columnas(self, columnas=None):
        """Elimina columnas no necesarias."""
        columnas = columnas or ["Area", "Phase"]

        columnas_existentes = [
            columna
            for columna in columnas
            if columna in self.datos.columns
        ]

        self.datos.drop(
            columns=columnas_existentes,
            inplace=True
        )

        return self

    def guardar(self, archivo_salida: str | Path):
        """Guarda el resultado procesado."""
        archivo_salida = Path(archivo_salida)

        archivo_salida.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        self.datos.to_csv(
            archivo_salida,
            index=False,
            encoding="utf-8-sig"
        )

        print(f"Archivo guardado en: {archivo_salida}")

        return self

    def mostrar(self, filas: int = 10):
        """Muestra una vista previa del dataset."""
        print(self.datos.head(filas))
        return self