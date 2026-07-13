# 🌊 OceanoIA — Asistente Inteligente para Monitoreo Costero y Pesca Sostenible en Costa Rica

[![Carga de datos](https://img.shields.io/badge/Cargar%20datos-CSV-4CAF50?style=for-the-badge&logo=files)]()
[![Limpieza de datos](https://img.shields.io/badge/Limpieza%20y%20EDA-Procesamiento-2196F3?style=for-the-badge&logo=databricks)]()
[![Modelo CNN](https://img.shields.io/badge/Modelo-CNN-FF5722?style=for-the-badge&logo=tensorflow&logoColor=white)]()
[![Modelo RNN/LSTM](https://img.shields.io/badge/Modelo-RNN%2FLSTM-9C27B0?style=for-the-badge&logo=tensorflow&logoColor=white)]()
[![Documentación](https://img.shields.io/badge/Documentación-Notebook-795548?style=for-the-badge&logo=jupyter)]()  
[![Dashboard](https://img.shields.io/badge/Dashboard-Streamlit-E91E63?style=for-the-badge&logo=streamlit)]()  
[![API](https://img.shields.io/badge/🔌_API-Disponible-blue?style=for-the-badge)]()

> Proyecto académico del Colegio Universitario de Cartago (CUC) · Curso de Inteligencia Artificial

OceanoIA es un sistema inteligente que apoya a **INCOPESCA, MarViva, Guardacostas y pescadores artesanales** de Costa Rica integrando tres modelos de aprendizaje profundo en una sola plataforma:

| Módulo | Modelo | Objetivo |
|--------|--------|----------|
| 🐟 **Identificación de especies** | **CNN** | Clasificar peces capturados para verificar vedas, tallas mínimas y especies protegidas |
| 🌊 **Pronóstico oceanográfico** | **RNN / LSTM** | Predecir oleaje, viento, marea y SST en las próximas 24–72 h |

---

## 📂 Estructura del proyecto

```
OceanoIA/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/                    # Datos crudos (imágenes, CSV, mseed...)
│   └── processed/              # Datos limpios listos para entrenar
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_CNN_Especies.ipynb
│   ├── 03_RNN_Oceanografia.ipynb
├── src/
│   ├── data_prep.py            # Preprocesamiento general
│   ├── marine_api.py           # Cliente Open-Meteo Marine
│   └── train/
│       ├── cnn.py
│       ├── rnn.py
├── models/                     # Modelos entrenados .h5 / .keras
├── api/
│   └── main.py                 # FastAPI: /predict/especie, /predict/oceano
└── app/
    ├── Home.py                 # Frontend Streamlit
    └── assets/
```

---

## 🚀 Instalación rápida

### Requisitos previos
- Python 3.10+
- pip
- (Opcional) GPU con CUDA para acelerar el entrenamiento de la CNN

### Pasos

```bash
# 1. Clonar el repositorio
git clone https://github.com/<tu-usuario>/OceanoIA.git
cd OceanoIA

# 2. Crear entorno virtual
python -m venv venv
source venv/bin/activate          # Linux / macOS
# venv\Scripts\activate           # Windows

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Lanzar Jupyter para los notebooks
jupyter notebook notebooks/
```

---

## 🧪 Uso

### Entrenar los modelos

```bash
# CNN — Identificación de especies
python src/train/cnn.py

# RNN — Pronóstico oceanográfico
python src/train/rnn.py

# ANN — Recomendación de pesca
python src/train/ann.py
```

Los modelos entrenados se guardan en `models/`.

### Lanzar la app demo

```bash
streamlit run app/Home.py
```

Abrirá `http://localhost:8501` con tres pestañas:
- 📷 Identificador de especies
- 🌊 Pronóstico oceánico
- 📊 Dashboard combinado

### Lanzar la API REST (opcional)

```bash
uvicorn api.main:app --reload
```

Endpoints disponibles:
- `POST /predict/especie` (imagen)
- `POST /predict/oceano` (serie temporal)

---

## 📊 Datasets utilizados

| Dataset | Uso | URL |
|---------|-----|-----|
| Large-Scale Fish Dataset | CNN | https://www.kaggle.com/datasets/crowww/a-large-scale-fish-dataset |
| Fish4Knowledge | CNN | https://homepages.inf.ed.ac.uk/rbf/Fish4Knowledge/ |
| Open-Meteo Marine API | RNN | https://open-meteo.com/en/docs/marine-weather-api |
| NOAA ERDDAP | RNN | https://coastwatch.pfeg.noaa.gov/erddap/ |
| IMN Costa Rica | RNN | https://www.imn.ac.cr/ |

---

## 📈 Métricas objetivo

| Módulo | Métrica | Umbral |
|--------|---------|--------|
| CNN | Accuracy / F1-score | ≥ 90% |
| RNN | RMSE / MAE | Bajo (según escala) |

---

## 🏆 Rúbrica de evaluación

| Peso | Criterio |
|------|----------|
| 40 % | Modelo (métricas sobre datos reales) |
| 30 % | Producto (app funcional, demo en vivo) |
| 20 % | Documentación (informe técnico + README) |
| 10 % | Innovación (funcionalidades extra) |

---

## 🌟 Innovaciones implementadas

- ✅ Integración en tiempo real con **Open-Meteo Marine API**
- ✅ **Alerta automática** al detectar especies protegidas (tortugas, tiburones martillo, marlines)
- 🔄 Modo offline (TensorFlow Lite) — *en progreso*
- 🔄 Bot Telegram/WhatsApp — *creado*

---

## 🤝 Equipo

- Marco Álvarez Quirós.
- Sharon Obando Gómez.
- Fabián Brenes Loría.
- Johel Barquero Carvajal.

**Profesor:** Osvaldo Gónzalez Chaves  
**Curso:** Inteligencia Artificial 2026  
**Fecha de entrega:** 13 de julio de 2026

---

## 📜 Licencia

Este proyecto es de uso académico. Datasets utilizados respetan sus licencias originales.

---

## 🌎 Impacto social

OceanoIA contribuye al **ODS 14 (Vida Submarina)** apoyando a **+14,000 pescadores artesanales** de Costa Rica y al monitoreo de los **+589,000 km²** de mar territorial nacional, alineado con la estrategia de **economía azul** del MINAE.

---

## 🎯 Conclusión

_Este proyecto nos dejó un aprendizaje muy importante, el rendimiento de un modelo no depende tanto de la **cantidad** de imagenes/datos utilizados, sino de la **calidad** de estos._
