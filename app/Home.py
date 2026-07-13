"""
Home.py
=======
App Streamlit demo de OceanoIA con cuatro pestañas:
  1. 📷 Identificador de especies (CNN)
  2. 🌊 Pronóstico oceánico (RNN)
  3. 📊 Dashboard combinado

Recomendación:
La API debe de ejecutarse primero
    uvicorn api.main:app --reload

Ejecutar:
    streamlit run app/Home.py
"""

from __future__ import annotations # Nos deja usar dict[int, str] como tipo de datos para que
# Python no llegue a tener errores, es más para tener compatibilidad.

import sys # Libreria que nos ayuda a encontrar las carpetas del proyecto.
from pathlib import Path # Libreria que nos ayuda a manejar rutas de carpetas/archivos.

# Permite a Python importar módulos desde la carpeta raíz del proyecto.
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np # Librería para trabajar con números y matrices.
import pandas as pd # Libreria para trabajar con tablas y datos.
import requests # Libreria se comunica con la API.
import streamlit as st # Crea la interfaz gráfica de la aplicación.

# ===================== Config general =====================
st.set_page_config(
    page_title="OceanoIA · Pesca Sostenible",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# URL base de la API FastAPI
API_BASE_URL = "http://127.0.0.1:8000"

# ===================== Sidebar =====================
st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0a/Flag_of_Costa_Rica.svg/320px-Flag_of_Costa_Rica.svg.png",
    width=180,
)
st.sidebar.title("🌊 OceanoIA")
st.sidebar.markdown("**Monitoreo Costero y Pesca Sostenible en Costa Rica**")
st.sidebar.markdown("---")
st.sidebar.markdown("**Proyecto académico — CUC**")
st.sidebar.markdown("Curso de Inteligencia Artificial 2026")

# ===================== Header principal =====================
st.title("🌊 OceanoIA")
st.markdown(
    "#### Asistente Inteligente para **Monitoreo Costero y Pesca Sostenible**"
)
st.markdown("---")

# ===================== Tabs =====================
tab1, tab2, tab3 = st.tabs([
    "📷 Identificar Especie",
    "🌊 Pronóstico Oceánico",
    "📊 Dashboard",
])

# ===================== TAB 1: CNN Identificación =====================
with tab1:
    st.header("📷 Identificación de especies marinas (CNN)")
    st.markdown(
        "Sube una foto del pez capturado. El modelo identifica la especie y verifica "
        "si está en veda, es protegida o cumple talla mínima."
    )

    uploaded = st.file_uploader("Selecciona una imagen", type=["jpg", "jpeg", "png"])
    if uploaded:
        col1, col2 = st.columns(2)
        with col1:
            st.image(uploaded, caption="Imagen cargada", use_column_width=True)

        with col2:
            with st.spinner("Analizando..."):
                try:
                    # uploaded es un BytesIO-like que ya viene desde st.file_uploader;
                    # reseteamos el puntero por si Streamlit ya lo leyó para el st.image de arriba.
                    uploaded.seek(0)
                    files = {"file": (uploaded.name, uploaded, uploaded.type)}
                    resp = requests.post(
                        f"{API_BASE_URL}/predict/especie",
                        files=files,
                        timeout=120,
                    )
                    resp.raise_for_status()
                    resultado = resp.json()
                except requests.exceptions.ConnectionError:
                    st.error(
                        "❌ No se pudo conectar con la API. "
                        f"¿Está corriendo uvicorn en `{API_BASE_URL}`? "
                        "Ejecuta `uvicorn api.main:app --reload` en otra terminal."
                    )
                    st.stop()
                except requests.exceptions.HTTPError:
                    detalle = resp.json().get("detail", resp.text)
                    st.error(f"❌ Error de la API ({resp.status_code}): {detalle}")
                    st.stop()
                except Exception as e:
                    st.error(f"❌ Error inesperado: {e}")
                    st.stop()

            especie_pred = resultado["especie"]
            confianza    = resultado["confianza"]
            protegida    = resultado["protegida"]
            todas        = resultado["todas"]

            st.success(f"**Especie:** {especie_pred.replace('_', ' ').title()}")
            st.metric("Confianza", f"{confianza * 100:.1f}%")

            if protegida:
                st.error("🚨 **ALERTA — Especie protegida.** Devolver al mar inmediatamente. Está prohibida su captura.")
            else:
                st.info("✅ Especie no protegida. Verificar talla mínima y veda.")

            st.markdown("**Probabilidades por clase:**")
            df_probs = pd.DataFrame({
                "Especie": list(todas.keys()),
                "Probabilidad": list(todas.values()),
            }).sort_values("Probabilidad", ascending=False)
            st.bar_chart(df_probs.set_index("Especie"))

# ===================== TAB 2: RNN Pronóstico =====================
with tab2:
    st.header("🌊 Pronóstico oceánico — próximas 72 horas")
    st.markdown("Predicción de oleaje, temperatura del mar y viento.")

    zonas = [
        "Caribe",
        "Pacifico_Central",
        "Pacifico_Sur",
    ]

    zona = st.selectbox("Selecciona una zona costera:", zonas)

    if st.button("Obtener pronóstico"):

        try:
            r = requests.get(
                f"{API_BASE_URL}/predict/oceano",
                params={"zona": zona}
            )

            datos = r.json()

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Oleaje máx. 72h",
                f"{datos['metricas']['oleaje_max']:.2f} m"
            )

            col2.metric(
                "SST promedio",
                f"{datos['metricas']['sst']:.1f} °C"
            )

            col3.metric(
                "Período promedio",
                f"{datos['metricas']['periodo']:.1f} s"
            )

            df = pd.DataFrame({
                "Oleaje": datos["wave_height"],
                "Temperatura": datos["sea_surface_temperature"]
            })

            st.line_chart(df["Oleaje"])
            st.line_chart(df["Temperatura"])

        except Exception as e:
            st.error(e)

# ===================== TAB 3: Dashboard =====================
with tab3:
    st.header("📊 Dashboard integrado")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Especies catalogadas", "16")
    col2.metric("Zonas monitoreadas", "7")
    col3.metric("Pescadores beneficiados", "14,000+")
    col4.metric("Km² de mar territorial", "589,000")

    st.markdown("---")
    st.subheader("Datasets utilizados")
    st.table(pd.DataFrame({
        "Módulo": ["CNN", "RNN/LSTM"],
        "Dataset principal": [
            "Large-Scale Fish Dataset (Kaggle)",
            "Open-Meteo Marine API + NOAA + IMN",
        ],
        "Métrica":  ["Accuracy / F1 ≥ 90%", "RMSE / MAE"],
    }))

    st.markdown("---")
    st.markdown(
        "**Equipo:** Marco Álvarez Quirós · Sharon Obando Gómez · Fabián Brenes Loría · Johel Barquero Carvajal ·"
        "**Profesor:** Osvaldo González Chaves · "
        "**Curso:** Inteligencia Artificial 2026  ·  "
        "**Entrega:** 13 de julio 2026 ·"
    )
