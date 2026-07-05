import streamlit as st
import folium
from streamlit_folium import st_folium
import requests
import pandas as pd
from datetime import date, timedelta

# ----------------------------------------------------------------------------
# Configuración de la página
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Dashboard Marino Open-Meteo", layout="wide")
st.title("⚓ Consulta de Datos Marinos Históricos")
st.write(
    "1) Haz clic en un punto del mapa (preferiblemente en el mar). "
    "2) Indica cuántos años de histórico quieres consultar (máx. 4). "
    "3) Presiona 'Obtener datos'."
)

# ----------------------------------------------------------------------------
# Límite real de disponibilidad de datos históricos marinos en Open-Meteo
# (el reanálisis de oleaje no llega a 80 años como el de variables
# atmosféricas; en la práctica arranca alrededor de 2021-2022). Se fija
# el máximo permitido en la app en 4 años, según lo solicitado.
# ----------------------------------------------------------------------------
ANIOS_POR_DEFECTO = 4
MAX_ANIOS_DISPONIBLES = min(ANIOS_POR_DEFECTO, date.today().year - 2021)

# ----------------------------------------------------------------------------
# Estado persistente de la sesión (aquí quedan guardados lat/lon, años y
# los DataFrames finales -horario y diario- para que otras secciones
# -análisis, predicción, etc.- puedan reutilizarlos sin volver a pedirlos)
# ----------------------------------------------------------------------------
if "marine_data_hourly" not in st.session_state:
    st.session_state["marine_data_hourly"] = None
if "marine_data_daily" not in st.session_state:
    st.session_state["marine_data_daily"] = None
if "lat" not in st.session_state:
    st.session_state["lat"] = None
if "lon" not in st.session_state:
    st.session_state["lon"] = None

col1, col2 = st.columns([1.2, 1])

# ----------------------------------------------------------------------------
# 1. Mapa interactivo para capturar coordenadas
# ----------------------------------------------------------------------------
with col1:
    st.subheader("🗺️ Selecciona una ubicación")
    m = folium.Map(location=[28.0, -15.0], zoom_start=4)
    folium.LatLngPopup().add_to(m)
    map_data = st_folium(m, width=700, height=450, key="mapa_marino")

    if map_data and map_data.get("last_clicked"):
        st.session_state["lat"] = map_data["last_clicked"]["lat"]
        st.session_state["lon"] = map_data["last_clicked"]["lng"]

# ----------------------------------------------------------------------------
# 2. Panel de control: coordenadas + años de histórico + botón de consulta
# ----------------------------------------------------------------------------
with col2:
    st.subheader("⚙️ Parámetros de la consulta")

    if st.session_state["lat"] is not None:
        st.success(
            f"**Latitud:** {st.session_state['lat']:.4f} | "
            f"**Longitud:** {st.session_state['lon']:.4f}"
        )
    else:
        st.info("👈 Haz clic en el mapa para fijar una ubicación.")

    anios = st.number_input(
        "¿Cuántos años de datos históricos quieres obtener?",
        min_value=1,
        max_value=MAX_ANIOS_DISPONIBLES,
        value=MAX_ANIOS_DISPONIBLES,
        step=1,
        help=(
            "Open-Meteo solo tiene reanálisis marino confiable desde "
            f"~2021, así que el máximo permitido en esta app es de "
            f"{MAX_ANIOS_DISPONIBLES} años."
        ),
    )

    consultar = st.button(
        "📥 Obtener datos",
        disabled=st.session_state["lat"] is None,
        use_container_width=True,
    )

# ----------------------------------------------------------------------------
# 3. Petición a la API con rango histórico (start_date / end_date)
# ----------------------------------------------------------------------------
if consultar and st.session_state["lat"] is not None:
    lat = st.session_state["lat"]
    lon = st.session_state["lon"]

    # end_date se fija "ayer" porque el dato histórico más reciente
    # todavía no está consolidado en el archivo
    end_date = date.today() - timedelta(days=1)
    start_date = end_date - timedelta(days=int(anios * 365))

    url = "https://marine-api.open-meteo.com/v1/marine"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": [
            "wave_height",
            "wave_direction",
            "wave_period",
            "swell_wave_height",
            "swell_wave_period",
            "wind_wave_height",
            "wind_wave_direction",
            "wind_wave_period",
            "sea_surface_temperature",
        ],
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "timezone": "auto",
    }

    with col2:
        with st.spinner(f"Consultando {anios} año(s) de histórico en Open-Meteo..."):
            try:
                response = requests.get(url, params=params, timeout=60)
            except requests.exceptions.RequestException as e:
                st.error(f"❌ Error de conexión: {e}")
                response = None

        if response is not None:
            if response.status_code == 200:
                data = response.json()
                hourly_data = data.get("hourly", {})

                if hourly_data and "time" in hourly_data:
                    df = pd.DataFrame(hourly_data)
                    df.rename(columns={"time": "fecha_hora"}, inplace=True)
                    df["fecha_hora"] = pd.to_datetime(df["fecha_hora"])

                    # --------------------------------------------------------
                    # Versión diaria: se calcula agregando el detalle horario
                    # (promedio para direcciones/periodos/temperatura, y
                    # máximo para las alturas de ola, que es lo relevante
                    # para análisis de riesgo/operatividad marítima)
                    # --------------------------------------------------------
                    columnas_valor = [c for c in df.columns if c != "fecha_hora"]
                    df_diario = (
                        df.set_index("fecha_hora")[columnas_valor]
                        .resample("D")
                        .agg(
                            {
                                "wave_height": "max",
                                "wave_direction": "mean",
                                "wave_period": "mean",
                                "swell_wave_height": "max",
                                "swell_wave_period": "mean",
                                "wind_wave_height": "max",
                                "wind_wave_direction": "mean",
                                "wind_wave_period": "mean",
                                "sea_surface_temperature": "mean",
                            }
                        )
                        .reset_index()
                        .rename(columns={"fecha_hora": "fecha"})
                    )

                    # --------------------------------------------------------
                    # Aquí quedan guardadas las variables que luego se pueden
                    # reutilizar en análisis / modelos de predicción
                    # --------------------------------------------------------
                    st.session_state["marine_data_hourly"] = df
                    st.session_state["marine_data_daily"] = df_diario

                    st.success(
                        f"✅ {len(df)} registros horarios "
                        f"({len(df_diario)} días) guardados en memoria."
                    )
                else:
                    st.warning(
                        "No se encontraron datos para ese punto/rango. "
                        "Prueba con coordenadas en el mar o menos años."
                    )
            else:
                st.error(f"❌ Error en la API. Código: {response.status_code}")
                try:
                    st.json(response.json())
                except ValueError:
                    st.write(response.text)

# ----------------------------------------------------------------------------
# 4. Vista previa de los datos ya guardados en session_state
# ----------------------------------------------------------------------------
df_horario = st.session_state["marine_data_hourly"]
df_diario = st.session_state["marine_data_daily"]

if df_horario is not None:
    st.markdown("---")
    st.subheader("📊 Datos disponibles para análisis y predicción")
    st.caption(
        "Los datos viven en `st.session_state['marine_data_hourly']` y "
        "`st.session_state['marine_data_daily']`, listos para pasarse a un "
        "modelo de series de tiempo, regresión, etc."
    )

    tab_horaria, tab_diaria = st.tabs(["🕐 Horaria", "📅 Diaria"])

    with tab_horaria:
        st.dataframe(df_horario, use_container_width=True)
        st.line_chart(data=df_horario, x="fecha_hora", y="wave_height")
        st.line_chart(data=df_horario, x="fecha_hora", y="sea_surface_temperature")

        csv_horario = df_horario.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Descargar CSV horario",
            data=csv_horario,
            file_name="datos_marinos_horarios.csv",
            mime="text/csv",
        )

    with tab_diaria:
        st.dataframe(df_diario, use_container_width=True)
        st.line_chart(data=df_diario, x="fecha", y="wave_height")
        st.line_chart(data=df_diario, x="fecha", y="sea_surface_temperature")

        csv_diario = df_diario.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Descargar CSV diario",
            data=csv_diario,
            file_name="datos_marinos_diarios.csv",
            mime="text/csv",
        )

st.markdown("---")
st.caption(
    "Datos meteorológicos provistos de manera gratuita por "
    "[Open-Meteo.com](https://open-meteo.com/) (Uso no comercial)."
)
