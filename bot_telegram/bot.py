"""
telegram_bot.py
================
Bot de Telegram para OceanoIA

Se debe tener ejecutado la API y para poder ejecutar el bot:
    python bot_telegram/bot.py

Como se creó este Bot:
1. En telegram se debe de buscar @BotFather, que es donde se crean los bots de telegram.
2. Le escribimos /start, este nos dara un menu en donde tenemos varias opciones.
3. Le marcamos la opción /newbot.
4. Este nos pedirá un nombre para nuestro bot el cual llamamos: OceanoIA CR Bot.
5. Luego de hacer el nombre nos pedirá un user el cual debe de terminar siempre en bot: @oceanoia_cr_bot.

Ya que tenemos nuestro bot, debemos de buscarlo por el user que le escribimos, este puede ser
usado por varios usuarios, siempre y cuando la API esté prendida.
"""

from __future__ import annotations # Nos deja usar dict[int, str] como tipo de datos para que
# Python no llegue a tener errores, es más para tener compatibilidad.

import io  # Libreria que trabaja con imágenes que la guarda en la memoria de la compu.
import os  # Libreria para leer variables de entorno (el token del bot).
import sys # Libreria que nos ayuda a encontrar las carpetas del proyecto.
from pathlib import Path # Libreria que nos ayuda a manejar rutas de carpetas/archivos.

# Esto permite que el bot pueda importar módulos de otras carpetas del proyecto.
sys.path.append(str(Path(__file__).resolve().parents[1]))

import requests  # Libreria para llamar a nuestra API.
from dotenv import load_dotenv  # Libreria para leer el archivo .env con el token

# python-telegram-bot: librería oficial para construir bots de Telegram.
from telegram import Update
from telegram.ext import (
    Application,  # El "motor" principal del bot
    CommandHandler,  # Para manejar comandos como /start, /clima
    ContextTypes,  # Tipo de dato que trae info del contexto de cada mensaje
    MessageHandler,  # Para manejar mensajes que NO son comandos (ej. fotos)
    filters,  # Para filtrar qué tipo de mensaje activa cada handler
)

# Carga las variables definidas en el archivo .env (como TELEGRAM_BOT_TOKEN)
# hacia las variables de entorno del sistema, para poder leerlas con os.getenv().
load_dotenv()

# Token de telegram, se realiza un .env en donde esta el token y en .gitignore
# se debe escribir .env para que no se suba.
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# API.
API_BASE_URL = "http://127.0.0.1:8000"

# Lista de zonas costeras.
ZONAS_VALIDAS = [
    "caribe",
    "pacifico_central",
    "pacifico_sur",
]


# ============================================================================
# HANDLERS
# ============================================================================
# Un "handler" es una función que Telegram ejecuta automáticamente cuando
# ocurre cierto evento (el usuario manda un comando, una foto, etc.).

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Se ejecuta cuando el usuario envía el comando /start.
    Es el primer mensaje que ve cualquiera al abrir el bot por primera vez.
    """
    await update.message.reply_text(
        "🌊 *¡Bienvenido a OceanoIA CR Bot 🇨🇷!*\n\n"
        "🤝 *Equipo*\n\n"
        "Marco Álvarez Quirós.\n\n"
        "Sharon Obando Gómez.\n\n"
        "Fabián Brenes Loría.\n\n"
        "Johel Barquero Carvajal.\n\n"
        "👨🏻‍🏫 *Profesor*\n\n"
        "Osvaldo Gónzalez Chaves.\n\n"
        "📚 *Curso*\n\n"
        "Inteligencia Artificial 2026.\n\n"
        "Somos estudiantes de la carrera Big Data del Colegio Universitario de Cartago, hemos creado este Bot para poder ayudar a:\n\n"
        "INCOPESCA, MarViva, Guardacostas y pescadores artesanales a identificar especies marinas y predecir condiciones oceanográficas\n\n"
        "🆘 *Como te podemos ayudar?*\n\n"
        "📷 Envía una *foto* de un pez capturado y te diré la especie "
        "y si está en veda, si es protegida o no.\n\n"
        "⛅ Usa `/clima <zona>` para el pronóstico oceánico. Zonas disponibles:\n"
        + "\n".join(f"• `{z}`" for z in ZONAS_VALIDAS),
        parse_mode="Markdown",  # Permite usar *negrita* y `código` en el mensaje
    )

# ============================================================================
# CNN - IMAGENES
# ============================================================================
async def manejar_foto(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Se ejecuta automáticamente cada vez que el bot recibe una FOTO
    (no un comando de texto). Este es el corazón:
    Foto -> especie + legalidad.
    """
    # Mostramos un mensaje temporal de "cargando" mientras se procesa,
    # porque la API puede tardar unos segundos (carga del modelo CNN).
    msg = await update.message.reply_text("°‧ 𓆝 𓆟 𓆞 ·｡ Analizando la especie...")

    try:
        # Telegram entrega la misma foto en varias resoluciones (thumbnails
        # más pequeños y la versión completa). El índice [-1] siempre
        # corresponde a la de MAYOR calidad disponible.
        foto = update.message.photo[-1]

        # get_file() obtiene la referencia al archivo en los servidores de
        # Telegram (aún no lo descarga, solo da el "puntero" al archivo).
        archivo = await foto.get_file()

        # Descargamos la imagen directo a memoria (BytesIO), sin necesidad
        # de guardarla como archivo temporal en el disco.
        buffer = io.BytesIO()
        await archivo.download_to_memory(buffer)
        buffer.seek(0)  # Regresamos el "cursor" al inicio del buffer para poder leerlo

        # Enviamos la imagen a nuestra propia API, exactamente igual que
        # hace el uploader de Streamlit.
        resp = requests.post(
            f"{API_BASE_URL}/predict/especie",
            files={"file": ("foto.jpg", buffer, "image/jpeg")},
            timeout=120,  # Le damos margen por si es la primera predicción
            # (carga del modelo TensorFlow, que es lenta la 1ra vez)
        )
        resp.raise_for_status()  # Lanza una excepción si la API respondió con error (4xx/5xx)
        resultado = resp.json()  # Convierte la respuesta JSON en un diccionario de Python

        # Extraemos los campos que nos interesan de la respuesta de la API
        # (mismo JSON que devuelve /predict/especie, ver main.py).
        especie = resultado["especie"].replace("_", " ").title()  # "dorado" -> "Dorado"
        confianza = resultado["confianza"] * 100  # 0.87 -> 87.0
        protegida = resultado["protegida"]  # True / False

        # Armamos el mensaje de "legalidad" según si la especie está protegida.
        # Esta es la parte de "legalidad" que pide el punto 4 del enunciado.
        if protegida:
            legalidad = (
                "🚨 **ALERTA — Especie protegida.**\n"
                "Devolver al mar inmediatamente. Está prohibida su captura."
            )
        else:
            legalidad = (
                "✅ Especie no protegida.\n"
                "Verificar talla mínima y veda."
            )

        # Armamos el mensaje final combinando especie + confianza + legalidad.
        texto = (
            f"🐟 *Especie:* {especie}\n"
            f"📊 *Confianza:* {confianza:.1f}%\n\n"
            f"{legalidad}"
        )

        # Editamos el mensaje de "Analizando..." con el resultado final,
        # en vez de enviar un mensaje nuevo.
        await msg.edit_text(texto, parse_mode="Markdown")

    # Manejo de errores: cada except cubre un tipo de falla distinto ---
    except requests.exceptions.ConnectionError:
        # Pasa si uvicorn no está corriendo, o la URL/puerto está mal.
        await msg.edit_text(
            "❌ No se pudo conectar con la API. Verifica que uvicorn esté corriendo."
        )
    except requests.exceptions.HTTPError:
        # Pasa si la API respondió pero con un código de error (4xx/5xx),
        # por ejemplo si el modelo no se pudo cargar (503).
        detalle = resp.json().get("detail", resp.text)
        await msg.edit_text(f"❌ Error de la API: {detalle}")
    except Exception as e:
        # Cualquier otro error inesperado (imagen corrupta, timeout, etc.)
        await msg.edit_text(f"❌ Error inesperado: {e}")

# ============================================================================
# RNN - CLIMA
# ============================================================================
async def clima(update: Update, context: ContextTypes.DEFAULT_TYPE):

    if not context.args:
        await update.message.reply_text(
            "Uso: /clima <zona>\n\n"
            "Zonas disponibles:\n"
            + "\n".join(ZONAS_VALIDAS)
        )
        return

    zona = context.args[0].lower()

    if zona not in ZONAS_VALIDAS:
        await update.message.reply_text("Zona no válida.")
        return

    msg = await update.message.reply_text("🌊 Consultando pronóstico...")

    try:
        import requests

        API_URL = "http://127.0.0.1:8000"

        r = requests.get(
            f"{API_URL}/predict/oceano",
            params={"zona": zona}
        )

        r.raise_for_status()

        datos = r.json()

        texto = (
            f"🌊 *Pronóstico — {zona.replace('_',' ').title()}*\n\n"
            f"🌊 Oleaje máx. 72h: {datos['metricas']['oleaje_max']:.2f} m\n"
            f"🌡️ SST promedio: {datos['metricas']['sst']:.1f} °C\n"
            f"⏱️ Período promedio: {datos['metricas']['periodo']:.1f} s"
        )

        await msg.edit_text(texto, parse_mode="Markdown")

    except Exception as e:
        await msg.edit_text(f"❌ Error: {e}")


# ============================================================================
# PUNTO DE ENTRADA DEL PROGRAMA
# ============================================================================

def main() -> None:
    """
    Arma el bot.
    """
    if not TOKEN:
        # Si no hay token, paramos con un mensaje claro.
        raise RuntimeError(
            "No se encontró TELEGRAM_BOT_TOKEN. "
            "Crea un archivo .env en la raíz de OceanoIA con esa variable."
        )

    # Application es el objeto principal de python-telegram-bot; se conecta
    # a los servidores de Telegram usando el token.
    app = Application.builder().token(TOKEN).build()

    # Registramos cada handler.
    app.add_handler(CommandHandler("start", start))  # /start
    app.add_handler(CommandHandler("clima", clima))  # /clima <zona>
    app.add_handler(MessageHandler(filters.PHOTO, manejar_foto))  # cualquier foto

    print("📲 Bot activado")

    # Bloquea el programa aquí.
    app.run_polling()


if __name__ == "__main__":
    main()
