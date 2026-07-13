# **Uso de la IA 🤖**

[![Claude](https://img.shields.io/badge/Claude-CC785C?style=for-the-badge&logo=claude&logoColor=white)]()
[![Gemini](https://img.shields.io/badge/Gemini-4285F4?style=for-the-badge&logo=googlegemini&logoColor=white)]()

-------------------------------------------------

## **Requisitos**
1. Qué herramientas de IA utilizaron y para qué tareas específicas
2. Ejemplos de prompts utilizados (mínimo 3-5 ejemplos representativos)
3. Reflexión sobre cómo la IA ayudó en su aprendizaje (1-2 párrafos)
4. Qué modificaciones hicieron al código/análisis generado por IA

-------------------------------------------------


## **Información sobre archivos extencion .nc**
**Herramienta IA utilizada:** **Gemini**

**Prompt:**
```
¿que son y como poder ver archivos de formato .nc?
```
**Respuesta recibida:**
```
Los archivos con la extensión .nc corresponden al formato NetCDF (Network Common Data Form).
En el mundo científico, son muy comunes. Piensa en ellos como super-archivos de Excel diseñados específicamente para almacenar datos científicos multidimensionales. Se usan muchísimo en meteorología, oceanografía, climatología y geografía (por ejemplo, para guardar la temperatura global, la presión atmosférica o la humedad de los últimos 50 años).


¿Cómo abrir y ver archivos .nc?
Como no son archivos de texto plano, no puedes abrirlos con el bloc de notas. Dependiendo de si solo quieres "ver" los mapas o si quieres trabajar con los datos, tienes varias opciones:
3. Para Programadores o Analistas de Datos
Si necesitas extraer los números para hacer cálculos o estadísticas, la mejor opción es usar código: Python: Es el rey para esto. Utiliza librerías como xarray o netCDF4. Con un par de líneas de código puedes abrir el archivo y transformarlo:
```

```python
import xarray as xr
dataset = xr.open_dataset('tu_archivo.nc')
print(dataset) # Esto te muestra todas las variables y dimensiones
```
## **Consulta sobre proceso de imputación de datos**
**Herramienta IA utilizada:** **Copilot**

**Prompt**
```
¿qué es la imputación por interpolación?
```
**Respuesta recibida:**
```
*La imputación por interpolación es una técnica utilizada para rellenar datos faltantes en un conjunto de datos estimando los valores ausentes a partir de los valores conocidos cercanos. La idea básica es que, si tienes observaciones antes y después del dato faltante, puedes calcular un valor intermedio razonable.*

Ejemplo sencillo

Supongamos que una serie de temperaturas tiene un dato faltante:
```
| Día | Temperatura |
|------|------------|
| 1 | 20 °C |
| 2 | ? |
| 3 | 24 °C |
```
Con interpolación lineal, el valor faltante del día 2 sería:
```
$
20+24−202=2220 + \frac{24-20}{2} = 2220+224−20​=22
$
```
Por tanto:
```

| Día | Temperatura |
|------|------------|
| 1 | 20 °C |
| 2 | 22 °C |
| 3 | 24 °C |

## Modificaciones realizadas al código/análisis generado por la IA

 **Eliminación de `Category` como variable objetivo.** La IA propuso tres
   caminos posibles (mantenerla, reemplazarla por una clasificación real de
   oleaje, o eliminarla) tras detectar la fuga de datos. Se decidió
   eliminarla por completo del pipeline (ni feature de clasificación ni
   target), en vez de aceptar alguna de las otras dos alternativas
   propuestas, priorizando un modelo honesto sobre uno con una métrica de
   accuracy artificialmente alta.

**Ajuste del alcance de la meta de 90% de accuracy.** En lugar de exigir
   ese porcentaje de forma pareja para todas las variables (lo cual el
   propio análisis mostró que no era razonable para variables como
   `Precipitacion` u `ocean_current_velocity`), se redirigió el trabajo para
   que el esfuerzo de mejora (codificación circular de direcciones,
   tolerancias específicas por tipo de variable, función de pérdida más
   robusta) se concentrara en las variables donde alcanzar 90% sí tenía
   sentido físico y estadístico, documentando explícitamente en el notebook
   final cuáles variables cumplen la meta y cuáles no, y por qué.

---
## **Explicación sobre la carga perezosa para CNN**
**Herramienta IA utilizada:** **Gemini**

**Prompt #1:**
```
Me podrias explicar esta funcion para saber como puedo cargar el modelo?
```
```python
_models = {}

def _load(key: str, path: str):
    """Carga modelos solo cuando se necesitan."""
    if key not in _models:
        import tensorflow as tf
        try:
            _models[key] = tf.keras.models.load_model(path)
        except Exception as e:
            raise HTTPException(
                status_code=503,
                detail=f"Modelo no disponible: {path}. Entrénalo primero. ({e})",
            )
    return _models[key]
```
**Respuesta recibida:**
```
Esta función implementa un patrón llamado Carga Perezosa. 
Su objetivo es que el modelo de Inteligencia Artificial no se cargue en la memoria
RAM en cuanto se enciende el servidor (lo cual ralentizaría el inicio de la API), 
sino únicamente cuando un usuario realiza la primera petición que lo necesita. 
La línea `if key not in _models:` revisa si el modelo ya fue cargado previamente;
si ya existe en el diccionario `_models`, se lo salta y lo devuelve de inmediato, 
ahorrando muchísimo tiempo de procesamiento en las siguientes llamadas.
```
**Prompt #2:**
```
En el codigo que te pase para que sirve el _models = {} que esta afuera de la funcion?
```
**Respuesta recibida:**
```
Al estar declarado fuera de la función, _models actúa como una variable global o 
caché en memoria. En Python, las variables dentro de una función desaparecen cuando 
esta termina (local scope), pero las globales persisten durante todo el ciclo de 
vida en que el servidor de FastAPI esté encendido. Funciona como una "bóveda": la 
primera vez guardas el modelo ahí dentro usando una clave (key), y en las siguientes 
peticiones la función va directamente a la bóveda a sacarlo en lugar de leer el disco duro cada vez.
```
## Reflexión.
```
Esta IA me sirvio mucho como para poder entender esa parte del codigo ya
que se me hacia un enredo. Me explico como funciona la "carga perezosa" 
y como se usa el diccionario global _models = {} para guardar los modelos en la memoria 
de la compu, con esto entendi que no hay que cargar todo desde el inicio
para no saturar la RAM y al dejar el modelo guardado ahi, las respuestas de la 
API son mas rapidas a partir de la segunda consulta.
```