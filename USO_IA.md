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

## **Primer Caso - Nivel de Riesgo**

**1. Herramienta IA:** Utilice **Claude** y fue para la tarea de crear el risk_score del Modelo2.

**2. Ejemplo del prompts:**
```
Estoy desarrollando un proyecto de un sistema de prediccion de abandono de clientes, 
donde se utiliza la variable objetivo Churn, pero necesito entender como implementar un 
modelo de scoring de riesgo que pueda predecir la probabilidad del Churn entre 0.0 y 1.0 
y clasificar los clientes por el nicel de riesgo en Alto, Medio, Bajo, que funciones deberia
de usar

Codigo generado:

¡Buena pregunta! Para implementar un modelo de scoring de riesgo de Churn necesitás combinar una red neuronal con una función de clasificación posterior.

**La clave está en la capa de salida.** Usás `sigmoid` como función de activación, que matemáticamente convierte cualquier número en un valor entre 0 y 1, que interpretás directamente como probabilidad:

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Arquitectura del modelo
model = Sequential()
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))  # ← devuelve valor entre 0.0 y 1.0

model.compile(loss='binary_crossentropy', optimizer='adam')
```

**Para obtener el score y clasificar por nivel:**

```python
scores = model.predict(X_test)

            resultados = pd.DataFrame()
            resultados['risk_score'] = scores.flatten()

            def nivel(score): 
                if score >= 0.7:
                    return 'Alto riesgo'
                elif score >= 0.4:
                    return 'Medio riesgo'
                else:
                    return 'Bajo riesgo'

            resultados['nivel_riesgo'] = resultados['risk_score'].apply(nivel)

            print('\nDistribución por nivel de riesgo:')
            print(resultados['nivel_riesgo'].value_counts())
            return resultados
```

**Las funciones clave y para qué sirve cada una:**

| Función | Para qué sirve |
|---|---|
| `sigmoid` | Convierte la salida en probabilidad 0.0–1.0 |
| `binary_crossentropy` | Loss ideal para predecir probabilidades entre dos clases |
| `model.predict()` | Genera el score por cliente |
| `flatten()` | Convierte la salida de `[[0.8]]` a `[0.8]` |
| `EarlyStopping` | Detiene el entrenamiento antes de que memorice los datos |

La diferencia con un modelo de clasificación binaria es que no aplicás un umbral fijo de 0.5 — dejás el número exacto para poder ordenar y priorizar clientes por nivel de urgencia.

¿Necesitás que profundice en alguna parte específica?

----------------------
**3. Reflexión:** La IA me ayudo bastante con este metodo, ya que, me hizo que recordar sobre 
la materia de Fundamentos de Programacion, ya que, al consultar sobre como pueda clasificar
a los clientes por el nivel de riesgo, la IA me genero una estructura de control condicional,
en donde la aplico en una funcion, lo que me ayudo a recordar a como debo de organizar mi 
logica a la hora de programar.  

**4. Modificacion del código:** Lo que hice con lo que me genero Claude fue que cree la clase de **predecir_riesgo**
y ahi agregue el metodo:
```python
# 5. Creacion para predecir el nivel de riesgo (risk_score) (Consultado con Claude).
    def predecir_riesgo(self, X_test): # Creamos el metodo de predecir el riesgo y como parametro agregamos el X.test
            scores = self.model.predict(X_test)

            resultados = pd.DataFrame()
            resultados['risk_score'] = scores.flatten()

            def nivel(score): # creamos el nivel de riesgo.
                if score >= 0.7:
                    return 'Alto riesgo'
                elif score >= 0.4:
                    return 'Medio riesgo'
                else:
                    return 'Bajo riesgo'

            resultados['nivel_riesgo'] = resultados['risk_score'].apply(nivel)

            print('\nDistribución por nivel de riesgo:')
            print(resultados['nivel_riesgo'].value_counts())
            return resultados
```
## **Segundo Caso - Gráfico**

**1. Herramienta IA:** Utilice **Claude** y fue que el gráfico no lograba entender.

**2. Ejemplo del prompts:**
```
Explicame porque mi grafico sale asi
# Visualizamos nuestra prediccion
        plt.scatter(y_test, predictions)
        plt.xlabel('Valor Real')
        plt.ylabel('Prediccion')
#Predicción perfecta
        plt.plot(y_test,y_test,'r')
        plt.show()
        
Codigo generado:

¡El problema está en esta línea!

```python
# ❌ Esto es lo que genera todas esas rayas
plt.plot(y_test, y_test, 'r')
```

Lo que está pasando es que `plt.plot()` conecta todos los puntos con líneas en el orden en que aparecen en el dataset. Como `y_test` tiene valores de 0 y 1 desordenados, traza una línea de cada punto al siguiente saltando de 0 a 1 y de 1 a 0 constantemente — por eso ves ese caos de rayas rojas cruzadas.

Lo que querés dibujar es simplemente una línea diagonal recta del punto (0,0) al punto (1,1) que represente la predicción perfecta. Debés cambiar esa línea por esto:

```python
plt.scatter(y_test, predictions)
plt.xlabel('Valor Real')
plt.ylabel('Prediccion')
# Cambio
plt.plot([0, 1], [0, 1], 'r--', label='Predicción perfecta') 
plt.show()
```

La diferencia es que ahora le decís exactamente dos puntos fijos `[0,1]` en X y `[0,1]` en Y, y traza una línea recta entre ellos. El `'r--'` la pone roja y punteada para que se distinga del scatter. 😊

----------------------
**3. Reflexión:** Cuando estaba realizando la clase de **prediccion** me encontre con un error en la 
visualizacion del grafico en donde en lugar de mostrar una linea diagonal, el grafico generaba un monton 
de lineas cruzadas que hacian imposible interpretar los resultados. 
Al consultar con Claude, el problema estaba en el uso de plt.plot(y_test, y_test, 'r'), ya que este conectaba 
todos los puntos del conjunto de prueba en el orden en que aparecian en el dataset, generando esa dificultad de
poder interpretar el grafico.
Por lo que, la IA me explico que para representar una linea de prediccion perfecta se debia indicar dos puntos
fijos [0, 1] en lugar de pasarle los datos reales, lo que me ayudo a comprender mejor como funciona 
plt.plot() y la diferencia entre graficar datos reales y graficar una referencia visual.  

**4. Modificacion del código:** Agregue el cambio de esa linea, y el grafico ya se puede interpretar mejor.
```python
# Visualizamos nuestra prediccion
        plt.scatter(y_test, predictions)
        plt.xlabel('Valor Real')
        plt.ylabel('Prediccion')

#Predicción perfecta
        plt.plot([0, 1], [0, 1], 'r--', label='Predicción perfecta') # Consultado con Claude (error con grafico)
        plt.show()
```
## **Tercer Caso - Creación de set de datos *"dummy"* para la prediccion y prueba del modelo**
**1. Herramienta IA:** Utilice **Copilot**  

**2. Ejemplo del propmt:**  
*usando como ejemplo el .csv compartido, crea un set de datos con información numérica dummy,   
para usar como testeo de un modelo ANN, que tengan al menos 5 variables y 3000 individuos para descargar, puede ser en formato CSV*

Se creo satisfactoriamente el documento solicitado, el cual se encuentra en el repositorio de github con el nombre: **"telco_churn_modelo_3000.csv"**, con lo que queda evidenciado la efectividad y rapidéz del IA en la creación de documentos, ahorrando con esto tiempo que puede ser utilizado en desarrollo.
```
```
## **Cuarto Caso - Interpretación y solución a error recibido**  
**1. Herramienta IA:** Utilice **Gemini**  

**2. Ejemplo del propmt:**  
*cuál es la razón y cómo solucionar el siguiente error recibido en Pycharm: \.venv\Scripts\python.exe: can't open file 'C:\\Users\\fab_t\\Documents\\Codigo\\IA_Aplicada\\ProyectoB_ChurnClientes\\models\\clasificacion_binaria.py': [Errno 2] No such file or directory*  
**Respuesta recibida:**  

Problema con la selección en el IDE (visto en tu captura/contexto)
Si estás usando un IDE como PyCharm o VS Code e intentaste ejecutar solo una parte del código o el archivo actual haciendo clic derecho, a veces el entorno se confunde con las rutas relativas.
	• En PyCharm: Haz clic derecho directamente sobre el archivo clasificacion_binaria.py en el árbol de archivos de la izquierda y selecciona Run 'clasificacion_binaria'.

Con la respuesta dada por la IA, se logro detectar un copartamiento del IDE que no es fácil de reconocer, y a la vez tomarlo en cuenta para futuras situaciones similares que puedan ocurren en el que IDE no detecta las rutas relativas del los archivos.
```
```
## **Quinto Caso - Interpretación y solución a error recibido al testear el modelo binario**  
**1. Herramienta IA:** Utilice **Copilot**  

**2. Ejemplo del propmt:**  
*por qué razón se da el siguiente error y cómo puedo resolverlo: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior. _warn_prf(average, modifier, f"{metric.capitalize()} is", result.shape[0])*

**Respuesta recibida:**  
*Este no es un error, es un warning de scikit‑learn que te está avisando de un problema en tu modelo.
Hay clases (labels) que están en y_true (datos reales),
pero tu modelo nunca las predijo en y_pred.*

La respuesta dada por la IA, permitió identificar que el problema era debido a que el modelo no pudo predecir ninguna clase en el conjunto de datos de prueba para la variable con menor presencia en el conjunto de datos de entrenamiento.
Error que se logró resolver aplicando balance de cargas y ajustando el modelo.

```python
weights = class_weight.compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_train),
            y=y_train
        )

        class_weights = dict(enumerate(weights))
```
---