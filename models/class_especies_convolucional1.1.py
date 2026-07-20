"""
Clasificador de Especies Marinas - Red Neuronal Convolucional (CNN)
=====================================================================
Arquitectura:
    Conv2D(32) -> MaxPool -> Conv2D(64) -> MaxPool -> Conv2D(128)
    -> Flatten -> Dense(256) -> Dropout(0.5) -> Softmax

Exporta el modelo entrenado en formato .keras (nativo de Keras 3 / TF 2.x)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ---------------------------------------------------------------------------
# 1.COnfiguracion de hiperparametros y constantes del proyecto.
# ---------------------------------------------------------------------------
IMG_HEIGHT = 128 #Alto de las imagenes de 128 pixeles
IMG_WIDTH = 128 #Ancho de las imagenes de 128 pixeles
BATCH_SIZE = 32 #Cantidad de imagenes que procesara el modelo
EPOCHS = 30 #Cantidad de pasadas que le va realizar al dataset de entrenamiento
SEED = 123 #Semilla para asegurar resultados reproducibles


DATASET_DIR = "dataset_raw" #Ruta del dataset donde estan las imagenes que se seraparan en train(80) y test (20)
VALIDATION_SPLIT = 0.2  # 20% de las imágenes se reservan para test

MODEL_OUTPUT_PATH = "modelo_especies_marinas1.keras" #Nombre del modelo que se exportara el modelo .keras


# ---------------------------------------------------------------------------
# 2. CARGA DE DATOS= "dataset_raw"
# ---------------------------------------------------------------------------
def cargar_datasets(dataset_dir: str):
    """
    Carga las imágenes desde 'dataset_raw' (una subcarpeta por especie) y
    genera automáticamente la partición train/test usando validation_split.
    Usar el mismo 'seed' en ambas llamadas garantiza que las particiones
    sean complementarias (ninguna imagen se repite entre train y test).
    """

    #Carga de la particion Train
    train_ds = keras.utils.image_dataset_from_directory(
        dataset_dir,
        validation_split=VALIDATION_SPLIT,
        subset="training", #Train 80%
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        label_mode="categorical",  # etiquetas one-hot, compatible con softmax
        seed=SEED,
    )
    #Carga de la particion test
    test_ds = keras.utils.image_dataset_from_directory(
        dataset_dir,
        validation_split=VALIDATION_SPLIT,
        subset="validation", #Test 20%
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        seed=SEED,  # mismo seed que train -> particiones complementarias
    )

    #Extraccion de los nombres de la carpetas de la ruta "dataset_raw"
    class_names = train_ds.class_names
    print(f"Clases detectadas ({len(class_names)}): {class_names}") #Carpetas train
    print(f"Lotes de entrenamiento: {len(train_ds)} | Lotes de test: {len(test_ds)}") #Carpetas test

    # Optimización de rendimiento (cache + prefetch)
    #Autotune permite ajustar el tamaño del bufer segun sea el CPU/GPU del equipo.
    autotune = tf.data.AUTOTUNE
    #Permite cargar 1000 elementos aleatorios para evitar saturacion en train
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=autotune)
    #Carga los datos de test
    test_ds = test_ds.cache().prefetch(buffer_size=autotune)

    #Devuelve
    #train_ds =dataset de train 80%
    #test_ds = dataset de test 20%
    #class_names= nombre de los columnas del dataset inicial -dataset_raw-
    return train_ds, test_ds, class_names


# ---------------------------------------------------------------------------
# 3. AUMENTO DE DATOS (data augmentation)
# ---------------------------------------------------------------------------
def crear_capa_aumento():
    #Las capas de aumento se van a ejecutar en el entrenamiento model.fit() ya que en las demas etapas se desactivara.
    return keras.Sequential(
        [
            layers.RandomFlip("horizontal"), #Genera un volteo de la imagen como un espejo
            layers.RandomRotation(0.1), #Genera una rotacion a la imagen con el rango entre -36° y 36°
            layers.RandomZoom(0.1), #Genera un zoom aleatorio entre 90 y 110% de su tamaño original
            layers.RandomContrast(0.1), #Genera un contraste aleatorio del 10%
        ],
        name="aumento_datos",
    )


# ---------------------------------------------------------------------------
# 4. ARQUITECTURA DEL MODELO CON CAPAS CONVOLUCIONALES
# ---------------------------------------------------------------------------
def construir_modelo(num_clases: int) -> keras.Model:

    #Arquitectura resumida:
    #Conv2D(32) -> MaxPool -> Conv2D(64) -> MaxPool -> Conv2D(128)
    #-> Flatten -> Dense(256) -> Dropout(0.5) -> Softmax

    #Datos de entrada, define dimensiones de las imagenes a procesar y cantidad de dimensiones (RGB)
    entrada = keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3), name="imagen_entrada")
    #Normalizacion de las imagenes
    x = layers.Rescaling(1.0 / 255)(entrada)          # Normalización de la imagen del rango [0, 255] al rango [0.0, 1.0]
    x = crear_capa_aumento()(x)                        #Creacion de capa de aumento en las imagenes en train

    # Etapa#1 Detecta bordes y patrones simples (Salida: 64x64x32),
    #Capa convolucional (3x3) con 32 filtros
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same", name="conv2d_32")(x)
    #Capa maxpooling #(2x2)
    x = layers.MaxPooling2D((2, 2), name="maxpool_1")(x)

    # Etapa 2 Detecta texturas y formas intermedias (Salida: 32x32x64)
    # Capa convolucional (3x3) con 64 filtros
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same", name="conv2d_64")(x)
    # Capa maxpooling #(2x2)
    x = layers.MaxPooling2D((2, 2), name="maxpool_2")(x)

    # Etapa 3 Detecta estructuras complejas (Salida: 32x32x128)
    # Capa convolucional (3x3) con 128 filtros
    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same", name="conv2d_128")(x)

    # Conversion de las caracteristicas de 3D (32, 32, 128)- 1D (131,072 elementos)
    x = layers.Flatten(name="flatten")(x)

    # Capa interpretacion de caracteristicas aprendidas
    x = layers.Dense(256, activation="relu", name="dense_256")(x)

    #Capa dropout que apaga el 50% de las neuronas para evitar el sobre ajuste
    x = layers.Dropout(0.5, name="dropout_05")(x)

    #Capa salida multiclase softmax(1.0)
    salida = layers.Dense(num_clases, activation="softmax", name="softmax_salida")(x)

    #Creacion del modelo clasificador de especies
    modelo = keras.Model(inputs=entrada, outputs=salida, name="clasificador_especies_marinas")

    #Retorno del modelo
    return modelo


# ---------------------------------------------------------------------------
# 5. ENTRENAMIENTO DEL MODELO
# ---------------------------------------------------------------------------
def entrenar_modelo(modelo: keras.Model, train_ds, test_ds):

    #Compilacion del modelo con Adam, supervisando la perdida y la precision.
    modelo.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    #Supervision durante el entrenamiento del modelo
    callbacks = [
        #EarlyStopping evita que el modelo siga ejecutandose si detecta sobreajuste(El modelo memoriza y no aprende)
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        ),
        #ReduceLROnPlateau= Se asegura que cuando el modelo requiere mejorar pero no puede dar pasos grandes el modelo ajusta con pasos pequeños
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6
        ),
        #Punto de control donde el modelo obtuvo el valor mas alto precision
        keras.callbacks.ModelCheckpoint(
            "mejor_modelo.keras", monitor="val_accuracy", save_best_only=True
        ),
    ]

    # Nota: aunque el conjunto se llama 'test_ds', Keras internamente sigue
    # nombrando las métricas de validación como 'val_loss' / 'val_accuracy'.
    historial = modelo.fit(
        train_ds,
        validation_data=test_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
    )
    return historial


# ---------------------------------------------------------------------------
# 6. GRÁFICOS PARA ANÁLISIS DEL MODELO
# ---------------------------------------------------------------------------
def graficar_historial(historial, ruta_salida: str = "graficos_entrenamiento.png"):
    """
    Genera y guarda las curvas de precisión (accuracy) y pérdida (loss),
    comparando entrenamiento vs. test, por cada época.
    """
    acc = historial.history["accuracy"]
    val_acc = historial.history["val_accuracy"]
    loss = historial.history["loss"]
    val_loss = historial.history["val_loss"]
    rango_epocas = range(1, len(acc) + 1)

    fig, (ax_acc, ax_loss) = plt.subplots(1, 2, figsize=(12, 5))

    ax_acc.plot(rango_epocas, acc, marker="o", label="Entrenamiento")
    ax_acc.plot(rango_epocas, val_acc, marker="o", label="Test")
    ax_acc.set_title("Precisión (Accuracy) por época")
    ax_acc.set_xlabel("Época")
    ax_acc.set_ylabel("Precisión")
    ax_acc.legend()
    ax_acc.grid(alpha=0.3)

    ax_loss.plot(rango_epocas, loss, marker="o", label="Entrenamiento")
    ax_loss.plot(rango_epocas, val_loss, marker="o", label="Test")
    ax_loss.set_title("Pérdida (Loss) por época")
    ax_loss.set_xlabel("Época")
    ax_loss.set_ylabel("Pérdida")
    ax_loss.legend()
    ax_loss.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(ruta_salida, dpi=150)
    plt.show()
    print(f"Gráfico de entrenamiento guardado en: {ruta_salida}")


def graficar_matriz_confusion(
    modelo: keras.Model, test_ds, class_names, ruta_salida: str = "matriz_confusion.png"
):
    """
    Recorre el set de test, obtiene las predicciones del modelo y genera:
    - Matriz de confusión (imagen guardada en 'ruta_salida')
    - Reporte de clasificación (precision/recall/f1 por clase, impreso en consola)
    """
    y_true, y_pred = [], []
    for imagenes, etiquetas in test_ds:
        predicciones = modelo.predict(imagenes, verbose=0)
        y_pred.extend(np.argmax(predicciones, axis=1))
        y_true.extend(np.argmax(etiquetas.numpy(), axis=1))

    matriz = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(9, 8))
    plt.imshow(matriz, interpolation="nearest", cmap="Blues")
    plt.title("Matriz de Confusión")
    plt.colorbar()
    marcas = np.arange(len(class_names))
    plt.xticks(marcas, class_names, rotation=90)
    plt.yticks(marcas, class_names)

    umbral = matriz.max() / 2.0
    for i in range(matriz.shape[0]):
        for j in range(matriz.shape[1]):
            plt.text(
                j, i, format(matriz[i, j], "d"),
                ha="center", va="center",
                color="white" if matriz[i, j] > umbral else "black",
            )

    plt.ylabel("Etiqueta real")
    plt.xlabel("Etiqueta predicha")
    plt.tight_layout()
    plt.savefig(ruta_salida, dpi=150)
    plt.show()
    print(f"Matriz de confusión guardada en: {ruta_salida}")

    print("\nReporte de clasificación (por especie):")
    print(classification_report(y_true, y_pred, target_names=class_names))


# ---------------------------------------------------------------------------
# 7. EXPORTACIÓN DEL MODELO FUNCIONAL (.keras)
# ---------------------------------------------------------------------------
def exportar_modelo(modelo: keras.Model, ruta: str = MODEL_OUTPUT_PATH):
    """Guarda el modelo completo (arquitectura + pesos + optimizador) en .keras"""
    modelo.save(ruta)  # formato nativo .keras (recomendado desde Keras 3)
    print(f"Modelo exportado correctamente en: {ruta}")


def verificar_modelo_exportado(ruta: str = MODEL_OUTPUT_PATH):
    """Carga el modelo .keras y confirma que es funcional."""
    modelo_cargado = keras.models.load_model(ruta)
    modelo_cargado.summary()
    print("El modelo .keras se cargó correctamente y está listo para inferencia.")
    return modelo_cargado


# ---------------------------------------------------------------------------
# 8. FLUJO PRINCIPAL
# ---------------------------------------------------------------------------
def main():
    if not os.path.isdir(DATASET_DIR):
        raise FileNotFoundError(
            f"No se encontró la carpeta '{DATASET_DIR}'. "
            f"Debe contener una subcarpeta por cada especie, por ejemplo:\n"
            f"  {DATASET_DIR}/tiburon/\n"
            f"  {DATASET_DIR}/delfin/\n"
            f"  {DATASET_DIR}/tortuga/ ..."
        )

    train_ds, test_ds, class_names = cargar_datasets(DATASET_DIR)
    num_clases = len(class_names)

    modelo = construir_modelo(num_clases)
    modelo.summary()

    historial = entrenar_modelo(modelo, train_ds, test_ds)
    graficar_historial(historial)

    print("\nEvaluación final sobre el conjunto de test:")
    perdida, precision = modelo.evaluate(test_ds)
    print(f"  Pérdida (loss): {perdida:.4f}")
    print(f"  Precisión (accuracy): {precision:.4f}")

    graficar_matriz_confusion(modelo, test_ds, class_names)

    exportar_modelo(modelo, MODEL_OUTPUT_PATH)
    verificar_modelo_exportado(MODEL_OUTPUT_PATH)


if __name__ == "__main__":
    main()
