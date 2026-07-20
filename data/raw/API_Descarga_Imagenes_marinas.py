import os
import requests
import time


def descargar_atun_seguro():
    # 1. Crear la estructura de carpetas
    folder_path = os.path.join('dataset_raw', 'Marlin Pez Vela')
    os.makedirs(folder_path, exist_ok=True)

    # ID taxonómico REAL de iNaturalist para Thunnus albacares
    taxon_id = 115167

    #69675 Atun aleta amarilla
    #124316 Pargo rojo
    #278854,318285 Corvina Reina
    #115167 Marlin Pez Vela Espada
    #39682 Tortuga Marina
    #56764,47231 Tiburon martillo

    # Configuramos los parámetros para pedir 50 fotos verificadas por científicos (Research Grade)
    url = "https://api.inaturalist.org/v1/observations"
    params = {
        'taxon_id': taxon_id,
        'quality_grade': 'research',
        'per_page': 250,
        'has[]': 'photos'
    }

    # Encabezado para identificarnos amigablemente ante el servidor
    headers = {'User-Agent': 'MiProyectoCNN-Estudiante/1.0'}

    print("Conectando de forma segura con la API de iNaturalist...")
    try:
        response = requests.get(url, params=params, headers=headers)
        data = response.json()
    except Exception as e:
        print(f"Error en la conexión de red: {e}")
        return

    results = data.get('results', [])
    if not results:
        print("No se encontraron observaciones con esos filtros. Intentando búsqueda por texto...")
        # Plan alternativo si el ID fallara en tu región: buscar por texto directo
        params_alt = {'q': 'yellowfin tuna', 'per_page': 50, 'has[]': 'photos'}
        response = requests.get(url, params=params_alt, headers=headers)
        results = response.json().get('results', [])

    count = 0
    # 2. Bucle de descarga de archivos
    for obs in results:
        photos = obs.get('photos', [])
        if not photos:
            continue

        # iNaturalist por defecto da urls miniatura ("square"). Cambiamos a "medium" para la CNN
        photo_url = photos[0].get('url', '')
        if not photo_url:
            continue
        photo_url = photo_url.replace('square', 'medium')

        try:
            img_data = requests.get(photo_url, headers=headers).content
            file_name = f"Marlin Pez Vela_{count:03d}.jpg"
            full_path = os.path.join(folder_path, file_name)

            with open(full_path, 'wb') as f:
                f.write(img_data)

            print(f"-> Guardada exitosamente: {file_name}")
            count += 1

            # Una micro pausa de 200ms para respetar el servidor
            time.sleep(0.2)

        except Exception as e:
            print(f"Error al guardar la imagen número {count}: {e}")
            continue

    print(f"\n¡Listo! Descarga finalizada.")
    print(f"Imágenes totales en la carpeta '{folder_path}': {count}")


if __name__ == '__main__':
    descargar_atun_seguro()