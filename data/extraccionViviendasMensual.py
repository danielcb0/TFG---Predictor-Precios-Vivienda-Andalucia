import http.client
import json
import csv
import time
import os

# Cargar JSON desde un archivo externo
with open('config.json', 'r') as file:
    data = json.load(file)

# Parámetros generales
MAX_PROPERTIES_PER_API = 20000
MAX_ITEMS_PER_REQUEST = 40
MAX_RETRIES = 5

# Directorio donde se guardarán los archivos CSV
output_directory = 'raw/'

# Verificar si la carpeta `raw` existe, si no crearla
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

def get_properties(api_key, province_id, sort_order, province_name):
    conn = http.client.HTTPSConnection("idealista2.p.rapidapi.com")
    headers = {
        'x-rapidapi-key': api_key,
        'x-rapidapi-host': "idealista2.p.rapidapi.com"
    }
    
    num_page = 1
    total_properties = 0
    
    # Crear archivo CSV para guardar datos en la carpeta `raw`
    filename = os.path.join(output_directory, f"{province_name}_SALE_{sort_order.upper()}.csv")
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Price', 'Property Type', 'Size (m2)', 'Number of Rooms', 'Number of Bathrooms', 'Latitude', 'Longitude', 'Location'])

        while total_properties < MAX_PROPERTIES_PER_API:
            retries = 0
            while retries < MAX_RETRIES:
                try:
                    # Crear la solicitud para la página actual
                    params = f"/properties/list?numPage={num_page}&maxItems={MAX_ITEMS_PER_REQUEST}&locationId={province_id}&sort={sort_order}&locale=es&operation=sale&country=es"
                    conn.request("GET", params, headers=headers)
                    res = conn.getresponse()
                    data = res.read()

                    # Verificar si el tipo de contenido es JSON
                    content_type = res.getheader('Content-Type')
                    if 'application/json' not in content_type:
                        raise ValueError(f"Respuesta no JSON recibida: {content_type}")

                    # Decodificar la respuesta
                    properties_data = json.loads(data.decode("utf-8"))

                    # Comprobar si la respuesta contiene propiedades
                    if 'elementList' not in properties_data or not properties_data['elementList']:
                        print(f"No hay más propiedades disponibles para {province_name} en orden {sort_order}.")
                        return
                    
                    # Escribir los datos de cada propiedad en el CSV
                    for property in properties_data['elementList']:
                        writer.writerow([
                            property.get('price', 'N/A'),
                            property.get('propertyType', 'N/A'),
                            property.get('size', 'N/A'),
                            property.get('rooms', 'N/A'),
                            property.get('bathrooms', 'N/A'),
                            property.get('latitude', 'N/A'),
                            property.get('longitude', 'N/A'),
                            property.get('address', 'N/A')
                        ])
                        total_properties += 1
                    
                    # Salir del bucle de reintento si la solicitud fue exitosa
                    num_page += 1
                    time.sleep(1)  # Para evitar ser bloqueados por demasiadas solicitudes
                    break

                except (json.JSONDecodeError, ValueError) as e:
                    retries += 1
                    print(f"Error al decodificar la respuesta o respuesta inesperada (Intento {retries}/{MAX_RETRIES}): {e}")
                    print(f"Esperando antes de reintentar...")
                    time.sleep(5)  # Esperar antes de intentar de nuevo

                except http.client.HTTPException as e:
                    retries += 1
                    print(f"Error en la solicitud HTTP (Intento {retries}/{MAX_RETRIES}): {e}")
                    print(f"Esperando antes de reintentar...")
                    time.sleep(5)  # Esperar antes de intentar de nuevo

                # Si se alcanza el máximo de reintentos, salir del bucle
                if retries == MAX_RETRIES:
                    print("Número máximo de reintentos alcanzado. Saltando a la siguiente solicitud.")
                    return

            # Si se alcanzan o superan las propiedades máximas, detener la iteración
            if total_properties >= MAX_PROPERTIES_PER_API:
                break

    print(f"Se han guardado un total de {total_properties} propiedades en '{filename}'.")

# Lógica para iterar sobre claves API y provincias
current_api_index = 0

for province_name, province_id in data["provinces"].items():
    for sort_order in ["asc", "desc"]:
        if current_api_index >= len(data["api_keys"]):
            print("No hay más claves API disponibles.")
            break

        current_api_key = data["api_keys"][current_api_index]
        get_properties(current_api_key, province_id, sort_order, province_name)
        
        # Cambiar a la siguiente clave API si se agotó la actual
        current_api_index += 1
