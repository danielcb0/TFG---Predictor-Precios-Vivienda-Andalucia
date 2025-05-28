import http.client
import json
import csv
import time
import os

# Cargar JSON desde un archivo externo
with open('config.json', 'r') as file:
    config_data = json.load(file)

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
    total_properties_in_file = 0
    
    filename = os.path.join(output_directory, f"3{province_name}_SALE_{sort_order.upper()}.csv")
    print(f"Iniciando extracción para: {province_name} ({sort_order}). Archivo: {filename}")

    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Price', 'Property Type', 'Size (m2)', 'Number of Rooms', 'Number of Bathrooms', 'Latitude', 'Longitude', 'Location'])

        while total_properties_in_file < MAX_PROPERTIES_PER_API:
            retries = 0
            request_successful = False
            current_page_properties_data = None

            while retries < MAX_RETRIES:
                try:
                    params = f"/properties/list?numPage={num_page}&maxItems={MAX_ITEMS_PER_REQUEST}&locationId={province_id}&sort={sort_order}&locale=es&operation=sale&country=es"
                    conn.request("GET", params, headers=headers)
                    res = conn.getresponse()
                    response_body = res.read()

                    content_type = res.getheader('Content-Type')
                    if res.status != 200 or 'application/json' not in content_type:
                        error_message = f"Respuesta inesperada (Status: {res.status}, Content-Type: {content_type})."
                        print(f"{error_message} Body: {response_body[:200].decode('utf-8', 'ignore')}")
                        # Considerar ciertos códigos de estado (ej. 429, 401, 403) como fallo de API key más directamente
                        if res.status in [401, 403, 429]: # Unauthorized, Forbidden, Too Many Requests
                             retries = MAX_RETRIES # Forzar fallo para esta API key
                             break
                        raise ValueError(error_message)

                    current_page_properties_data = json.loads(response_body.decode("utf-8"))
                    request_successful = True
                    break 

                except (json.JSONDecodeError, ValueError) as e:
                    retries += 1
                    print(f"Error decodificando/validando JSON para {province_name}({sort_order}), pág {num_page} (Intento {retries}/{MAX_RETRIES}): {e}")
                    if retries < MAX_RETRIES: time.sleep(5)
                except http.client.HTTPException as e:
                    retries += 1
                    print(f"Error HTTP para {province_name}({sort_order}), pág {num_page} (Intento {retries}/{MAX_RETRIES}): {e}")
                    if retries < MAX_RETRIES: time.sleep(5)
                except Exception as e:
                    retries +=1
                    print(f"Error inesperado durante la solicitud para {province_name}({sort_order}), pág {num_page} (Intento {retries}/{MAX_RETRIES}): {e}")
                    if retries < MAX_RETRIES: time.sleep(5)


            if not request_successful:
                print(f"Fallaron todos los reintentos para obtener pág {num_page} de {province_name} ({sort_order}). API Key {api_key[:10]}... podría estar agotada.")
                return 'FAILURE_API_LIMIT_OR_ERROR'

            if 'elementList' not in current_page_properties_data or not current_page_properties_data['elementList']:
                if num_page == 1 and total_properties_in_file == 0:
                    print(f"No hay propiedades disponibles para {province_name} ({sort_order}) desde la primera página.")
                    return 'SUCCESS_NO_MORE_DATA'
                else:
                    print(f"No hay más propiedades en páginas subsiguientes para {province_name} ({sort_order}) después de {total_properties_in_file} propiedades.")
                    return 'SUCCESS_DATA_FETCHED'
            
            properties_on_page = 0
            for property_item in current_page_properties_data['elementList']:
                writer.writerow([
                    property_item.get('price', 'N/A'),
                    property_item.get('propertyType', 'N/A'),
                    property_item.get('size', 'N/A'),
                    property_item.get('rooms', 'N/A'),
                    property_item.get('bathrooms', 'N/A'),
                    property_item.get('latitude', 'N/A'),
                    property_item.get('longitude', 'N/A'),
                    property_item.get('address', 'N/A')
                ])
                total_properties_in_file += 1
                properties_on_page += 1
                if total_properties_in_file >= MAX_PROPERTIES_PER_API:
                    break
            
            print(f"Página {num_page}: {properties_on_page} propiedades guardadas para {province_name} ({sort_order}). Total acumulado: {total_properties_in_file}.")

            if total_properties_in_file >= MAX_PROPERTIES_PER_API:
                print(f"Se alcanzó el límite de {MAX_PROPERTIES_PER_API} propiedades para {province_name} ({sort_order}).")
                break 

            num_page += 1
            time.sleep(1)

    print(f"Finalizada la extracción para {province_name} ({sort_order}). Total guardado: {total_properties_in_file} propiedades en '{filename}'.")
    return 'SUCCESS_DATA_FETCHED'

# Lógica mejorada para iterar sobre claves API y provincias
api_keys_list = config_data["api_keys"]
provinces_dict = config_data["provinces"]
num_api_keys = len(api_keys_list)
current_api_key_idx = 0

tasks = []
for p_name, p_id in provinces_dict.items():
    for s_order in ["desc", "asc"]:
        tasks.append({
            "province_name": p_name,
            "province_id": p_id,
            "sort_order": s_order
        })

current_task_idx = 0
while current_task_idx < len(tasks):
    if current_api_key_idx >= num_api_keys:
        print("\nTodas las claves API disponibles han sido probadas y parecen agotadas o han fallado.")
        print(f"Tareas restantes ({len(tasks) - current_task_idx}) no se completarán.")
        break

    task = tasks[current_task_idx]
    api_key_to_use = api_keys_list[current_api_key_idx]

    print(f"\nIntentando tarea: {task['province_name']} (ID: {task['province_id']}), Orden: {task['sort_order'].upper()}")
    print(f"Usando API Key Index: {current_api_key_idx} (Clave: ...{api_key_to_use[-6:]})")
    
    status = get_properties(api_key_to_use, task['province_id'], task['sort_order'], task['province_name'])

    if status == 'SUCCESS_DATA_FETCHED' or status == 'SUCCESS_NO_MORE_DATA':
        print(f"Tarea para {task['province_name']} ({task['sort_order']}) manejada (Estado: {status}) con API Key Index {current_api_key_idx}.")
        current_task_idx += 1 
        # Rotar la API key para la siguiente tarea para distribuir la carga
        if num_api_keys > 0 : # Evitar división por cero si no hay API keys
             current_api_key_idx = (current_api_key_idx + 1) % num_api_keys
    elif status == 'FAILURE_API_LIMIT_OR_ERROR':
        print(f"Fallo (posiblemente límite/error de API) para {task['province_name']} ({task['sort_order']}) con API Key Index {current_api_key_idx}.")
        print("Intentando con la siguiente API key para la MISMA tarea.")
        current_api_key_idx += 1 
    else:
        print(f"Estado desconocido '{status}' recibido de get_properties para {task['province_name']} ({task['sort_order']}).")
        print("Tratando como fallo y probando la siguiente API key para la MISMA tarea.")
        current_api_key_idx += 1

print("\n--- Proceso de extracción finalizado ---")
if current_task_idx == len(tasks):
    print("Todas las tareas se han procesado exitosamente.")
else:
    print(f"{len(tasks) - current_task_idx} de {len(tasks)} tareas no se pudieron completar.")
