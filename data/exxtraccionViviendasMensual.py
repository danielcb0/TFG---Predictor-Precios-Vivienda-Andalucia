import http.client
import json
import csv
import time

# Cargar JSON desde un archivo externo
with open('config.json', 'r') as file:
    data = json.load(file)

# Parámetros generales
MAX_PROPERTIES_PER_API = 20000
MAX_ITEMS_PER_REQUEST = 40

def get_properties(api_key, province_id, sort_order, province_name):
    conn = http.client.HTTPSConnection("idealista2.p.rapidapi.com")
    headers = {
        'x-rapidapi-key': api_key,
        'x-rapidapi-host': "idealista2.p.rapidapi.com"
    }
    
    num_page = 1
    total_properties = 0
    
    # Crear archivo CSV para guardar datos
    filename = f"{province_name}_SALE_{sort_order.upper()}.csv"
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Price', 'Property Type', 'Size (m2)', 'Number of Rooms', 'Number of Bathrooms', 'Latitude', 'Longitude', 'Location'])

        while total_properties < MAX_PROPERTIES_PER_API:
            # Crear la solicitud para la página actual
            params = f"/properties/list?numPage={num_page}&maxItems={MAX_ITEMS_PER_REQUEST}&locationId={province_id}&sort={sort_order}&locale=es&operation=sale&country=es"
            conn.request("GET", params, headers=headers)
            res = conn.getresponse()
            data = res.read()
            
            # Decodificar la respuesta
            properties_data = json.loads(data.decode("utf-8"))
            
            if 'elementList' not in properties_data or not properties_data['elementList']:
                print(f"No hay más propiedades disponibles para {province_name} en orden {sort_order}.")
                break
            
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
            
            num_page += 1
            
            # Si se alcanzan o superan las propiedades máximas, detener la iteración
            if total_properties >= MAX_PROPERTIES_PER_API:
                break
            
            time.sleep(1)  # Para evitar ser bloqueados por demasiadas solicitudes
    
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
