import os
import pandas as pd
import glob

def merge_csv_files():
    # Definir las rutas correctamente
    raw_path = 'raw/'     # En lugar de 'data/raw/'
    processed_path = 'processed/'   # En lugar de 'data/processed/'
    
    # Crear la carpeta processed si no existe
    os.makedirs(processed_path, exist_ok=True)
    
    # Obtener todos los archivos CSV en la carpeta raw
    all_csv_files = glob.glob(os.path.join(raw_path, '*.csv'))
    
    if not all_csv_files:
        print("No se encontraron archivos CSV en la carpeta raw.")
        return
    
    print(f"Se encontraron {len(all_csv_files)} archivos CSV.")
    
    # Lista para almacenar todos los DataFrames
    all_dataframes = []
    
    # Leer cada archivo CSV y añadirlo a la lista
    for file in all_csv_files:
        try:
            # Leer el archivo CSV y añadirlo a la lista
            df = pd.read_csv(file)
            all_dataframes.append(df)
            print(f"Procesado: {os.path.basename(file)}")
        except Exception as e:
            print(f"Error al procesar {file}: {e}")
    
    if not all_dataframes:
        print("No se pudieron leer datos de ningún archivo CSV.")
        return
    
    # Concatenar todos los DataFrames
    merged_df = pd.concat(all_dataframes, ignore_index=True)
    
    # Guardar el DataFrame combinado en un nuevo archivo CSV
    output_file = os.path.join(processed_path, 'andalucia_raw.csv')
    merged_df.to_csv(output_file, index=False)
    
    print(f"Fusión completada. Se han combinado {len(all_dataframes)} archivos en: {output_file}")
    print(f"El archivo combinado contiene {len(merged_df)} filas.")

if __name__ == "__main__":
    merge_csv_files()