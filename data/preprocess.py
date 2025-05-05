#!/usr/bin/env python3
# 1_consolidate_raw.py
import os
import glob
import pandas as pd

def consolidate_raw_data(raw_dir='data/raw', processed_dir='data/processed'):
    """
    Carga todos los CSV *_SALE_*.csv de raw_dir, añade las columnas Province y SortOrder,
    concatena en un único DataFrame y guarda el resultado en processed_dir/andalucia_raw.csv.
    """
    # Asegurar que la carpeta de procesado existe
    os.makedirs(processed_dir, exist_ok=True)

    # Buscar ficheros
    pattern = os.path.join(raw_dir, '*_SALE_*.csv')
    csv_files = glob.glob(pattern)
    if not csv_files:
        print(f"No se encontraron archivos en {pattern}")
        return

    # Leer y concatenar
    df_list = []
    for filepath in csv_files:
        df = pd.read_csv(filepath)
        # Obtener metadata del nombre de archivo
        filename = os.path.basename(filepath)
        parts = filename.split('_')
        province   = parts[0]
        sort_order = parts[2].replace('.csv', '')
        # Añadir columnas
        df['Province']  = province
        df['SortOrder'] = sort_order.upper()
        df_list.append(df)

    # Concatenar todos los DataFrames
    df_all = pd.concat(df_list, ignore_index=True)
    print(f"Total filas consolidadas: {len(df_all)}")

    # Guardar CSV maestro
    output_path = os.path.join(processed_dir, 'andalucia_raw.csv')
    df_all.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"CSV consolidado guardado en: {output_path}")

if __name__ == '__main__':
    # Ajusta los paths si tus carpetas están en otra ubicación
    consolidate_raw_data(raw_dir='data/raw', processed_dir='data/processed')
