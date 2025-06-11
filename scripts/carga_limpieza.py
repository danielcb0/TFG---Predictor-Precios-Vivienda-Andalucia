"""
Carga y Limpieza de Datos de Viviendas en Andalucía

Proceso inicial de carga y limpieza de datos del conjunto de datos de viviendas en venta en Andalucía,
preparando los datos para análisis posteriores.

Autor: Daniel Carrera Bonilla
Trabajo Final de Grado
"""

# Cargar Librerías y Datos Crudos
# Importación de librerías necesarias para el procesamiento de datos y carga del dataset.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

# Configuración del entorno
plt.style.use('seaborn-v0_8-whitegrid')
pd.set_option('display.max_columns', None)

# Definir rutas de archivos
data_dir = '../data'
raw_data_dir = os.path.join(data_dir, 'processed')
clean_data_dir = os.path.join(data_dir, 'clean')

# Crear directorio para datos limpios si no existe
if not os.path.exists(clean_data_dir):
    os.makedirs(clean_data_dir)

# Cargar el dataset de Andalucía raw
try:
    file_path = os.path.join(raw_data_dir, 'andalucia_rawv2.csv')
    df_raw = pd.read_csv(file_path)
    print(f"Dataset cargado exitosamente desde: {file_path}")
    print(f"Dimensiones del dataset: {df_raw.shape}")
except FileNotFoundError:
    print(f"Error: No se pudo encontrar el archivo en {file_path}")
    df_raw = pd.DataFrame()  # Crear DataFrame vacío si no se carga el archivo

# Mostrar las primeras filas del DataFrame para inspección inicial
if not df_raw.empty:
    print("Primeras 5 filas del conjunto de datos:")
    print(df_raw.head())
else:
    print("El DataFrame está vacío porque no se pudo cargar el archivo.")

# Inspección Inicial de Datos Crudos
# Análisis preliminar del conjunto de datos: información general, estadísticas descriptivas,
# dimensiones y tipos de datos. Identificación de posibles problemas como inconsistencias
# en los tipos de datos o valores que requieren transformación.

# Información general del DataFrame: tipos de datos y valores no nulos
if not df_raw.empty:
    print("\nInformación general del DataFrame:")
    df_raw.info()
else:
    print("El DataFrame está vacío.")

# Estadísticas descriptivas
if not df_raw.empty:
    # Estadísticas descriptivas de variables numéricas
    print("\nEstadísticas descriptivas de variables numéricas:")
    print(df_raw.describe())
    
    # Estadísticas descriptivas de variables categóricas
    print("\nEstadísticas descriptivas de variables categóricas:")
    print(df_raw.describe(include=['object']))
else:
    print("El DataFrame está vacío.")

# Verificar los valores únicos en la columna de tipo de propiedad
if not df_raw.empty and 'Property Type' in df_raw.columns:
    property_types = df_raw['Property Type'].value_counts()
    print("\nTipos de propiedades y su frecuencia:")
    print(property_types)
    
    # Visualizar la distribución de tipos de propiedades
    plt.figure(figsize=(10, 6))
    sns.countplot(y=df_raw['Property Type'], order=property_types.index)
    plt.title('Distribución de Tipos de Propiedades')
    plt.xlabel('Cantidad')
    plt.ylabel('Tipo de Propiedad')
    plt.tight_layout()
    plt.show()
else:
    print("El DataFrame está vacío o no contiene la columna 'Property Type'.")

# Manejo de Valores Nulos y Duplicados
# Detección y tratamiento de valores nulos en el dataset mediante técnicas como imputación
# o eliminación según corresponda. Identificación y eliminación de registros duplicados
# para garantizar la calidad de los datos.

# Verificar valores nulos en cada columna
if not df_raw.empty:
    missing_values = df_raw.isnull().sum()
    missing_percentage = (df_raw.isnull().sum() / len(df_raw)) * 100
    
    missing_info = pd.DataFrame({
        'Valores Nulos': missing_values,
        'Porcentaje (%)': missing_percentage.round(2)
    })
    
    print("\nAnálisis de valores nulos por columna:")
    print(missing_info[missing_info['Valores Nulos'] > 0])
    
    if missing_info['Valores Nulos'].sum() == 0:
        print("No se encontraron valores nulos en el dataset.")
else:
    print("El DataFrame está vacío.")

# Verificar registros duplicados
df_clean = pd.DataFrame()  # Inicializar df_clean como DataFrame vacío
if not df_raw.empty:
    duplicates = df_raw.duplicated().sum()
    print(f"\nNúmero de registros duplicados: {duplicates}")
    
    if duplicates > 0:
        # Eliminar duplicados y crear una copia limpia
        df_clean = df_raw.drop_duplicates().reset_index(drop=True)
        print(f"Se eliminaron {duplicates} registros duplicados.")
        print(f"Dimensiones originales: {df_raw.shape}")
        print(f"Dimensiones después de eliminar duplicados: {df_clean.shape}")
    else:
        df_clean = df_raw.copy()
        print("No se encontraron registros duplicados.")
else:
    print("El DataFrame está vacío.")
    # df_clean ya está inicializado como DataFrame vacío

# Limpieza y Transformación de Columnas
# Normalización de los nombres de columnas, conversión de tipos de datos,
# estandarización de unidades y formatos, traducción de columnas al español,
# y otras transformaciones necesarias para preparar el dataset para análisis posteriores.

# Crear una copia del DataFrame para las transformaciones (si no se hizo al eliminar duplicados)
if not df_raw.empty and df_clean.empty: # Solo si df_clean no fue asignado antes
    df_clean = df_raw.copy()

if not df_clean.empty:
    print("\nTipos de datos antes de la transformación:")
    print(df_clean.dtypes)
else:
    print("El DataFrame df_clean está vacío.")

# Renombrar columnas al español
if not df_clean.empty:
    column_mapping = {
        'Price': 'precio',
        'Property Type': 'tipo_propiedad',
        'Size (m2)': 'superficie',
        'Number of Rooms': 'habitaciones',
        'Number of Bathrooms': 'baños',
        'Latitude': 'latitud',
        'Longitude': 'longitud',
        'Location': 'ubicacion'
    }
    df_clean = df_clean.rename(columns=column_mapping)
    print("\nColumnas renombradas al español:")
    print(df_clean.columns.tolist())
else:
    print("El DataFrame df_clean está vacío, no se pueden renombrar columnas.")

# Traducir y estandarizar valores en la columna tipo_propiedad
if not df_clean.empty and 'tipo_propiedad' in df_clean.columns:
    property_type_mapping = {
        'flat': 'piso',
        'chalet': 'chalet',
        'countryHouse': 'casa_rural',
        'duplex': 'duplex',
        'studio': 'estudio',
        'penthouse': 'atico'
    }
    df_clean['tipo_propiedad'] = df_clean['tipo_propiedad'].map(property_type_mapping).fillna(df_clean['tipo_propiedad'])
    print("\nValores únicos en la columna tipo_propiedad después de la traducción:")
    print(df_clean['tipo_propiedad'].value_counts())
else:
    print("El DataFrame df_clean está vacío o no contiene la columna 'tipo_propiedad'.")

# Verificar y tratar valores extremos o inconsistentes en la columna superficie
if not df_clean.empty and 'superficie' in df_clean.columns:
    print("\nResumen estadístico de superficie antes de la limpieza:")
    print(df_clean['superficie'].describe())
    
    large_properties = df_clean[df_clean['superficie'] > 10000]
    if not large_properties.empty:
        print(f"\nPropiedades con superficie extremadamente grande (>10,000 m²): {len(large_properties)}")
        print(large_properties[['precio', 'tipo_propiedad', 'superficie', 'habitaciones', 'ubicacion']])
        df_clean = df_clean[df_clean['superficie'] <= 10000]
        print(f"Se filtraron propiedades con superficie > 10,000 m²")
    
    invalid_surface = df_clean[(df_clean['superficie'] <= 0)]
    if not invalid_surface.empty:
        print(f"\nPropiedades con superficie inválida (<=0 m²): {len(invalid_surface)}")
        df_clean = df_clean[df_clean['superficie'] > 0]
        print(f"Se eliminaron propiedades con superficie inválida")
    
    print("\nResumen estadístico de superficie después de la limpieza:")
    print(df_clean['superficie'].describe())
else:
    print("El DataFrame df_clean está vacío o no contiene la columna 'superficie'.")

# Verificar y tratar valores extremos o inconsistentes en la columna precio
if not df_clean.empty and 'precio' in df_clean.columns:
    print("\nResumen estadístico de precio antes de la limpieza:")
    print(df_clean['precio'].describe())
    
    low_price = df_clean[df_clean['precio'] < 1000]
    if not low_price.empty:
        print(f"\nPropiedades con precio extremadamente bajo (<1,000 €): {len(low_price)}")
        print(low_price[['precio', 'tipo_propiedad', 'superficie', 'habitaciones', 'ubicacion']].head())
        
    high_price = df_clean[df_clean['precio'] > 1000000]
    if not high_price.empty:
        print(f"\nPropiedades con precio extremadamente alto (>1,000,000 €): {len(high_price)}")
        print(high_price[['precio', 'tipo_propiedad', 'superficie', 'habitaciones', 'ubicacion']].head())
    
    Q1 = df_clean['precio'].quantile(0.25)
    Q3 = df_clean['precio'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    print(f"\nLímite inferior para outliers (Q1 - 1.5*IQR): {lower_bound}")
    print(f"Límite superior para outliers (Q3 + 1.5*IQR): {upper_bound}")
    
    outliers = df_clean[(df_clean['precio'] < lower_bound) | (df_clean['precio'] > upper_bound)]
    print(f"\nNúmero de outliers según método IQR: {len(outliers)}")
    print("\nNo se eliminarán outliers de precio en esta etapa, se considerarán en el análisis exploratorio.")
else:
    print("El DataFrame df_clean está vacío o no contiene la columna 'precio'.")

# Crear variables derivadas útiles
if not df_clean.empty:
    if 'precio' in df_clean.columns and 'superficie' in df_clean.columns:
        # Asegurarse de que 'superficie' no tenga ceros para evitar DivisionByZeroError
        if not df_clean[df_clean['superficie'] == 0].empty:
            print("\nAdvertencia: Existen propiedades con superficie igual a 0. Se omitirán para el cálculo de precio_m2.")
            # Opcional: eliminar o imputar estas filas antes de calcular precio_m2
            # df_clean = df_clean[df_clean['superficie'] > 0] 
        
        # Calcular precio_m2 solo para filas con superficie > 0
        df_clean.loc[df_clean['superficie'] > 0, 'precio_m2'] = df_clean['precio'] / df_clean['superficie']
        print("\nSe creó la columna precio_m2 (precio por metro cuadrado)")
        
        if 'precio_m2' in df_clean.columns:
            print("\nEstadísticas de precio por metro cuadrado (€/m²):")
            print(df_clean['precio_m2'].describe())
            
            extreme_price_m2 = df_clean[(df_clean['precio_m2'] > 10000) | (df_clean['precio_m2'] < 10)]
            if not extreme_price_m2.empty:
                print(f"\nPropiedades con precio por m² extremo (<10 € o >10,000 €): {len(extreme_price_m2)}")
                df_clean = df_clean[(df_clean['precio_m2'] >= 10) & (df_clean['precio_m2'] <= 10000)]
                print(f"Se filtraron propiedades con precio por m² extremo")
    
    if 'habitaciones' in df_clean.columns and 'superficie' in df_clean.columns:
         # Calcular densidad_habitaciones solo para filas con superficie > 0
        df_clean.loc[df_clean['superficie'] > 0, 'densidad_habitaciones'] = df_clean['habitaciones'] / df_clean['superficie']
        print("\nSe creó la columna densidad_habitaciones (ratio habitaciones/superficie)")
    
    print("\nColumnas finales del DataFrame:")
    print(df_clean.columns.tolist())
    
    print("\nPrimeras filas del DataFrame después de las transformaciones:")
    print(df_clean.head())
else:
    print("El DataFrame df_clean está vacío.")

# Guardar Datos Limpios en CSV
# Almacenamiento del DataFrame procesado y limpio en un nuevo archivo CSV
# para su uso en análisis posteriores y modelado.
if not df_clean.empty:
    timestamp = datetime.now().strftime("%Y%m%d")
    clean_file_path = os.path.join(clean_data_dir, f'andalucia_clean_{timestamp}.csv')
    
    df_clean.to_csv(clean_file_path, index=False)
    print(f"\nDataset limpio guardado exitosamente en: {clean_file_path}")
    print(f"Dimensiones del dataset limpio: {df_clean.shape}")
    
    standard_clean_path = os.path.join(clean_data_dir, 'andalucia_clean.csv')
    df_clean.to_csv(standard_clean_path, index=False)
    print(f"Dataset limpio también guardado como: {standard_clean_path}")
else:
    print("\nEl DataFrame df_clean está vacío. No se puede guardar.")

print("\nProceso de carga y limpieza completado.")

