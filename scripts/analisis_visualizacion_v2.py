"""
Análisis Avanzado y Visualización de Datos de Viviendas en Andalucía: Enfoque Provincial

Este script se enfoca en el análisis exploratorio avanzado y la visualización de datos
del conjunto de viviendas en Andalucía, con un énfasis particular en el análisis
comparativo entre sus 8 provincias.

Autor: Daniel Carrera Bonilla
Trabajo Final de Grado

Objetivos Principales:
1.  Analizar distribuciones y outliers de variables numéricas clave.
2.  Explorar variables categóricas y su impacto en el precio.
3.  Investigar correlaciones y relaciones multivariadas.
4.  Realizar análisis geoespacial general de precios y características.
5.  Ingeniería de la característica 'provincia' y análisis de distribución de propiedades.
6.  Realizar análisis descriptivo de variables clave por provincia.
7.  Comparar precios (`precio` y `precio_m2`) entre las 8 provincias.
8.  Analizar la distribución de tipos de propiedad por provincia y su relación con el precio.
9.  Visualizar densidades de propiedades y precios medios con un enfoque provincial.

Dataset:
Se utilizará un dataset limpio previamente procesado (ej. 'andalucia_clean.csv').
"""

# Importación de librerías necesarias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis
import os

# Configuración de visualizaciones
# %matplotlib inline # No necesario en script, plt.show() se usa explícitamente
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x) # Formato para floats

# Definir rutas de archivos
# Se asume que el script está en 'scripts' y los datos en 'data/clean'
data_dir = '../data/clean'
# Intentar cargar 'andalucia_clean.csv' como el archivo estándar limpio
file_name = 'andalucia_clean.csv' 
# Si se quiere usar un archivo con fecha específica, cambiar file_name aquí:
# file_name = 'andalucia_clean_20250516.csv' # Ejemplo del notebook
file_path = os.path.join(data_dir, file_name)

# Cargar el dataset
try:
    df = pd.read_csv(file_path)
    print(f"Dataset cargado exitosamente desde: {file_path}")
    print(f"Dimensiones del dataset: {df.shape}")
except FileNotFoundError:
    print(f"Error: No se pudo encontrar el archivo en {file_path}")
    print(f"Asegúrate de que la ruta '{data_dir}' y el archivo '{file_name}' son correctos.")
    df = pd.DataFrame() # DataFrame vacío para evitar errores

# Mostrar primeras filas e información básica
if not df.empty:
    print("\nPrimeras 5 filas del dataset:")
    print(df.head()) # Reemplazar display(df.head())
    print("\nInformación general del DataFrame:")
    df.info()
    print("\nEstadísticas descriptivas básicas:")
    print(df.describe()) # Reemplazar display(df.describe())
else:
    print("El DataFrame está vacío.")

# --- 1. Análisis de Distribuciones y Outliers en Variables Numéricas Clave ---
# Analizaremos las distribuciones de las variables numéricas más importantes:
# `precio`, `superficie`, `precio_m2`, `habitaciones` y `baños`.
# Utilizaremos histogramas y boxplots para visualizar su forma, identificar outliers
# y calcular coeficientes de asimetría y curtosis. También exploraremos el efecto
# de transformaciones logarítmicas en variables con alta asimetría.

df_analysis = pd.DataFrame() # Inicializar para evitar errores si df está vacío

if not df.empty:
    numerical_cols = ['precio', 'superficie', 'precio_m2', 'habitaciones', 'baños']
    
    # Eliminar filas con NaN en estas columnas específicas para el análisis de distribución
    # Usar .copy() para evitar SettingWithCopyWarning
    df_analysis = df.dropna(subset=numerical_cols).copy() 

    print("\nAnálisis de Distribuciones de Variables Numéricas Clave")
    print("=======================================================")

    for col in numerical_cols:
        if col in df_analysis.columns:
            print(f"\n--- Análisis de '{col}' ---")
            
            # Estadísticas
            print(f"Media: {df_analysis[col].mean():.2f}")
            print(f"Mediana: {df_analysis[col].median():.2f}")
            print(f"Desviación Estándar: {df_analysis[col].std():.2f}")
            print(f"Asimetría (Skewness): {skew(df_analysis[col]):.2f}")
            print(f"Curtosis (Kurtosis): {kurtosis(df_analysis[col]):.2f}")
            
            # Histogramas
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            sns.histplot(df_analysis[col], kde=True, bins=50)
            plt.title(f'Histograma de {col}')
            plt.xlabel(col)
            plt.ylabel('Frecuencia')
            
            # Boxplots
            plt.subplot(1, 2, 2)
            sns.boxplot(y=df_analysis[col])
            plt.title(f'Boxplot de {col}')
            plt.ylabel(col)
            
            plt.tight_layout()
            plt.show()

            # Considerar transformación logarítmica para precio y precio_m2 si son muy asimétricas
            if col in ['precio', 'precio_m2'] and skew(df_analysis[col]) > 1:
                # Asegurarse de que no haya valores <= 0 antes de aplicar log
                if (df_analysis[col] > 0).all():
                    df_analysis[f'{col}_log'] = np.log(df_analysis[col])
                    print(f"\nAplicando transformación logarítmica a '{col}'")
                    print(f"Asimetría de '{col}_log': {skew(df_analysis[f'{col}_log']):.2f}")
                    print(f"Curtosis de '{col}_log': {kurtosis(df_analysis[f'{col}_log']):.2f}")
                    
                    plt.figure(figsize=(6, 4))
                    sns.histplot(df_analysis[f'{col}_log'], kde=True, bins=50)
                    plt.title(f'Histograma de {col}_log')
                    plt.xlabel(f'Log({col})')
                    plt.ylabel('Frecuencia')
                    plt.show()
                else:
                    print(f"No se puede aplicar transformación logarítmica a '{col}' debido a valores no positivos.")
        else:
            print(f"La columna '{col}' no existe en el DataFrame.")
else:
    print("El DataFrame está vacío. No se puede realizar el análisis de distribuciones.")

# Discusión de Distribuciones y Outliers:
# *   Precio: Generalmente muestra una fuerte asimetría positiva. La transformación logarítmica suele ayudar a normalizar esta distribución.
# *   Superficie: Similar al precio, tiende a ser asimétrica a la derecha.
# *   Precio_m2: También puede ser asimétrica. Su transformación logarítmica puede ser útil.
# *   Habitaciones y Baños: Variables discretas. Sus histogramas muestran la frecuencia de cada número.

# --- 2. Exploración de Variables Categóricas y su Impacto en el Precio ---
# Analizaremos la variable categórica `tipo_propiedad` y su relación con el `precio` (y `precio_m2`).
# También exploraremos la variable `ubicacion` para entender su estructura antes del análisis provincial detallado.

if not df.empty:
    print("\nAnálisis de Variables Categóricas")
    print("===================================")

    # Análisis de 'tipo_propiedad'
    if 'tipo_propiedad' in df.columns:
        print("\n--- Análisis de 'tipo_propiedad' ---")
        
        # Distribución de tipo_propiedad
        plt.figure(figsize=(10, 6))
        sns.countplot(data=df, y='tipo_propiedad', order=df['tipo_propiedad'].value_counts().index, palette='viridis')
        plt.title('Distribución de Tipos de Propiedad')
        plt.xlabel('Cantidad')
        plt.ylabel('Tipo de Propiedad')
        plt.show()
        
        print("\nFrecuencia de Tipos de Propiedad (%):")
        print(df['tipo_propiedad'].value_counts(normalize=True) * 100)

        # Relación entre tipo_propiedad y precio
        if 'precio' in df.columns:
            plt.figure(figsize=(12, 7))
            # Ordenar por precio mediano para mejor visualización
            order_tp = df.groupby('tipo_propiedad')['precio'].median().sort_values().index
            sns.boxplot(data=df, x='precio', y='tipo_propiedad', order=order_tp, palette='viridis')
            plt.title('Relación entre Tipo de Propiedad y Precio')
            plt.xlabel('Precio')
            plt.ylabel('Tipo de Propiedad')
            plt.xscale('log') # Usar escala logarítmica para el precio
            plt.show()

        # Relación entre tipo_propiedad y precio_m2
        if 'precio_m2' in df.columns:
            plt.figure(figsize=(12, 7))
            # Ordenar por precio_m2 mediano
            order_tpm2 = df.groupby('tipo_propiedad')['precio_m2'].median().sort_values().index
            sns.boxplot(data=df, x='precio_m2', y='tipo_propiedad', order=order_tpm2, palette='viridis')
            plt.title('Relación entre Tipo de Propiedad y Precio por m²')
            plt.xlabel('Precio por m²')
            plt.ylabel('Tipo de Propiedad')
            # plt.xscale('log') # Opcional para precio_m2
            plt.show()
    else:
        print("La columna 'tipo_propiedad' no existe.")

    # Análisis preliminar de 'ubicacion'
    if 'ubicacion' in df.columns:
        print("\n--- Análisis preliminar de 'ubicacion' ---")
        num_unique_locations = df['ubicacion'].nunique()
        print(f"Número de ubicaciones únicas: {num_unique_locations}")

        if num_unique_locations > 1:
            top_n = 10 
            print(f"\nTop {top_n} ubicaciones más frecuentes:")
            top_locations = df['ubicacion'].value_counts().nlargest(top_n)
            print(top_locations)
            # No se graficará aquí, ya que el análisis provincial será más detallado.
            # Se mostrará cómo se deriva 'provincia' de 'ubicacion' más adelante.
        else:
            print("La columna 'ubicacion' tiene un solo valor único o está vacía.")
            
    else:
        print("La columna 'ubicacion' no existe.")
else:
    print("El DataFrame está vacío. No se puede realizar el análisis de variables categóricas.")

# Discusión de Variables Categóricas:
# *   Tipo de Propiedad: Los diagramas revelan los tipos más comunes y cómo varían los precios entre ellos.
# *   Ubicación: Esta variable tiene alta cardinalidad. En secciones posteriores, intentaremos extraer la provincia para un análisis más estructurado.

# --- 3. Análisis de Correlación Multivariada ---
# Calcularemos y visualizaremos la matriz de correlación para todas las variables numéricas
# para entender las relaciones lineales entre ellas. Usaremos `pairplots` para un subconjunto de variables clave.

if not df.empty:
    print("\nAnálisis de Correlación Multivariada")
    print("======================================")
    
    # Seleccionar solo columnas numéricas para la matriz de correlación
    numerical_features_for_corr = df.select_dtypes(include=np.number).columns.tolist()
    
    df_corr = df.copy()
    # Añadir columnas logarítmicas si existen en df_analysis (creado en la sección 1)
    # y no están ya en df_corr (lo cual no deberían estar si df_corr es copia de df original)
    if not df_analysis.empty: # Asegurarse que df_analysis fue creado
        if 'precio_log' in df_analysis.columns and 'precio_log' not in df_corr.columns:
            df_corr = df_corr.join(df_analysis['precio_log'])
        if 'precio_m2_log' in df_analysis.columns and 'precio_m2_log' not in df_corr.columns:
            df_corr = df_corr.join(df_analysis['precio_m2_log'])
            
    # Filtrar solo las columnas numéricas que realmente existen en df_corr
    # y que no sean todo NaN después de un posible join.
    numerical_features_for_corr_final = [
        col for col in df_corr.select_dtypes(include=np.number).columns 
        if df_corr[col].notna().any()
    ]

    if numerical_features_for_corr_final:
        correlation_matrix = df_corr[numerical_features_for_corr_final].corr()
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
        plt.title('Matriz de Correlación de Variables Numéricas')
        plt.show()

        print("\nCorrelaciones más altas con 'precio':")
        if 'precio' in correlation_matrix:
            print(correlation_matrix['precio'].sort_values(ascending=False))
        elif 'precio_log' in correlation_matrix: # Si precio no está, pero precio_log sí
            print("\nCorrelaciones más altas con 'precio_log':")
            print(correlation_matrix['precio_log'].sort_values(ascending=False))

        # Pairplots para un subconjunto de variables clave
        pairplot_cols_base = ['precio', 'superficie', 'habitaciones', 'baños', 'precio_m2']
        pairplot_cols = [col for col in pairplot_cols_base if col in df_corr.columns and df_corr[col].notna().any()]

        if 'precio_log' in df_corr.columns and df_corr['precio_log'].notna().any():
             if 'precio' in pairplot_cols: pairplot_cols.remove('precio') 
             pairplot_cols.append('precio_log')
        
        if len(pairplot_cols) > 1:
            print(f"\nGenerando Pairplot para: {pairplot_cols}")
            # Tomar una muestra si el dataset es muy grande
            sample_df_corr = df_corr[pairplot_cols].sample(n=min(1000, len(df_corr)), random_state=42) if len(df_corr) > 1000 else df_corr[pairplot_cols]
            
            sns.pairplot(sample_df_corr.dropna(), kind='reg', plot_kws={'line_kws':{'color':'red', 'lw':1}, 'scatter_kws': {'alpha': 0.3, 's': 10}}, corner=True)
            plt.suptitle('Pairplot de Variables Clave', y=1.02)
            plt.show()
        else:
            print("No hay suficientes columnas válidas para generar el pairplot.")
            
    else:
        print("No se encontraron columnas numéricas válidas para el análisis de correlación.")
else:
    print("El DataFrame está vacío. No se puede realizar el análisis de correlación.")

# Discusión de Correlaciones y Pairplots:
# *   Matriz de Correlación: Muestra relaciones lineales. `precio` suele correlacionarse positivamente con `superficie`, `habitaciones`, `baños`.
# *   Pairplots: Permiten visualizar relaciones bivariadas (lineales y no lineales) y distribuciones individuales.

# --- 4. Análisis Geoespacial General de Precios y Características ---
# Crearemos gráficos de dispersión usando `latitud` y `longitud` para visualizar
# la distribución geográfica de las propiedades en Andalucía. Colorearemos los puntos
# por `precio` y `precio_m2`, y también podríamos usar el tamaño de los puntos para representar `superficie`.

if not df.empty and 'latitud' in df.columns and 'longitud' in df.columns:
    print("\nAnálisis Geoespacial General")
    print("=============================")
    
    # Crear una copia para el análisis geoespacial, eliminando NaNs en coordenadas y variables de interés
    df_geo = df.dropna(subset=['latitud', 'longitud', 'precio', 'precio_m2', 'superficie']).copy()

    if not df_geo.empty:
        # Scatter plot: latitud vs longitud, coloreado por precio (log)
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(df_geo['longitud'], df_geo['latitud'], 
                              c=np.log1p(df_geo['precio']), # Usar log de precio para mejor visualización
                              cmap='viridis', alpha=0.6, s=10)
        plt.colorbar(scatter, label='Log(Precio + 1)')
        plt.title('Distribución Geográfica de Propiedades por Precio')
        plt.xlabel('Longitud')
        plt.ylabel('Latitud')
        plt.grid(True)
        plt.show()

        # Scatter plot: latitud vs longitud, coloreado por precio_m2 (log)
        plt.figure(figsize=(10, 8))
        scatter_pm2 = plt.scatter(df_geo['longitud'], df_geo['latitud'], 
                                  c=np.log1p(df_geo['precio_m2']), # Usar log de precio_m2
                                  cmap='plasma', alpha=0.6, s=10)
        plt.colorbar(scatter_pm2, label='Log(Precio por m² + 1)')
        plt.title('Distribución Geográfica de Propiedades por Precio por m²')
        plt.xlabel('Longitud')
        plt.ylabel('Latitud')
        plt.grid(True)
        plt.show()

        # Scatter plot: latitud vs longitud, coloreado por precio, tamaño por superficie
        # Limitar el tamaño máximo para mejor visualización
        max_superficie_display = df_geo['superficie'].quantile(0.95) 
        sizes = (df_geo['superficie'] / max_superficie_display) * 100 
        sizes = np.clip(sizes, 5, 100) # Limitar tamaño mínimo y máximo de los puntos

        plt.figure(figsize=(12, 10))
        scatter_size = plt.scatter(df_geo['longitud'], df_geo['latitud'], 
                                   c=np.log1p(df_geo['precio']), 
                                   cmap='viridis', alpha=0.6, s=sizes)
        plt.colorbar(scatter_size, label='Log(Precio + 1)')
        plt.title('Distribución Geográfica: Precio (color) y Superficie (tamaño)')
        plt.xlabel('Longitud')
        plt.ylabel('Latitud')
        plt.grid(True)
        plt.show()
        
    else:
        print("No hay suficientes datos después de eliminar NaNs para el análisis geoespacial.")
        
else:
    print("El DataFrame está vacío o no contiene columnas de latitud/longitud.")

# Discusión del Análisis Geoespacial General:
# *   Estos gráficos ayudan a visualizar concentraciones de propiedades y variaciones de precios a nivel regional en Andalucía.
# *   Se pueden identificar "puntos calientes" (zonas caras) y "puntos fríos" (zonas baratas) de forma general.

# --- 5. Ingeniería de la Característica 'Provincia' y Conteo de Propiedades ---
# Crearemos una nueva columna `provincia` en el DataFrame.
# El objetivo es asignar cada propiedad a una de las 8 provincias andaluzas.
# Se utilizará un método simplificado basado en la búsqueda de nombres de ciudades clave o provincias en `ubicacion`.

if not df.empty:
    print("\nIngeniería de la Característica 'Provincia'")
    print("===========================================")

    provincias_andalucia = ['Almería', 'Cádiz', 'Córdoba', 'Granada', 'Huelva', 'Jaén', 'Málaga', 'Sevilla']

    def extraer_provincia(ubicacion_str):
        if not isinstance(ubicacion_str, str):
            return 'Desconocida'
        
        ubicacion_lower = ubicacion_str.lower()
        
        map_terminos_provincia = {
            'almería': 'Almería', 'almeria': 'Almería', 'roquetas de mar': 'Almería', 'el ejido': 'Almería',
            'cádiz': 'Cádiz', 'cadiz': 'Cádiz', 'jerez': 'Cádiz', 'algeciras': 'Cádiz', 'san fernando': 'Cádiz', 'el puerto de santa maría': 'Cádiz',
            'córdoba': 'Córdoba', 'cordoba': 'Córdoba', 'lucena': 'Córdoba',
            'granada': 'Granada', 'motril': 'Granada',
            'huelva': 'Huelva', 'lepe': 'Huelva',
            'jaén': 'Jaén', 'jaen': 'Jaén', 'linares': 'Jaén', 'úbeda': 'Jaén', # Úbeda es Jaén
            'málaga': 'Málaga', 'malaga': 'Málaga', 'marbella': 'Málaga', 'fuengirola': 'Málaga', 'torremolinos': 'Málaga', 'estepona': 'Málaga', 'benalmádena': 'Málaga',
            'sevilla': 'Sevilla', 'dos hermanas': 'Sevilla', 'alcalá de guadaíra': 'Sevilla', 'utrera': 'Sevilla'
        }
        
        for prov in provincias_andalucia:
            if prov.lower() in ubicacion_lower:
                return prov
        
        for termino, provincia_map in map_terminos_provincia.items():
            if termino in ubicacion_lower:
                return provincia_map
        
        return 'Desconocida'

    if 'ubicacion' in df.columns:
        df_prov = df.copy() # Usar df_prov para no modificar df globalmente aún
        df_prov['provincia'] = df_prov['ubicacion'].apply(extraer_provincia)
        
        print("\nConteo de propiedades por provincia extraída (incluyendo 'Desconocida'):")
        print(df_prov['provincia'].value_counts())

        conteo_provincias_identificadas = df_prov[df_prov['provincia'] != 'Desconocida']['provincia'].value_counts()

        if not conteo_provincias_identificadas.empty:
            plt.figure(figsize=(12, 7))
            sns.barplot(x=conteo_provincias_identificadas.index, y=conteo_provincias_identificadas.values, palette='Set2')
            plt.title('Número de Propiedades por Provincia Identificada en Andalucía')
            plt.xlabel('Provincia')
            plt.ylabel('Número de Propiedades')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.show()
            
            # Actualizamos el df principal para que las siguientes celdas usen la columna 'provincia'
            df['provincia'] = df_prov['provincia']
        else:
            print("No se pudieron identificar provincias a partir de la columna 'ubicacion' con el método actual.")
            if 'provincia' not in df.columns:
                 df['provincia'] = 'Desconocida'

        num_desconocidas = df_prov[df_prov['provincia'] == 'Desconocida'].shape[0]
        if num_desconocidas > 0:
            print(f"\nAdvertencia: {num_desconocidas} propiedades no pudieron ser asignadas a una provincia (etiquetadas como 'Desconocida').")
            print("Esto puede afectar la representatividad del análisis provincial.")
            print("Ejemplos de 'ubicacion' no mapeadas:")
            print(df_prov[df_prov['provincia'] == 'Desconocida']['ubicacion'].value_counts().head()) # Reemplazar display()
    else:
        print("La columna 'ubicacion' no existe, no se puede extraer la provincia.")
        df['provincia'] = 'No disponible' 

    # Comentario sobre geocodificación (como en el notebook)
    # if 'latitud' in df.columns and 'longitud' in df.columns:
    #     print("\nConsiderar geocodificación inversa para mejorar la asignación de provincias.")
else:
    print("El DataFrame está vacío. No se puede realizar la ingeniería de 'provincia'.")

# --- 6. Análisis Descriptivo de Variables Clave por Provincia ---
# Calcularemos estadísticas descriptivas para variables numéricas clave, agrupadas por `provincia`.

if not df.empty and 'provincia' in df.columns and df[~df['provincia'].isin(['Desconocida', 'No disponible'])]['provincia'].nunique() > 0:
    print("\nAnálisis Descriptivo de Variables Clave por Provincia")
    print("=====================================================")
    
    df_analisis_prov = df[~df['provincia'].isin(['Desconocida', 'No disponible'])].copy()
    
    if not df_analisis_prov.empty:
        variables_descriptivas = ['precio', 'superficie', 'habitaciones', 'baños', 'precio_m2']
        variables_existentes = [var for var in variables_descriptivas if var in df_analisis_prov.columns]
        
        if variables_existentes:
            print(f"Analizando para las variables: {variables_existentes}\n")
            
            stats_por_provincia = df_analisis_prov.groupby('provincia')[variables_existentes].agg(
                ['mean', 'median', 'std', 'min', 'max', 'count']
            )
            
            for var in variables_existentes:
                print(f"--- Estadísticas para '{var}' por Provincia ---")
                print(stats_por_provincia[var].sort_values(by='median', ascending=False)) # Reemplazar display()
                print("\n") # Separador
        else:
            print("Ninguna de las variables clave para análisis descriptivo se encuentra en el DataFrame.")
    else:
        print("No hay datos suficientes con provincias identificadas para realizar el análisis descriptivo.")
else:
    print("El DataFrame está vacío, la columna 'provincia' no existe, o no hay provincias identificadas.")

# --- 7. Análisis Comparativo de Precios (`precio` y `precio_m2`) por Provincia ---
# Generaremos boxplots y/o violin plots para comparar visualmente las distribuciones de precios.

if not df.empty and 'provincia' in df.columns and df[~df['provincia'].isin(['Desconocida', 'No disponible'])]['provincia'].nunique() > 0:
    print("\nAnálisis Comparativo de Precios por Provincia")
    print("=============================================")
    
    df_analisis_prov = df[~df['provincia'].isin(['Desconocida', 'No disponible'])].copy()

    if not df_analisis_prov.empty and 'precio' in df_analisis_prov.columns:
        order_provincias_precio = df_analisis_prov.groupby('provincia')['precio'].median().sort_values().index

        plt.figure(figsize=(14, 8))
        sns.boxplot(data=df_analisis_prov, x='precio', y='provincia', order=order_provincias_precio, palette='coolwarm')
        plt.title('Distribución de Precios por Provincia')
        plt.xlabel('Precio (€)')
        plt.ylabel('Provincia')
        plt.xscale('log')
        plt.grid(True, axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

        median_precio_prov = df_analisis_prov.groupby('provincia')['precio'].median().sort_values(ascending=False)
        plt.figure(figsize=(12, 7))
        sns.barplot(x=median_precio_prov.index, y=median_precio_prov.values, palette='coolwarm_r', order=median_precio_prov.index)
        plt.title('Precio Mediano por Provincia')
        plt.xlabel('Provincia')
        plt.ylabel('Precio Mediano (€)')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()
    else:
        print("No hay datos de 'precio' o provincias identificadas para el análisis de precios.")

    if not df_analisis_prov.empty and 'precio_m2' in df_analisis_prov.columns:
        order_provincias_precio_m2 = df_analisis_prov.groupby('provincia')['precio_m2'].median().sort_values().index
        
        plt.figure(figsize=(14, 8))
        sns.boxplot(data=df_analisis_prov, x='precio_m2', y='provincia', order=order_provincias_precio_m2, palette='viridis_r')
        plt.title('Distribución de Precio por m² por Provincia')
        plt.xlabel('Precio por m² (€/m²)')
        plt.ylabel('Provincia')
        plt.grid(True, axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

        median_precio_m2_prov = df_analisis_prov.groupby('provincia')['precio_m2'].median().sort_values(ascending=False)
        plt.figure(figsize=(12, 7))
        sns.barplot(x=median_precio_m2_prov.index, y=median_precio_m2_prov.values, palette='viridis', order=median_precio_m2_prov.index)
        plt.title('Precio Mediano por m² por Provincia')
        plt.xlabel('Provincia')
        plt.ylabel('Precio Mediano por m² (€/m²)')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()
    else:
        print("No hay datos de 'precio_m2' o provincias identificadas para el análisis de precio/m2.")
else:
    print("El DataFrame está vacío, 'provincia' no existe, o no hay provincias identificadas.")

# --- 8. Distribución de Tipos de Propiedad por Provincia y su Relación con el Precio ---
# Visualizaremos cómo se distribuyen los `tipo_propiedad` dentro de cada provincia.

if not df.empty and 'provincia' in df.columns and 'tipo_propiedad' in df.columns and df[~df['provincia'].isin(['Desconocida', 'No disponible'])]['provincia'].nunique() > 0:
    print("\nDistribución de Tipos de Propiedad por Provincia y su Relación con el Precio")
    print("============================================================================")
    
    df_analisis_prov = df[~df['provincia'].isin(['Desconocida', 'No disponible'])].copy()

    if not df_analisis_prov.empty:
        conteo_tipo_prov = df_analisis_prov.groupby(['provincia', 'tipo_propiedad']).size().unstack(fill_value=0)
        
        conteo_tipo_prov.plot(kind='bar', stacked=True, figsize=(15, 8), colormap='tab20')
        plt.title('Distribución de Tipos de Propiedad por Provincia')
        plt.xlabel('Provincia')
        plt.ylabel('Número de Propiedades')
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Tipo de Propiedad', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()

        conteo_tipo_prov_pct = conteo_tipo_prov.apply(lambda x: x / x.sum() * 100, axis=1)
        conteo_tipo_prov_pct.plot(kind='bar', stacked=True, figsize=(15, 8), colormap='tab20')
        plt.title('Distribución Porcentual de Tipos de Propiedad por Provincia')
        plt.xlabel('Provincia')
        plt.ylabel('Porcentaje de Propiedades (%)')
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Tipo de Propiedad', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()

        if 'precio' in df_analisis_prov.columns:
            top_n_tipos = df_analisis_prov['tipo_propiedad'].value_counts().nlargest(5).index
            df_top_tipos = df_analisis_prov[df_analisis_prov['tipo_propiedad'].isin(top_n_tipos)]

            median_precio_tipo_prov = df_top_tipos.groupby(['provincia', 'tipo_propiedad'])['precio'].median().unstack()
            
            if not median_precio_tipo_prov.empty:
                median_precio_tipo_prov.plot(kind='bar', figsize=(18, 9), colormap='Spectral', width=0.8)
                plt.title(f'Precio Mediano por Tipo de Propiedad (Top {len(top_n_tipos)}) y Provincia')
                plt.xlabel('Provincia')
                plt.ylabel('Precio Mediano (€)')
                plt.xticks(rotation=45, ha='right')
                plt.legend(title='Tipo de Propiedad')
                plt.yscale('log')
                plt.grid(True, axis='y', linestyle='--', alpha=0.7)
                plt.tight_layout()
                plt.show()
            else:
                print("No se pudo calcular el precio mediano por tipo de propiedad y provincia.")
        else:
            print("Columna 'precio' no disponible para análisis de precios por tipo y provincia.")
    else:
        print("No hay datos suficientes con provincias y tipos de propiedad identificados.")
else:
    print("El DataFrame está vacío o faltan 'provincia' o 'tipo_propiedad', o no hay provincias identificadas.")

# --- 9. Visualización de Densidad de Propiedades y Precios Medios con Enfoque Provincial ---
# Utilizaremos mapas de densidad (KDE plots) o hexbin plots.

if not df.empty and 'provincia' in df.columns and 'latitud' in df.columns and 'longitud' in df.columns and df[~df['provincia'].isin(['Desconocida', 'No disponible'])]['provincia'].nunique() > 0:
    print("\nVisualización de Densidad y Precios Medios con Enfoque Provincial")
    print("===================================================================")
    
    df_analisis_geo_prov = df.dropna(subset=['latitud', 'longitud', 'precio', 'provincia'])
    df_analisis_geo_prov = df_analisis_geo_prov[~df_analisis_geo_prov['provincia'].isin(['Desconocida', 'No disponible'])].copy()

    if not df_analisis_geo_prov.empty:
        plt.figure(figsize=(12, 10))
        provincias_a_mostrar = df_analisis_geo_prov['provincia'].unique()
        palette_prov = sns.color_palette("husl", n_colors=len(provincias_a_mostrar))
        
        sns.scatterplot(data=df_analisis_geo_prov, x='longitud', y='latitud', hue='provincia', 
                        palette=palette_prov, s=15, alpha=0.7, legend='full')
        plt.title('Distribución Geográfica de Propiedades Coloreadas por Provincia')
        plt.xlabel('Longitud')
        plt.ylabel('Latitud')
        plt.legend(title='Provincia', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(10, 8))
        sns.kdeplot(data=df_analisis_geo_prov, x='longitud', y='latitud', cmap="viridis_r", fill=True, thresh=0.05, levels=100)
        plt.title('Estimación de Densidad del Kernel (KDE) de Ubicaciones de Propiedades (Andalucía)')
        plt.xlabel('Longitud')
        plt.ylabel('Latitud')
        plt.grid(True)
        plt.show()

        plt.figure(figsize=(12, 9))
        hb = plt.hexbin(df_analisis_geo_prov['longitud'], df_analisis_geo_prov['latitud'], C=np.log1p(df_analisis_geo_prov['precio']), 
                        gridsize=40, cmap='inferno', reduce_C_function=np.mean, mincnt=3, alpha=0.9)
        cb = plt.colorbar(hb, label='Log(Precio Medio + 1)')
        plt.title('Hexbin Plot: Densidad de Propiedades y Precio Medio por Área (Andalucía)')
        plt.xlabel('Longitud')
        plt.ylabel('Latitud')
        plt.grid(True)
        plt.show()
        
    else:
        print("No hay suficientes datos para el análisis geoespacial provincial.")
else:
    print("El DataFrame está vacío o faltan columnas clave ('provincia', 'latitud', 'longitud'), o no hay provincias identificadas.")

# --- Conclusión del Análisis Provincial ---
# Este script ha extendido el análisis exploratorio de viviendas en Andalucía con un fuerte enfoque
# en las diferencias y similitudes entre sus 8 provincias.
#
# Principales Hallazgos y Observaciones del Análisis Provincial:
# 1.  Ingeniería de 'Provincia': Se implementó un método para derivar la provincia.
# 2.  Distribución de Propiedades: Se visualizó el número de propiedades por provincia.
# 3.  Estadísticas Descriptivas Provinciales: Variaciones significativas entre provincias.
# 4.  Comparativa de Precios: Identificación de provincias más caras/baratas.
# 5.  Tipos de Propiedad por Provincia: Mezcla de tipos y precios varían regionalmente.
# 6.  Visualización Geoespacial Provincial: Coherencia geográfica y concentración de datos.
#
# Implicaciones para el TFG:
# *   `provincia` es una característica potencialmente muy influyente.
# *   Considerar interacciones o modelos específicos por provincia.
# *   La calidad de la asignación de `provincia` es crucial.
#
# Este análisis provincial proporciona una base sólida para entender la dinámica del mercado
# inmobiliario andaluz a un nivel más granular.

print("\nProceso de análisis y visualización completado.")

