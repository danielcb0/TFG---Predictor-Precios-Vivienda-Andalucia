Te voy a presentar mi tfg,, como lo tengo estrucutrado y como quiero que me ayudes.

El proyecto tiene como objetivo construir una plataforma interactiva que utilice modelos de
machine learning entrenados para predecir precios inmobiliarios en las 8 provincias
andaluzas. La plataforma se centrará en el análisis de datos inmobiliarios ya procesados,
permitiendo a los usuarios elegir entre las provincias de Andalucía y ajustar parámetros
relevantes como el número de habitaciones, tamaño de la propiedad, y ubicación geográfica.
El sistema se adaptará automáticamente a la provincia seleccionada, aplicando el modelo de
machine learning adecuado para esa región. Esto garantizará que las predicciones se ajusten a
las características del mercado local.

Objetivos específicos:
1 Entrenar modelos predictivos basados en Big Data y machine learning para cada
provincia.
2 Implementar una aplicación web que permita a los usuarios explorar y analizar precios
de vivienda seleccionando parámetros personalizados.
3.- Proporcionar visualizaciones dinámicas de los precios en función de los parámetros
seleccionados



Ahora mismo tengo la idea de estructurar mi tfg en 4 bloques principales:
1. Introduccion y extracción de datos:  introducción del proyecto. Explicación de como se extraen los datos a través de una api de rapidapi, la cual se automatiza mediante el siguiente script --> [scirpt.py:
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

config.json:
{
    "api_keys": [
        "bac21d693fmshe58334755c3361fp111419jsn8c32e28e0ae9",
        "f2bc0f81a2msh04919275bf0bb7cp1cbe2cjsn78f09cab6db9",
        "4576b2a5b9msh2297cf55eb3c3eep1d2882jsn38ec95c6446e",
        "3c36737f20mshc0960271b010c80p1b44c5jsnf9da34fc275f",
        "e6dabcaf83msh17ab7975b03f4a1p17e3f8jsn1f9612bace60",
        "778ccf6711msh7ea5c4ad4ac0234p1fe8e0jsn874d3c02e253",
        "d729bde3damshf196c4d6678bfadp10299ejsn82ba1804d78e",
        "77131c23f8msh8aa7777b250d77cp10296ajsn8e3be71c84c8",
        "ddaaba1fb5msh7cab091c1a62819p1f6a9djsn227dc703fad0",
        "0b809214c0msh7fcd9d615cf928ap1c5924jsn670795415dec",
        "105d2455demshbb11cbd66ff3507p14d643jsn20a20fcd1b15",
        "8a7c72f194msh1329748676c28fep1725a9jsn0e5fcbf8e204",
        "07c1edd006msh85c523454fca2d2p1f6412jsn75fd18d82767"
    ],
    "provinces": {
        "Sevilla": "0-EU-ES-41",
        "Granada": "0-EU-ES-18",
        "Málaga": "0-EU-ES-29",
        "Cádiz": "0-EU-ES-11",
        "Córdoba": "0-EU-ES-14",
        "Huelva": "0-EU-ES-21",
        "Almería": "0-EU-ES-04",
        "Jaén": "0-EU-ES-23"
    }
}
]
Los datos que se obtienen son dos csv, uno por cada provincia de andalucia, y uno que empieza extrayendo los precios de forma ascendente y otro de forma descendente

2. Preprocesado de datos (Buscar otro título): una vez obtenidos los datos en crudo estos deberán ser procesados y limpiados mediante técnicas de big data y exploración de estos datos en un panel interactivo de power bi

3. Creación del modelo: una vez obtenido un csv llamado andalucia con todos los datos ya listos para entrenar un modelo basado en el siguiente codigo: [# Recuperacion Modulo 5
 
**Autor:** Daniel Carrera Bonilla

**Fecha:** 11 de Julio de 2024

### 1. Carga de datos
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import shap
import h2o
from h2o.automl import H2OAutoML

Se carga el dataset que contiene información sobre viviendas en Melbourne. Esta visualización inicial ayuda a entender la estructura de los datos y las primeras filas del dataset.
# Carga de datos
data = pd.read_csv("datos_casas_melbourne.csv")

# Visualización inicial de los datos
data.head()
### 2. Información general del dataset
Se obtiene información general del dataset, como el tipo de datos de cada columna y la cantidad de valores no nulos. Esto es importante para identificar posibles problemas como valores faltantes:

data.info()

### 3. Estadísticas Descriptivas del Dataset
Las  estadísticas descriptivas proporcionan una visión general de la distribución de los datos, como la media, la desviación estándar, y los percentiles. Esto ayudaa a identificar posibles valores atípicos y entender la escala de los datos:
# Estadísticas descriptivas del dataset
data.describe()

### 4. Histograma de Variables Numéricas

Se visualizan los histogramas de las variables numéricas para entender su distribución. Esto es útil para identificar distribuciones sesgadas o valores extremos:
# Histograma de variables numéricas
data.hist(bins=50, figsize=(20,15))
plt.show()

 

1. **Rooms (Habitaciones)**: La mayoría de las propiedades tienen entre 2 y 4 habitaciones, con un pico claro en 3 habitaciones. Hay muy pocas propiedades con más de 6 habitaciones.

2. **Bedroom2 (Habitaciones secundarias)**: Similar a las habitaciones, la mayoría de las propiedades tienen entre 2 y 3 habitaciones secundarias, con un pequeño número que tiene más de 6.

3. **Bathroom (Baños)**: La mayoría de las propiedades tienen 1 o 2 baños, con muy pocas propiedades que tienen más de 3 baños.

4. **Landsize (Tamaño del terreno)**: La distribución está altamente sesgada a la derecha, lo que indica que la mayoría de los terrenos son pequeños, pero hay algunos terrenos excepcionalmente grandes que influyen en la distribución.

5. **Latitude (Latitud)**: La distribución es aproximadamente normal, centrada alrededor de -37.8 grados de latitud.

6. **Longitude (Longitud)**: La distribución también es aproximadamente normal, centrada alrededor de 145 grados de longitud.

7. **Propertycount (Número de propiedades)**: La mayoría de las áreas tienen menos de 5000 propiedades, pero algunas áreas tienen más de 20000 propiedades, lo que sugiere una gran variabilidad en la densidad de propiedades.

8. **Price (Precio)**: La distribución está altamente sesgada a la derecha, indicando que la mayoría de las propiedades tienen precios más bajos, pero hay algunas propiedades con precios extremadamente altos.

 
### 5. Análisis de Correlación

Crearemos una matriz de correlación para las variables numéricas, visualizada con un heatmap. Esto nos ayudará a identificar relaciones lineales entre las variables, crucial para seleccionar características relevantes para los modelos.
# Filtrar solo las columnas numéricas para el análisis de correlación
numerical_data = data.select_dtypes(include=[np.number])

# Matriz de correlación
corr_matrix = numerical_data.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()

 
1. **Rooms (Habitaciones)**:
   - Alta correlación positiva con **Bedroom2 (0.94)** y **Bathroom (0.59)**, indicando que a medida que aumenta el número de habitaciones, también tiende a aumentar el número de habitaciones secundarias y baños.
   - Correlación moderada positiva con **Price (0.5)**, sugiriendo que más habitaciones están asociadas con precios más altos.

2. **Bedroom2 (Habitaciones secundarias)**:
   - Alta correlación positiva con **Rooms (0.94)** y **Bathroom (0.58)**, reflejando una relación similar a la mencionada anteriormente.
   - Correlación moderada positiva con **Price (0.48)**, similar a la relación de las habitaciones con el precio.

3. **Bathroom (Baños)**:
   - Alta correlación positiva con **Rooms (0.59)** y **Bedroom2 (0.58)**.
   - Correlación moderada positiva con **Price (0.47)**, indicando que más baños también están asociados con precios más altos.

4. **Landsize (Tamaño del terreno)**:
   - Muy baja correlación con la mayoría de las variables, excepto una ligera correlación positiva con **Price (0.038)**. Esto sugiere que el tamaño del terreno no tiene una relación lineal fuerte con otras variables en este conjunto de datos.

5. **Latitude (Latitud)**:
   - Correlación negativa moderada con **Longitude (-0.36)**, indicando que hay una tendencia a que propiedades en ciertos rangos de latitud estén en rangos opuestos de longitud.
   - Correlación negativa con **Price (-0.21)**, sugiriendo que la latitud puede tener alguna influencia en el precio, posiblemente debido a la ubicación geográfica de las propiedades.

6. **Longitude (Longitud)**:
   - Correlación negativa moderada con **Latitude (-0.36)**.
   - Correlación positiva baja con **Price (0.2)**, indicando que la longitud también puede influir en los precios.

7. **Propertycount (Número de propiedades)**:
   - Muy baja correlación con todas las demás variables, incluyendo **Price (-0.042)**, indicando que el número de propiedades en un área no tiene una relación lineal fuerte con estas variables.

8. **Price (Precio)**:
   - Correlación positiva moderada con **Rooms (0.5)**, **Bedroom2 (0.48)** y **Bathroom (0.47)**, sugiriendo que estas características son importantes al determinar el precio de una propiedad.
   - Correlación negativa con **Latitude (-0.21)**, sugiriendo una posible influencia geográfica en los precios.

En resumen, el heatmap de correlación nos está mostrando que las variables que tienen una relación más fuerte con el precio de la propiedad son el número de habitaciones, habitaciones secundarias y baños. El tamaño del terreno y el número de propiedades no muestran una correlación fuerte con el precio. La latitud y la longitud tienen alguna influencia, aunque no tan fuertee como las variables relacionadas con la cantidad de habitaciones y baños. 
### 6. Preprocesamiento de Datos

Se configura el preprocesamiento para las características numéricas y categóricas. Las características numéricas se imputan con la mediana y se escalan, mientras que las categóricas se imputan con la moda y se codifican con OneHotEncoder. Este paso es crucial para preparar los datos de manera que los modelos puedan procesarlos adecuadamente.
numeric_features = numerical_data.columns.drop('Price')
categorical_features = data.select_dtypes(include=[object]).columns

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])
### 7. Separación de Características y Variable Objetivo

Separamos las características independientes (X) de la variable objetivo (y), que en este caso es el precio de las viviendas.
X = data.drop('Price', axis=1)
y = data['Price']
### 8. División del Conjunto de Datos en Entrenamiento y Prueba

El dataset se divide en conjuntos de entrenamiento y prueba. El 80% de los datos se utiliza para entrenar los modelos y el 20% para evaluarlos, asegurando que los resultados sean representativos y no estén sobreajustados.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

### 9. Definición y Entrenamiento de Modelos

Entrenamos varios modelos de regresión utilizando un pipeline que incluye el preprocesador y el modelo. 

Se evalúan los modelos usando métricas como **MSE**, **MAE** y **R2**, y seleccionaremos  el mejor modelo basado en el **R2**, que mide la proporción de la varianza en la variable objetivo que es predecible a partir de las características independientes.
# 10. Definición y entrenamiento de modelos
models = {
    'Linear Regression': LinearRegression(),
    'Lasso': Lasso(),
    'K-Nearest Neighbors': KNeighborsRegressor(),
    'Decision Tree': DecisionTreeRegressor(),
    'Random Forest': RandomForestRegressor(),
    'Gradient Boosting': GradientBoostingRegressor(),
    'Support Vector Regressor': SVR(),
    'Neural Network': MLPRegressor(max_iter=500),
}

results = []
for name, model in models.items():
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results.append({'Model': name, 'MSE': mse, 'MAE': mae, 'R2': r2})

results_df = pd.DataFrame(results).sort_values(by='R2', ascending=False)
results_df
### 10. Selección y Explicación del Modelo Óptimo

Seleccionamos el modelo con el mejor rendimiento (mayor R2) y entrenamos nuevamente el pipeline con este modelo utilizando todos los datos de entrenamiento.

Luego usamos **SHAP** para interpretar el modelo seleccionado, proporcionando una visualización de los valores SHAP que muestran la importancia de cada característica en las predicciones. Esto nos ayuda a explicar el modelo de manera comprensible para las partes interesadas del negocio.
# Obtén el nombre del mejor modelo
best_model_name = results_df.iloc[0]['Model']
best_model = models[best_model_name]

# Crear y entrenar el pipeline con el mejor modelo
pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', best_model)])
pipeline.fit(X_train, y_train)


import joblib

# Guardar el mejor modelo
model_path = "best_model.pkl"
joblib.dump(pipeline, model_path)

# Preprocesar los datos de prueba
X_test_preprocessed = preprocessor.transform(X_test)

# Crear el explicador SHAP específico para árboles
explainer = shap.TreeExplainer(pipeline['model'])
shap_values = explainer.shap_values(X_test_preprocessed)

# Visualización de los valores SHAP
shap.summary_plot(shap_values, X_test_preprocessed, feature_names=X.columns)

### Análisis de la Visualización SHAP

1. **Importancia de las Características**:
   - En el eje vertical, se enumeran las características del modelo, ordenadas por importancia.
   - Cada punto en el gráfico representa un valor SHAP para una muestra particular, mostrando cómo esa característica influyó en la predicción del modelo.
   - Los colores indican el valor de la característica: rojo para valores altos y azul para valores bajos.

2. **Interpretación de los Valores SHAP**:
   - El eje horizontal muestra el impacto de cada característica en la predicción. Los valores positivos incrementan la predicción, mientras que los valores negativos la disminuyen.
   - La dispersión horizontal de los puntos indica la variabilidad en la influencia de la característica sobre diferentes predicciones.

3. **Principales Características**:
   - **La característica en la parte superior** (no identificada explícitamente en el gráfico, pero basada en la interpretación común de estos gráficos, podría ser "Rooms" o "Price") tiene un impacto significativo y variable en las predicciones.
   - Las características subsiguientes también tienen impactos considerables, aunque menores en comparación con la primera. Esto se observa en la menor dispersión horizontal y en la cantidad de puntos más concentrados cerca del centro (0).

4. **Patrones Específicos**:
   - **Valores altos de características** (rojo) suelen estar a la derecha del cero, indicando que incrementan las predicciones del modelo.
   - **Valores bajos de características** (azul) suelen estar a la izquierda del cero, indicando que disminuyen las predicciones del modelo.


Esta visualización SHAP ayuda a entender qué características son más influyentes en el modelo de predicción de precios de propiedades. 

Los resultados indican que algunas características tienen un impacto mucho mayor que otras, y que los valores altos de estas características generalmente incrementan la predicción del precio. 

### 12.Eleccion y uso del mejor modelo, haciendo uso de H2O AutoML: 

1. **Inicio de H2O**: Se inicializa el servidor H2O local.
2. **Conversión de datos**: El dataset de pandas se convierte a un formato H2OFrame, que es compatible con H2O.
3. **División del dataset**: Se divide el dataset en conjuntos de entrenamiento y prueba con una proporción de 80-20.
4. **Definición de características y variable objetivo**: Se seleccionan las columnas que se utilizarán como características y se define la variable objetivo (Price).
5. **Configuración y ejecución de AutoML**: Se configura H2O AutoML para entrenar varios modelos durante un máximo de 3600 segundos (1 hora) con 5 fold cross-validation.
6. **Resultados**: Se muestra el leaderboard con los 10 mejores modelos entrenados por AutoML.
7. **Evaluación del mejor modelo**: Se evalúa el rendimiento del mejor modelo en el conjunto de prueba, proporcionando métricas como MSE, RMSE, MAE, y R2.

 
# Iniciar H2O
h2o.init()

# Convertir los datos a un frame de H2O
h2o_df = h2o.H2OFrame(data)

# División del dataset en entrenamiento y prueba
train, test = h2o_df.split_frame(ratios=[0.8])

# Definir características y variable objetivo
x = train.columns
y = 'Price'
x.remove(y)

# Configuración de AutoML
aml = H2OAutoML(max_runtime_secs=3600, seed=42, nfolds=5)
aml.train(x=x, y=y, training_frame=train)

# Resultados del AutoML
lb = aml.leaderboard
print(lb.head(rows=10))

# Evaluación del mejor modelo
best_model = aml.leader
perf = best_model.model_performance(test)
print(perf)

## 13.Interpretacion final de los resultados del modelo entrenado

Por último, en un archivo externo llamado 'flask-api.py', hemos creado el siguiente script con python con el que, mediante el uso del microframework **FLASK** y el mejor modelo generado en el paso anterior, tras un entrenamiento de 59 minutos, creamos una API con la que poder acceder a los resulados.

Código:
```python
from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Cargar el modelo
model_path = "best_model.pkl"
model = joblib.load(model_path)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    data_df = pd.DataFrame(data)
    prediction = model.predict(data_df)
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(port=5000, debug=True)

```

Ejemplo uso de llamada al endpoint de la API:
`curl -X POST -H "Content-Type: application/json" -d '{"Rooms": [3], "Type": ["h"], "Bedroom2": [2], "Bathroom": [1], "Landsize": [156], "Lattitude": [-37.8079], "Longtitude": [144.9934], "Regionname": ["Northern Metropolitan"], "Propertycount": [4019]}' http://127.0.0.1:5000/predict
`

Resultado:
```bash
{
  "prediction": [
    1165880.0
  ]
}
```

Esto significa que el modelo predictivo ha estimado que el precio de la vivienda, basado en las características proporcionadas en la solicitud, es de aproximadamente 1,165,880 dólares.
]

4. Creación de una aplicación web para mostrar los resultados obtenidos


Una vez presentados los puntos con los que quiero trabajar en mi tfg, quiero que en primer lugar hagas lo siguiente. Te voy a proporcionar mi plantilla latex de tfg. Quiero que me la devuelvas modificada de manera que queden establecidos estos puntos y posibles subpuntos que se te ocurran. También debes rellenarme dichos puntos con una explicación de un parrafo sobre lo que debe ir en cada punto y subpunto.
Plantilla:
\documentclass[a4paper,11pt]{book}
%\documentclass[a4paper,twoside,11pt,titlepage]{book}
\usepackage{listings}
\usepackage[utf8]{inputenc}
\usepackage[spanish]{babel}

% \usepackage[style=list, number=none]{glossary} %
%\usepackage{titlesec}
%\usepackage{pailatino}

\decimalpoint
\usepackage{dcolumn}
\newcolumntype{.}{D{.}{\esperiod}{-1}}
\makeatletter
\addto\shorthandsspanish{\let\esperiod\es@period@code}
\makeatother


%\usepackage[chapter]{algorithm}
\RequirePackage{verbatim}
%\RequirePackage[Glenn]{fncychap}
\usepackage{fancyhdr}
\usepackage{graphicx}
\usepackage{afterpage}

\usepackage{longtable}

\usepackage[pdfborder={000}]{hyperref} %referencia

% ********************************************************************
% Re-usable information
% ********************************************************************
\newcommand{\myTitle}{Desarrollo y análisis de una Plataforma Predictiva de Precios Inmobiliarios mediante Machine Learning y Big Data para las Provincias de Andalucía\xspace}
\newcommand{\myDegree}{Grado en Ingeniería Informática\xspace}
\newcommand{\myName}{Daniel Carrera Bonilla\xspace}
\newcommand{\myProf}{Ignacio Javier Pérez Gálvez\xspace}
\newcommand{\myOtherProf}{Nombre Apllido1 Apellido2 (tutor2)\xspace}
%\newcommand{\mySupervisor}{Put name here\xspace}
\newcommand{\myFaculty}{Escuela Técnica Superior de Ingenierías Informática y de
Telecomunicación\xspace}
\newcommand{\myFacultyShort}{E.T.S. de Ingenierías Informática y de
Telecomunicación\xspace}
\newcommand{\myDepartment}{Departamento de Ciencias de la Computación e Inteligencia Artificial\xspace}
\newcommand{\myUni}{\protect{Universidad de Granada}\xspace}
\newcommand{\myLocation}{Granada\xspace}
\newcommand{\myTime}{\today\xspace}
\newcommand{\myVersion}{Version 0.1\xspace}


\hypersetup{
pdfauthor = {\myName (email (en) ugr (punto) es)},
pdftitle = {\myTitle},
pdfsubject = {},
pdfkeywords = {palabra_clave1, palabra_clave2, palabra_clave3, ...},
pdfcreator = {LaTeX con el paquete ....},
pdfproducer = {pdflatex}
}

%\hyphenation{}


%\usepackage{doxygen/doxygen}
%\usepackage{pdfpages}
\usepackage{url}
\usepackage{colortbl,longtable}
\usepackage[stable]{footmisc}
%\usepackage{index}

%\makeindex
%\usepackage[style=long, cols=2,border=plain,toc=true,number=none]{glossary}
% \makeglossary

% Definición de comandos que me son tiles:
%\renewcommand{\indexname}{Índice alfabético}
%\renewcommand{\glossaryname}{Glosario}

\pagestyle{fancy}
\fancyhf{}
\fancyhead[LO]{\leftmark}
\fancyhead[RE]{\rightmark}
\fancyhead[RO,LE]{\textbf{\thepage}}
\renewcommand{\chaptermark}[1]{\markboth{\textbf{#1}}{}}
\renewcommand{\sectionmark}[1]{\markright{\textbf{\thesection. #1}}}

\setlength{\headheight}{1.5\headheight}

\newcommand{\HRule}{\rule{\linewidth}{0.5mm}}
%Definimos los tipos teorema, ejemplo y definición podremos usar estos tipos
%simplemente poniendo \begin{teorema} \end{teorema} ...
\newtheorem{teorema}{Teorema}[chapter]
\newtheorem{ejemplo}{Ejemplo}[chapter]
\newtheorem{definicion}{Definición}[chapter]

\definecolor{gray97}{gray}{.97}
\definecolor{gray75}{gray}{.75}
\definecolor{gray45}{gray}{.45}
\definecolor{gray30}{gray}{.94}

\lstset{ frame=Ltb,
     framerule=0.5pt,
     aboveskip=0.5cm,
     framextopmargin=3pt,
     framexbottommargin=3pt,
     framexleftmargin=0.1cm,
     framesep=0pt,
     rulesep=.4pt,
     backgroundcolor=\color{gray97},
     rulesepcolor=\color{black},
     %
     stringstyle=\ttfamily,
     showstringspaces = false,
     basicstyle=\scriptsize\ttfamily,
     commentstyle=\color{gray45},
     keywordstyle=\bfseries,
     %
     numbers=left,
     numbersep=6pt,
     numberstyle=\tiny,
     numberfirstline = false,
     breaklines=true,
   }
 
% minimizar fragmentado de listados
\lstnewenvironment{listing}[1][]
   {\lstset{#1}\pagebreak[0]}{\pagebreak[0]}

\lstdefinestyle{CodigoC}
   {
	basicstyle=\scriptsize,
	frame=single,
	language=C,
	numbers=left
   }
\lstdefinestyle{CodigoC++}
   {
	basicstyle=\small,
	frame=single,
	backgroundcolor=\color{gray30},
	language=C++,
	numbers=left
   }

 
\lstdefinestyle{Consola}
   {basicstyle=\scriptsize\bf\ttfamily,
    backgroundcolor=\color{gray30},
    frame=single,
    numbers=none
   }


\newcommand{\bigrule}{\titlerule[0.5mm]}


%Para conseguir que en las páginas en blanco no ponga cabecerass
\makeatletter
\def\clearpage{%
  \ifvmode
    \ifnum \@dbltopnum =\m@ne
      \ifdim \pagetotal <\topskip
        \hbox{}
      \fi
    \fi
  \fi
  \newpage
  \thispagestyle{empty}
  \write\m@ne{}
  \vbox{}
  \penalty -\@Mi
}
\makeatother

\usepackage{pdfpages}
\begin{document}
\input{portada/portada}
\input{prefacios/prefacio}
%\frontmatter
%\tableofcontents
%\listoffigures
%\listoftables
%
%\mainmatter
%\setlength{\parskip}{5pt}

%\input{capitulos/01_Introduccion}
%
%\input{capitulos/02_EspecificacionRequisitos}
%
%\input{capitulos/03_Planificacion}
%
%\input{capitulos/04_Analisis}
%
%\input{capitulos/05_Diseno}
%
%\input{capitulos/06_Implementacion}
%
%\input{capitulos/07_Pruebas}
%
%\input{capitulos/08_Conclusiones}
%
%%\chapter{Conclusiones y Trabajos Futuros}
%
%
%%\nocite{*}
%\bibliography{bibliografia/bibliografia}\addcontentsline{toc}{chapter}{Bibliografía}
%\bibliographystyle{miunsrturl}
%
%\appendix
%\input{apendices/manual_usuario/manual_usuario}
%%\input{apendices/paper/paper}
%\input{glosario/entradas_glosario}
% \addcontentsline{toc}{chapter}{Glosario}
% \printglossary
\chapter*{}
\thispagestyle{empty}

\end{document}