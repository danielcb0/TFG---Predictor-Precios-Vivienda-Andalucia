from flask import Flask, request, jsonify, send_from_directory
import joblib
import pandas as pd
import os
import numpy as np
from flask_cors import CORS

# Determinar la ruta base del proyecto para acceder a la carpeta frontend
# __file__ es la ruta de app.py (backend/app.py)
# os.path.dirname(__file__) es backend/
# os.path.dirname(os.path.dirname(__file__)) es la raíz del proyecto TFG---Predictor-Precios-Vivienda-Andalucia/
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_FOLDER = os.path.join(PROJECT_ROOT, 'frontend', 'src')
MODELS_FOLDER = os.path.join(PROJECT_ROOT, 'models') # Carpeta de modelos en la raíz del proyecto

app = Flask(__name__, static_folder=FRONTEND_FOLDER, static_url_path='')
CORS(app) # Habilitar CORS para todas las rutas


# --- Carga del Modelo ---
MODEL_FILE = 'final_housing_price_model_andalucia_v3.joblib'
MODEL_PATH = os.path.join(MODELS_FOLDER, MODEL_FILE)
model = None

try:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"No se pudo encontrar el modelo en la ruta: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    app.logger.info(f"Modelo cargado exitosamente desde: {MODEL_PATH}")
except FileNotFoundError as e:
    app.logger.error(f"Error al cargar el modelo: {e}")
    app.logger.error(f"Asegúrate de que el modelo '{MODEL_FILE}' existe en la carpeta '{MODELS_FOLDER}'.")
    model = None
except Exception as e:
    app.logger.error(f"Ocurrió un error inesperado al cargar el modelo: {e}")
    model = None
# --- Fin Carga del Modelo ---

# --- Características esperadas por el modelo (según model_testing v2.ipynb) ---
# Estas son las columnas que el DataFrame de entrada debe tener ANTES del preprocesamiento por el pipeline.
EXPECTED_NUMERIC_FEATURES = ['superficie', 'habitaciones', 'baños', 'latitud', 'longitud']
EXPECTED_CATEGORICAL_FEATURES = ['tipo_propiedad']
ALL_EXPECTED_FEATURES = EXPECTED_NUMERIC_FEATURES + EXPECTED_CATEGORICAL_FEATURES
# --- Fin Características ---

# Ruta para servir el index.html principal del frontend
@app.route('/')
def serve_index():
    return send_from_directory(FRONTEND_FOLDER, 'index.html')

# Flask servirá automáticamente otros archivos estáticos (CSS, JS, imágenes, otros HTML)
# desde FRONTEND_FOLDER gracias a la configuración de static_folder y static_url_path.
# Por ejemplo, /predictor.html servirá frontend/src/predictor.html
# /partials/header.html servirá frontend/src/partials/header.html
# /styles/styles.css servirá frontend/src/styles/styles.css

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        app.logger.error("Intento de predicción pero el modelo no está cargado.")
        return jsonify({'error': 'Modelo no cargado. Revisa los logs del servidor.'}), 500

    try:
        data = request.get_json(force=True)
        app.logger.debug(f"Datos JSON recibidos: {data}")
        
        # Validar que los datos necesarios están presentes
        missing_features = [feature for feature in ALL_EXPECTED_FEATURES if feature not in data]
        if missing_features:
            error_msg = f'Faltan características: {", ".join(missing_features)}'
            app.logger.error(error_msg)
            return jsonify({'error': error_msg}), 400

        # Crear un DataFrame con los datos de entrada
        # El orden de las columnas se establece aquí, coincidiendo con ALL_EXPECTED_FEATURES
        input_data_dict = {feature: data.get(feature) for feature in ALL_EXPECTED_FEATURES}
        input_df = pd.DataFrame([input_data_dict], columns=ALL_EXPECTED_FEATURES)

        # Asegurar tipos numéricos para las columnas numéricas,
        # permitiendo NaN que serán manejados por el imputer del pipeline.
        for col in EXPECTED_NUMERIC_FEATURES:
            original_value = input_df.loc[0, col] # Valor antes de la conversión
            input_df[col] = pd.to_numeric(input_df[col], errors='coerce')
            # Advertir si la conversión resultó en NaN para un valor que originalmente no era None/NaN
            if pd.isna(input_df.loc[0, col]) and original_value is not None and not pd.isna(original_value):
                 app.logger.warning(f"Característica '{col}' con valor original '{original_value}' (tipo: {type(original_value)}) se convirtió a NaN. Verificar tipo de dato enviado.")

        # Las características categóricas (EXPECTED_CATEGORICAL_FEATURES) se pasarán como string/object.
        # El OneHotEncoder del pipeline las manejará (incluyendo imputación si es necesario).
        # Si se reciben como None, el imputer (most_frequent) del pipeline debería manejarlos.
        for col in EXPECTED_CATEGORICAL_FEATURES:
            if input_df.loc[0, col] is None:
                app.logger.debug(f"Característica categórica '{col}' es None. Será manejada por el imputer del pipeline.")
            # Asegurar que sean string si no son None, para el OneHotEncoder
            elif not isinstance(input_df.loc[0, col], str):
                app.logger.warning(f"Característica categórica '{col}' no es string (valor: {input_df.loc[0, col]}, tipo: {type(input_df.loc[0, col])}). Convirtiendo a string.")
                input_df[col] = str(input_df.loc[0, col])


        app.logger.debug(f"DataFrame de entrada para predicción (después de conversiones):\n{input_df.to_string()}")
        app.logger.debug(f"Tipos de datos del DataFrame de entrada:\n{input_df.dtypes}")

        prediction_result = model.predict(input_df)
        
        # La predicción es un array numpy, convertir a tipo nativo de Python para JSON
        if isinstance(prediction_result, np.ndarray) and prediction_result.ndim > 0:
            prediction_value = prediction_result[0]
        else:
            prediction_value = prediction_result 

        # Asegurar que el valor de predicción sea un tipo serializable por JSON
        if isinstance(prediction_value, (np.float32, np.float64)):
            prediction_value = float(prediction_value)
        elif isinstance(prediction_value, (np.int32, np.int64)):
            prediction_value = int(prediction_value)

        app.logger.info(f"Predicción exitosa: {prediction_value}")
        return jsonify({'prediction': prediction_value})
    
    except ValueError as ve:
        error_msg = f'Error en los datos de entrada o durante la conversión: {str(ve)}'
        app.logger.error(error_msg, exc_info=True)
        return jsonify({'error': error_msg}), 400
    except Exception as e:
        error_msg = f'Error durante la predicción: {str(e)}'
        app.logger.error(error_msg, exc_info=True)
        return jsonify({'error': error_msg}), 500

if __name__ == '__main__':
    # Configurar el logger de Flask
    if not app.debug: # No configurar si Flask ya lo hace en modo debug
        import logging
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO) # O DEBUG para más detalle
        app.logger.addHandler(stream_handler)
        app.logger.setLevel(logging.INFO) # O DEBUG

    port = int(os.environ.get('PORT', 5000))
    # use_reloader=False es bueno si la carga del modelo es costosa.
    # Para desarrollo puro, debug=True y use_reloader=True es común.
    # Si debug=True, Flask configura su propio logger.
    app.run(host='0.0.0.0', port=port, debug=True, use_reloader=False)