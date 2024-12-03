#!/bin/bash

# Navegar a la carpeta principal
#cd TFG---Predictor-Precios-Vivienda-Andalucia || exit

# Crear las carpetas principales
mkdir -p data/raw data/processed models/trained_models notebooks \
         frontend/src/{components,pages,styles} frontend/public \
         backend/{routes,controllers,models,utils} \
         tests/{frontend,backend} config docs

# Crear archivos básicos
touch data/preprocess.py
touch models/train.py
touch models/evaluate.py
touch notebooks/eda.ipynb
touch notebooks/model_testing.ipynb
touch frontend/package.json
touch backend/app.py
touch backend/requirements.txt
touch config/database_config.py
touch config/api_config.py
touch config/.env.example
touch docs/architecture.md
touch docs/user_manual.md
touch .gitignore
touch Dockerfile
touch docker-compose.yml

# Añadir contenido básico a algunos archivos
echo "# Descripción del Proyecto" >> README.md
echo "*.pyc" >> .gitignore
echo "__pycache__/" >> .gitignore
echo "node_modules/" >> .gitignore
echo "Dockerfile básico para el proyecto" >> Dockerfile
echo "version: '3.8'\nservices:" >> docker-compose.yml

# Confirmación
echo "Estructura de carpetas y archivos creada exitosamente."
