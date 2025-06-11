.gitattributes
.gitignore
docker-compose.yml
Dockerfile
README.md
requirements.txt
setup_structure.sh
backend/
    app.py
    requirements.txt
    controllers/
    models/
    routes/
    utils/
data/
    config.json 
    extraccionViviendasMensual.py
    extraccionViviendasMensualv2.py
    preprocess.py
    procesamientoCSVs.py
    clean/
    processed/
    raw/

frontend/
        style/
            style.css
    index.html
models/
    Aqui se guardan los modelos que se crean en model_testing_v3.ipynb
notebooks/
    `carga_limpieza.ipynb`// se encarga de la limpieza de los datos generando un csv final, 
    `analisis_visualizacion_por_provincia.ipynb` //analisis por provincia de andalucia, 
    `analisis_visualizacion v2.ipynb` //analisis general del dataset,  `model_testing v3.ipynb` // generacion del modelo predictivo
powerbi/
    aqui va un cuadro de mando en  powerbi
