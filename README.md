# 🎬 Sistema de Recomendación de Películas

Este proyecto utiliza los datasets públicos de TMDB (`tmdb_5000_movies.csv` y `tmdb_5000_credits.csv`) para construir un recomendador basado en contenido. Las recomendaciones se exponen a través de una API Flask.

## 🚀 Cómo correr

1. Coloca los archivos CSV dentro de `data/`
2. Instala las dependencias:
```bash
pip install -r requirements.txt
```

3. Genera los modelos de recomendación:
```bash
python -c "from app.model import MovieRecommender; recommender = MovieRecommender(); recommender.load_data(); recommender.train_model('knn'); recommender.train_model('rf')"
```

4. Inicia la aplicación:
```bash
python app/api.py
```

## 📁 Estructura del Proyecto

```
.
├── app/                    # Código de la aplicación
│   ├── api.py             # API Flask
│   ├── model.py           # Lógica del modelo de recomendación
│   ├── kafka_io.py        # Integración con Kafka
│   └── templates/         # Plantillas HTML
├── data/                  # Datos
│   ├── movies.csv         # Dataset de películas
│   └── ratings.csv        # Dataset de calificaciones
├── models/                # Modelos entrenados (generados automáticamente)
├── tests/                 # Tests unitarios
└── requirements.txt       # Dependencias
```

## 🔧 Configuración de Jenkins

El proyecto incluye configuración para CI/CD con Jenkins:

1. `Jenkinsfile`: Define el pipeline de CI/CD
2. `Dockerfile.jenkins`: Configura el entorno de Jenkins
3. `docker-compose.yml`: Orquesta los servicios necesarios

Para ejecutar Jenkins localmente:
```bash
docker-compose up -d
```

## 📝 Notas

- Los archivos de modelo (`*.joblib`) no se incluyen en el repositorio debido a su tamaño
- Los modelos se generan automáticamente al ejecutar el script de entrenamiento
- Los datasets deben ser colocados manualmente en el directorio `data/`

## 🤝 Contribuciones

1. Fork el repositorio
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request
