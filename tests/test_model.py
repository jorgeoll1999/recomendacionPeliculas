import pytest
import pandas as pd
import numpy as np
from app.model import MovieRecommender

@pytest.fixture
def sample_data():
    # Crear datos de ejemplo para pruebas
    movies = pd.DataFrame({
        'movieId': [1, 2, 3, 4, 5],
        'title': ['Movie1', 'Movie2', 'Movie3', 'Movie4', 'Movie5'],
        'genres': ['Action', 'Comedy', 'Drama', 'Sci-Fi', 'Romance']
    })
    
    ratings = pd.DataFrame({
        'userId': [1, 1, 1, 2, 2, 2, 3, 3, 3],
        'movieId': [1, 2, 3, 2, 3, 4, 1, 3, 5],
        'rating': [5.0, 3.5, 4.0, 4.0, 3.0, 5.0, 4.5, 4.0, 3.5]
    })
    
    return movies, ratings

def test_knn_recommender(sample_data):
    movies, ratings = sample_data
    
    # Guardar datos de ejemplo
    movies.to_csv('data/movies.csv', index=False)
    ratings.to_csv('data/ratings.csv', index=False)
    
    # Inicializar y entrenar modelo KNN
    recommender = MovieRecommender(model_type='knn')
    recommender.load_data()
    recommender.train_model()
    
    # Obtener recomendaciones
    recommendations = recommender.get_recommendations(1, n_recommendations=2)
    
    # Verificar que las recomendaciones son válidas
    assert len(recommendations) == 2
    assert all(movie_id in movies['movieId'].values for movie_id in recommendations)
    assert 1 not in recommendations  # No debería recomendar películas ya vistas

def test_rf_recommender(sample_data):
    movies, ratings = sample_data
    
    # Guardar datos de ejemplo
    movies.to_csv('data/movies.csv', index=False)
    ratings.to_csv('data/ratings.csv', index=False)
    
    # Inicializar y entrenar modelo Random Forest
    recommender = MovieRecommender(model_type='rf')
    recommender.load_data()
    recommender.train_model()
    
    # Obtener recomendaciones
    recommendations = recommender.get_recommendations(1, n_recommendations=2)
    
    # Verificar que las recomendaciones son válidas
    assert len(recommendations) == 2
    assert all(movie_id in movies['movieId'].values for movie_id in recommendations)
    assert 1 not in recommendations  # No debería recomendar películas ya vistas

def test_invalid_user():
    recommender = MovieRecommender(model_type='knn')
    recommender.load_data()
    recommender.train_model()
    
    # Intentar obtener recomendaciones para un usuario que no existe
    recommendations = recommender.get_recommendations(999999)
    assert len(recommendations) == 0 