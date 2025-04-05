import unittest
import os
import pandas as pd
import numpy as np
from app.model import MovieRecommender

class TestMovieRecommender(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Asegurar que estamos en el directorio correcto para los tests
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        os.chdir(self.base_dir)
        
        # Verificar que los archivos existen
        self.assertTrue(os.path.exists('data/movies.csv'), "movies.csv no encontrado")
        self.assertTrue(os.path.exists('data/ratings.csv'), "ratings.csv no encontrado")
        
        # Inicializar el recomendador
        self.recommender = MovieRecommender()
        self.recommender.load_data()
        self.recommender.train_model()  # Entrenar el modelo después de cargar datos
        
        # Guardar un usuario válido para las pruebas
        self.valid_user_id = self.recommender.ratings_df['userId'].iloc[0]
        
    def test_load_data(self):
        """Test if data is loaded correctly"""
        self.assertIsNotNone(self.recommender.movies_df)
        self.assertIsNotNone(self.recommender.ratings_df)
        self.assertIsNotNone(self.recommender.user_movie_matrix)
        
        # Verificar que los DataFrames tienen datos
        self.assertGreater(len(self.recommender.movies_df), 0)
        self.assertGreater(len(self.recommender.ratings_df), 0)
        
    def test_get_recommendations_rf(self):
        """Test Random Forest recommendations"""
        n_recommendations = 2  # Reducido porque tenemos pocos datos
        
        # Cambiar el modelo a RF y entrenar
        self.recommender.model_type = 'rf'
        self.recommender.train_model()
        
        # Usar un usuario válido de los datos
        result = self.recommender.get_recommendations(self.valid_user_id, n_recommendations)
        
        # Verificar la estructura del resultado
        self.assertTrue(result['success'])
        self.assertLessEqual(len(result['recommendations']), n_recommendations)
        
        # Verificar la estructura de cada recomendación
        for rec in result['recommendations']:
            self.assertIn('title', rec)
            self.assertIn('genres', rec)
            self.assertIn('predicted_rating', rec)
            self.assertIsInstance(rec['predicted_rating'], float)
            
    def test_get_recommendations_knn(self):
        """Test KNN recommendations"""
        n_recommendations = 2  # Reducido porque tenemos pocos datos
        
        # Cambiar el modelo a KNN y entrenar
        self.recommender.model_type = 'knn'
        self.recommender.train_model()
        
        # Usar un usuario válido de los datos
        result = self.recommender.get_recommendations(self.valid_user_id, n_recommendations)
        
        # Verificar la estructura del resultado
        self.assertTrue(result['success'])
        self.assertLessEqual(len(result['recommendations']), n_recommendations)
        
        # Verificar la estructura de cada recomendación
        for rec in result['recommendations']:
            self.assertIn('title', rec)
            self.assertIn('genres', rec)
            self.assertIn('predicted_rating', rec)
            self.assertIsInstance(rec['predicted_rating'], float)
            
    def test_invalid_user_id(self):
        """Test behavior with invalid user ID"""
        user_id = 999999  # Non-existent user ID
        n_recommendations = 5
        
        result = self.recommender.get_recommendations(user_id, n_recommendations)
        
        # Verificar que la respuesta indica error
        self.assertFalse(result['success'])
        self.assertIn('error', result)
        
    def test_invalid_model(self):
        """Test behavior with invalid model name"""
        self.recommender.model_type = 'invalid_model'
        
        # Intentar obtener recomendaciones debería fallar
        result = self.recommender.get_recommendations(self.valid_user_id, 5)
        self.assertFalse(result['success'])
        self.assertIn('error', result)

if __name__ == '__main__':
    unittest.main() 