import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors, KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.sparse import csr_matrix
import joblib
import os

class MovieRecommender:
    def __init__(self, model_type='knn'):
        """
        Inicializa el recomendador con el tipo de modelo especificado
        Args:
            model_type (str): 'knn' o 'rf' para seleccionar el modelo
        """
        self.model_type = model_type
        self.model = None
        self.movies_df = None
        self.ratings_df = None
        self.user_movie_matrix = None
        self.user_means = None  # Media de calificaciones por usuario
        self.movie_means = None  # Media de calificaciones por película
        self.global_mean = None  # Media global de calificaciones
        self.user_mapper = None
        self.movie_mapper = None
        self.n_users = None
        self.n_movies = None
        self.test_ratings = None
        
    def load_data(self):
        """Carga y prepara los datos para el modelo."""
        print("Cargando datos de películas...")
        self.movies_df = pd.read_csv('data/movies.csv')
        
        print("Cargando datos de ratings...")
        self.ratings_df = pd.read_csv('data/ratings.csv')
        
        print("Preparando datos...")
        # Filtrar usuarios y películas con al menos 1 rating
        movie_counts = self.ratings_df['movieId'].value_counts()
        user_counts = self.ratings_df['userId'].value_counts()
        
        valid_movies = movie_counts[movie_counts >= 1].index
        valid_users = user_counts[user_counts >= 1].index
        
        self.ratings_df = self.ratings_df[
            self.ratings_df['movieId'].isin(valid_movies) & 
            self.ratings_df['userId'].isin(valid_users)
        ]
        
        # Dividir datos en entrenamiento y prueba
        if len(self.ratings_df) > 1:  # Si hay más de un rating
            self.train_data, self.test_data = train_test_split(
                self.ratings_df, 
                test_size=0.2, 
                random_state=42
            )
        else:
            # Si solo hay un rating, usar todo para entrenamiento
            self.train_data = self.ratings_df
            self.test_data = pd.DataFrame(columns=self.ratings_df.columns)
        
        print("Datos procesados:")
        print(f"Total ratings: {len(self.ratings_df)}")
        print(f"Usuarios train: {self.train_data['userId'].nunique()}")
        print(f"Usuarios test: {self.test_data['userId'].nunique()}")
        print(f"Películas: {len(valid_movies)}")
        
        print("Calculando estadísticas...")
        # Calcular estadísticas por película
        self.movie_stats = self.ratings_df.groupby('movieId').agg({
            'rating': ['mean', 'count', 'std']
        }).reset_index()
        self.movie_stats.columns = ['movieId', 'avg_rating', 'rating_count', 'rating_std']
        
        print("Creando mapeos de IDs...")
        # Crear mapeos de IDs
        self.user_mapper = {id: idx for idx, id in enumerate(valid_users)}
        self.movie_mapper = {id: idx for idx, id in enumerate(valid_movies)}
        
        print("Creando matriz usuario-película...")
        # Crear matriz usuario-película
        self.user_movie_matrix = pd.pivot_table(
            self.train_data,
            values='rating',
            index='userId',
            columns='movieId',
            fill_value=0
        )
        
        print("Datos cargados y procesados exitosamente")
        
    def train_model(self):
        """Entrena el modelo seleccionado."""
        if self.model_type == 'knn':
            # Entrenar modelo KNN
            self.model = KNeighborsRegressor(n_neighbors=5)
            
            # Preparar datos para KNN
            X = self.user_movie_matrix.values
            y = X.ravel()  # Aplanar la matriz para regresión
            
            # Filtrar valores cero (no ratings)
            mask = y != 0
            X_filtered = X[mask.reshape(X.shape)]
            y_filtered = y[mask]
            
            if len(X_filtered) > 0:
                self.model.fit(X_filtered.reshape(-1, 1), y_filtered)
                print("Modelo KNN entrenado exitosamente")
            else:
                print("No hay suficientes datos para entrenar el modelo KNN")
                
        elif self.model_type == 'rf':
            # Entrenar Random Forest
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
            X = self._create_features(self.train_data)
            y = self.train_data['rating']
            self.model.fit(X, y)
            print("Modelo Random Forest entrenado exitosamente")
        else:
            raise ValueError(f"Modelo {self.model_type} no soportado")
    
    def _evaluate_knn(self, model, test_sample):
        """Evalúa el modelo KNN usando MSE."""
        if test_sample.empty:
            return 0.0  # Retornar 0 si no hay datos de prueba
            
        actuals = []
        predictions = []
        
        for _, row in test_sample.iterrows():
            user_id = row['userId']
            movie_id = row['movieId']
            actual_rating = row['rating']
            
            # Obtener el vector de usuario
            user_vector = self.user_movie_matrix.loc[user_id].values.reshape(1, -1)
            
            # Predecir el rating
            predicted_rating = model.predict(user_vector)[0]
            
            actuals.append(actual_rating)
            predictions.append(predicted_rating)
            
        return mean_squared_error(actuals, predictions) if actuals else 0.0
    
    def _create_features(self, ratings_df):
        """
        Crea características adicionales para el Random Forest
        """
        features = []
        
        # Características base: userId y movieId
        features.append(ratings_df[['userId', 'movieId']].values)
        
        # Media de calificaciones del usuario
        user_means = ratings_df.groupby('userId')['rating'].mean()
        user_means_array = ratings_df['userId'].map(user_means).values.reshape(-1, 1)
        features.append(user_means_array)
        
        # Media de calificaciones de la película
        movie_means = ratings_df.groupby('movieId')['rating'].mean()
        movie_means_array = ratings_df['movieId'].map(movie_means).values.reshape(-1, 1)
        features.append(movie_means_array)
        
        # Número de calificaciones por usuario
        user_counts = ratings_df.groupby('userId').size()
        user_counts_array = ratings_df['userId'].map(user_counts).values.reshape(-1, 1)
        features.append(user_counts_array)
        
        # Número de calificaciones por película
        movie_counts = ratings_df.groupby('movieId').size()
        movie_counts_array = ratings_df['movieId'].map(movie_counts).values.reshape(-1, 1)
        features.append(movie_counts_array)
        
        return np.hstack(features)
    
    def get_recommendations(self, user_id, n_recommendations=5):
        """Obtiene recomendaciones para un usuario."""
        try:
            if user_id not in self.user_movie_matrix.index:
                return {
                    'success': False,
                    'error': f'Usuario {user_id} no encontrado'
                }
            
            if self.model_type == 'knn':
                # Obtener el vector de usuario
                user_vector = self.user_movie_matrix.loc[user_id].values
                
                # Encontrar películas no vistas (ratings = 0)
                unwatched_mask = user_vector == 0
                unwatched_indices = np.where(unwatched_mask)[0]
                
                if len(unwatched_indices) == 0:
                    return {
                        'success': False,
                        'error': 'No hay películas nuevas para recomendar'
                    }
                
                # Predecir ratings para películas no vistas
                predictions = []
                for idx in unwatched_indices:
                    pred = self.model.predict([[user_vector[idx]]])[0]
                    predictions.append((idx, pred))
                
                # Ordenar por predicción
                predictions.sort(key=lambda x: x[1], reverse=True)
                
                # Obtener las mejores n recomendaciones
                recommendations = []
                for idx, pred_rating in predictions[:n_recommendations]:
                    movie_id = self.user_movie_matrix.columns[idx]
                    movie_info = self.movies_df[self.movies_df['movieId'] == movie_id].iloc[0]
                    
                    # Obtener estadísticas de la película
                    stats = self.movie_stats[self.movie_stats['movieId'] == movie_id].iloc[0]
                    
                    recommendations.append({
                        'title': movie_info['title'],
                        'genres': movie_info['genres'],
                        'predicted_rating': float(pred_rating),
                        'avg_rating': float(stats['avg_rating']),
                        'rating_count': int(stats['rating_count'])
                    })
                
                return {
                    'success': True,
                    'recommendations': recommendations
                }
                
            elif self.model_type == 'rf':
                # Obtener películas no vistas por el usuario
                user_ratings = self.ratings_df[self.ratings_df['userId'] == user_id]
                watched_movies = set(user_ratings['movieId'])
                all_movies = set(self.movies_df['movieId'])
                unwatched_movies = list(all_movies - watched_movies)
                
                if not unwatched_movies:
                    return {
                        'success': False,
                        'error': 'No hay películas nuevas para recomendar'
                    }
                
                # Crear features para predicción
                pred_data = pd.DataFrame({
                    'userId': [user_id] * len(unwatched_movies),
                    'movieId': unwatched_movies
                })
                X_pred = self._create_features(pred_data)
                
                # Predecir ratings
                predictions = self.model.predict(X_pred)
                
                # Ordenar películas por predicción
                movie_preds = list(zip(unwatched_movies, predictions))
                movie_preds.sort(key=lambda x: x[1], reverse=True)
                
                # Obtener las mejores n recomendaciones
                recommendations = []
                for movie_id, pred_rating in movie_preds[:n_recommendations]:
                    movie_info = self.movies_df[self.movies_df['movieId'] == movie_id].iloc[0]
                    
                    # Obtener estadísticas de la película
                    stats = self.movie_stats[self.movie_stats['movieId'] == movie_id].iloc[0]
                    
                    recommendations.append({
                        'title': movie_info['title'],
                        'genres': movie_info['genres'],
                        'predicted_rating': float(pred_rating),
                        'avg_rating': float(stats['avg_rating']),
                        'rating_count': int(stats['rating_count'])
                    })
                
                return {
                    'success': True,
                    'recommendations': recommendations
                }
            else:
                return {
                    'success': False,
                    'error': f'Modelo {self.model_type} no soportado'
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def save_model(self, path='models'):
        """
        Guarda el modelo entrenado y todos los datos procesados
        """
        if not os.path.exists(path):
            os.makedirs(path)
            
        # Guardar el modelo
        model_path = os.path.join(path, f'{self.model_type}_model.joblib')
        joblib.dump(self.model, model_path)
        
        # Guardar datos procesados
        data = {
            'movies_df': self.movies_df,
            'ratings_df': self.ratings_df,
            'user_means': self.user_means,
            'movie_means': self.movie_means,
            'global_mean': self.global_mean,
            'user_mapper': self.user_mapper,
            'movie_mapper': self.movie_mapper,
            'n_users': self.n_users,
            'n_movies': self.n_movies
        }
        
        if self.model_type == 'knn':
            data['user_movie_matrix'] = self.user_movie_matrix
            
        data_path = os.path.join(path, f'{self.model_type}_data.joblib')
        joblib.dump(data, data_path)
        print(f"Modelo y datos guardados en {path}")
        
    def load_model(self, path='models'):
        """
        Carga un modelo guardado y sus datos procesados
        Returns:
            bool: True si la carga fue exitosa, False si no
        """
        try:
            # Cargar el modelo
            model_path = os.path.join(path, f'{self.model_type}_model.joblib')
            data_path = os.path.join(path, f'{self.model_type}_data.joblib')
            
            if not (os.path.exists(model_path) and os.path.exists(data_path)):
                print("No se encontraron archivos de modelo guardados")
                return False
                
            print("Cargando modelo guardado...")
            self.model = joblib.load(model_path)
            
            # Cargar datos procesados
            print("Cargando datos pre-procesados...")
            data = joblib.load(data_path)
            
            self.movies_df = data['movies_df']
            self.ratings_df = data['ratings_df']
            self.user_means = data['user_means']
            self.movie_means = data['movie_means']
            self.global_mean = data['global_mean']
            self.user_mapper = data['user_mapper']
            self.movie_mapper = data['movie_mapper']
            self.n_users = data['n_users']
            self.n_movies = data['n_movies']
            
            if self.model_type == 'knn':
                self.user_movie_matrix = data['user_movie_matrix']
                
            print("Modelo y datos cargados exitosamente")
            print(f"Usuarios disponibles: {self.n_users}")
            print(f"Películas disponibles: {self.n_movies}")
            return True
            
        except Exception as e:
            print(f"Error cargando el modelo: {str(e)}")
            return False
            
    def initialize(self):
        """
        Inicializa el modelo intentando cargar uno guardado primero
        """
        print(f"Inicializando modelo {self.model_type}...")
        
        # Intentar cargar modelo guardado
        if self.load_model():
            print(f"Modelo {self.model_type} cargado exitosamente!")
            return
            
        # Si no hay modelo guardado, entrenar uno nuevo
        print("No se encontró modelo guardado, entrenando nuevo modelo...")
        self.load_data()
        self.train_model()
        
        # Guardar el modelo entrenado
        print("Guardando modelo entrenado...")
        self.save_model()
        
        print(f"Modelo {self.model_type} listo!")
        
    def recommend(self, user_id, n_recommendations=20):
        """
        Obtiene recomendaciones simples para un usuario
        """
        if user_id not in self.user_means.index:
            return None
            
        result = self.get_recommendations(user_id, n_recommendations)
        if not result['success']:
            return None
            
        # Devolver solo los IDs de las películas recomendadas
        return [rec['movieId'] for rec in result['recommendations']]
