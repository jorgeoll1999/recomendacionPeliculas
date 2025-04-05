import sys
import os
import time
from flask import Flask, jsonify, request, render_template
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import pandas as pd
from flask_cors import CORS

# Agregar el directorio raíz al path de Python
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from .kafka_io import enviar_recomendacion_kafka
from .model import MovieRecommender

app = Flask(__name__)
CORS(app)

print("Inicializando sistema de recomendación...")

# Inicializar modelos
print("Cargando modelo KNN...")
knn_recommender = MovieRecommender(model_type='knn')
knn_recommender.initialize()

print("Cargando modelo Random Forest...")
rf_recommender = MovieRecommender(model_type='rf')
rf_recommender.initialize()

print("Sistema listo!")

def evaluate_model(model, model_type):
    try:
        start_time = time.time()
        
        if model_type == 'knn':
            # Evaluar KNN con una muestra pequeña
            sample_users = np.random.choice(
                list(knn_recommender.user_mapper.keys()), 
                size=min(50, len(knn_recommender.user_mapper)), 
                replace=False
            )
            
            predictions = []
            true_ratings = []
            
            for user_id in sample_users:
                user_ratings = knn_recommender.ratings_df[knn_recommender.ratings_df['userId'] == user_id]
                if len(user_ratings) > 0:
                    sample_size = min(5, len(user_ratings))
                    sample_ratings = user_ratings.sample(n=sample_size)
                    
                    for _, rating in sample_ratings.iterrows():
                        true_ratings.append(rating['rating'])
                        
                        if user_id in knn_recommender.user_mapper:
                            user_idx = knn_recommender.user_mapper[user_id]
                            user_vector = knn_recommender.user_movie_matrix[user_idx]
                            
                            distances, indices = model.kneighbors(user_vector, n_neighbors=5)
                            similar_user_indices = indices[0][1:]
                            
                            similar_users = [list(knn_recommender.user_mapper.keys())[list(knn_recommender.user_mapper.values()).index(idx)] for idx in similar_user_indices]
                            similar_ratings = knn_recommender.ratings_df[
                                (knn_recommender.ratings_df['userId'].isin(similar_users)) & 
                                (knn_recommender.ratings_df['movieId'] == rating['movieId'])
                            ]['rating']
                            
                            if len(similar_ratings) > 0:
                                pred = similar_ratings.mean()
                            else:
                                pred = knn_recommender.ratings_df['rating'].mean()
                            
                            predictions.append(pred)
            
        else:  # Random Forest
            # Usar una muestra pequeña para evaluación
            test_size = min(1000, len(rf_recommender.ratings_df))
            test_sample = rf_recommender.ratings_df.sample(n=test_size)
            
            X_test = rf_recommender._create_features(test_sample)
            true_ratings = test_sample['rating'].values
            predictions = model.predict(X_test)
        
        inference_time = (time.time() - start_time) * 1000
        
        if len(predictions) == 0:
            return {
                'mse': 0,
                'mae': 0,
                'accuracy': 0,
                'precision': 0,
                'recall': 0,
                'f1': 0,
                'inference_time': inference_time
            }
        
        predictions = np.array(predictions)
        true_ratings = np.array(true_ratings)
        
        # Calcular métricas básicas
        mse = mean_squared_error(true_ratings, predictions)
        mae = mean_absolute_error(true_ratings, predictions)
        
        # Calcular métricas de clasificación (redondeando predicciones al entero más cercano)
        pred_rounded = np.round(predictions)
        true_rounded = np.round(true_ratings)
        
        # Calcular accuracy (exactitud)
        accuracy = np.mean(pred_rounded == true_rounded) * 100
        
        # Calcular precision por clase y promediar
        precision_scores = []
        recall_scores = []
        for rating in range(1, 6):
            true_pos = np.sum((pred_rounded == rating) & (true_rounded == rating))
            false_pos = np.sum((pred_rounded == rating) & (true_rounded != rating))
            false_neg = np.sum((pred_rounded != rating) & (true_rounded == rating))
            
            if true_pos + false_pos > 0:
                precision_scores.append(true_pos / (true_pos + false_pos))
            if true_pos + false_neg > 0:
                recall_scores.append(true_pos / (true_pos + false_neg))
        
        precision = np.mean(precision_scores) * 100 if precision_scores else 0
        recall = np.mean(recall_scores) * 100 if recall_scores else 0
        
        # Calcular F1-Score
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'mse': round(mse, 4),
            'mae': round(mae, 4),
            'accuracy': round(accuracy, 1),
            'precision': round(precision, 1),
            'recall': round(recall, 1),
            'f1': round(f1, 1),
            'inference_time': round(inference_time, 2)
        }
    except Exception as e:
        print(f"Error en evaluate_model: {str(e)}")
        return {
            'mse': 0,
            'mae': 0,
            'accuracy': 0,
            'precision': 0,
            'recall': 0,
            'f1': 0,
            'inference_time': 0,
            'error': str(e)
        }

@app.route('/')
def home():
    if request.headers.get('Accept', '').find('application/json') != -1:
        # Si se solicita JSON, devolver la respuesta API
        valid_users = sorted(knn_recommender.user_means.index.tolist())
        sample_users = valid_users[:10]  # Mostrar los primeros 10 usuarios como ejemplo
        
        return jsonify({
            'status': 'ok',
            'message': 'Sistema de recomendación de películas funcionando',
            'endpoints': {
                'recommendations': '/recommend/<user_id>?model=knn|rf'
            },
            'user_info': {
                'total_users': len(valid_users),
                'min_user_id': min(valid_users),
                'max_user_id': max(valid_users)
            },
            'sample_users': sample_users
        })
    else:
        # Si se solicita HTML, devolver la página web
        return render_template('index.html')

@app.route('/recommend/<int:user_id>')
def recommend(user_id):
    model_type = request.args.get('model', 'knn')
    
    if model_type not in ['knn', 'rf']:
        return jsonify({
            'error': 'Modelo no válido. Use "knn" o "rf"'
        }), 400
    
    try:
        model = knn_recommender if model_type == 'knn' else rf_recommender
        result = model.get_recommendations(user_id, n_recommendations=20)
        
        if not result['success']:
            return jsonify({
                'error': result['error']
            }), 404
            
        # Obtener métricas calculadas
        metrics = evaluate_model(model.model, model_type)
        
        # Enviar recomendaciones a Kafka
        try:
            enviar_recomendacion_kafka(user_id, {
                'model': model_type,
                'recommendations': result['recommendations'],
                'metrics': metrics
            })
        except Exception as kafka_error:
            print(f"Error enviando a Kafka: {str(kafka_error)}")
            # Continuamos aunque falle Kafka
            
        return jsonify({
            'user_id': user_id,
            'model': model_type,
            'recommendations': result['recommendations'],
            'metrics': metrics
        })
        
    except Exception as e:
        print(f"Error en recommend: {str(e)}")
        return jsonify({
            'error': str(e)
        }), 500

@app.route("/metrics/<model_type>")
def get_metrics(model_type):
    try:
        if model_type == 'knn':
            recommender = knn_recommender
        else:
            recommender = rf_recommender
            
        # Obtener una muestra de usuarios para evaluar
        sample_users = np.random.choice(
            list(recommender.user_means.index),
            size=min(50, len(recommender.user_means)),
            replace=False
        )
        
        all_metrics = []
        for user_id in sample_users:
            result = recommender.get_recommendations(user_id)
            if result['success'] and result['metrics']:
                all_metrics.append(result['metrics'])
        
        if not all_metrics:
            return jsonify({
                'error': 'No se pudieron calcular métricas',
                'metrics': {}
            }), 404
        
        # Calcular promedio de métricas
        avg_metrics = {}
        for metric in all_metrics[0].keys():
            avg_metrics[metric] = round(
                sum(m[metric] for m in all_metrics) / len(all_metrics),
                3
            )
            
        return jsonify(avg_metrics)
        
    except Exception as e:
        print(f"Error calculando métricas: {str(e)}")
        return jsonify({
            'error': str(e),
            'metrics': {}
        }), 500

@app.route('/api/metrics')
def get_all_metrics():
    try:
        # Obtener métricas para ambos modelos
        knn_metrics = evaluate_model(knn_recommender.model, 'knn')
        rf_metrics = evaluate_model(rf_recommender.model, 'rf')
        
        # Determinar el mejor modelo para cada métrica
        best_model = {
            'mse': 'KNN' if knn_metrics['mse'] < rf_metrics['mse'] else 'RF',
            'mae': 'KNN' if knn_metrics['mae'] < rf_metrics['mae'] else 'RF',
            'accuracy': 'KNN' if knn_metrics['accuracy'] > rf_metrics['accuracy'] else 'RF',
            'precision': 'KNN' if knn_metrics['precision'] > rf_metrics['precision'] else 'RF',
            'recall': 'KNN' if knn_metrics['recall'] > rf_metrics['recall'] else 'RF',
            'f1': 'KNN' if knn_metrics['f1'] > rf_metrics['f1'] else 'RF',
            'inference_time': 'KNN' if knn_metrics['inference_time'] < rf_metrics['inference_time'] else 'RF'
        }
        
        return jsonify({
            'knn': knn_metrics,
            'rf': rf_metrics,
            'best_model': best_model
        })
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8082, debug=True)

