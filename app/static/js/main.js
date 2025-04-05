// Función para mostrar/ocultar pestañas
function showTab(tabName) {
    if (tabName === 'recommendations') {
        document.getElementById('recommendationsTab').style.display = 'block';
        document.getElementById('evaluationTab').style.display = 'none';
    } else {
        document.getElementById('recommendationsTab').style.display = 'none';
        document.getElementById('evaluationTab').style.display = 'block';
        loadMetrics();
    }
}

// Función para cargar métricas
async function loadMetrics() {
    try {
        const response = await fetch('/api/metrics');
        const data = await response.json();
        
        if (data.error) {
            showError(data.error);
            return;
        }
        
        // Actualizar métricas KNN
        document.getElementById('knnMSE').textContent = data.knn.mse.toFixed(4);
        document.getElementById('knnMAE').textContent = data.knn.mae.toFixed(4);
        document.getElementById('knnAccuracy').textContent = data.knn.accuracy.toFixed(1) + '%';
        document.getElementById('knnPrecision').textContent = data.knn.precision.toFixed(1) + '%';
        document.getElementById('knnRecall').textContent = data.knn.recall.toFixed(1) + '%';
        document.getElementById('knnF1').textContent = data.knn.f1.toFixed(1) + '%';
        document.getElementById('knnTime').textContent = data.knn.inference_time.toFixed(2) + 'ms';
        
        // Actualizar métricas RF
        document.getElementById('rfMSE').textContent = data.rf.mse.toFixed(4);
        document.getElementById('rfMAE').textContent = data.rf.mae.toFixed(4);
        document.getElementById('rfAccuracy').textContent = data.rf.accuracy.toFixed(1) + '%';
        document.getElementById('rfPrecision').textContent = data.rf.precision.toFixed(1) + '%';
        document.getElementById('rfRecall').textContent = data.rf.recall.toFixed(1) + '%';
        document.getElementById('rfF1').textContent = data.rf.f1.toFixed(1) + '%';
        document.getElementById('rfTime').textContent = data.rf.inference_time.toFixed(2) + 'ms';
        
        // Actualizar mejor modelo
        document.getElementById('bestMSE').textContent = data.best_model.mse;
        document.getElementById('bestMAE').textContent = data.best_model.mae;
        document.getElementById('bestAccuracy').textContent = data.best_model.accuracy;
        document.getElementById('bestPrecision').textContent = data.best_model.precision;
        document.getElementById('bestRecall').textContent = data.best_model.recall;
        document.getElementById('bestF1').textContent = data.best_model.f1;
        document.getElementById('bestTime').textContent = data.best_model.inference_time;
        
        // Resaltar el mejor modelo en cada métrica
        Object.entries(data.best_model).forEach(([metric, bestModel]) => {
            const knnCell = document.getElementById(`knn${metric.toUpperCase()}`);
            const rfCell = document.getElementById(`rf${metric.toUpperCase()}`);
            
            if (knnCell && rfCell) {
                knnCell.style.fontWeight = bestModel === 'KNN' ? 'bold' : 'normal';
                rfCell.style.fontWeight = bestModel === 'RF' ? 'bold' : 'normal';
            }
        });
            
    } catch (error) {
        showError('Error cargando métricas: ' + error.message);
    }
}

// Función para obtener recomendaciones
async function getRecommendations(userId) {
    try {
        document.getElementById('loading').style.display = 'block';
        document.getElementById('errorMessage').style.display = 'none';
        
        // Obtener recomendaciones KNN
        const knnResponse = await fetch(`/recommend/${userId}?model=knn`);
        const knnData = await knnResponse.json();
        
        // Obtener recomendaciones RF
        const rfResponse = await fetch(`/recommend/${userId}?model=rf`);
        const rfData = await rfResponse.json();
        
        document.getElementById('loading').style.display = 'none';
        
        if (knnData.error || rfData.error) {
            showError(knnData.error || rfData.error);
            return;
        }
        
        // Mostrar recomendaciones KNN
        displayRecommendations('knnRecommendations', knnData.recommendations);
        
        // Mostrar recomendaciones RF
        displayRecommendations('rfRecommendations', rfData.recommendations);
        
    } catch (error) {
        document.getElementById('loading').style.display = 'none';
        showError('Error obteniendo recomendaciones: ' + error.message);
    }
}

// Función para mostrar recomendaciones
function displayRecommendations(containerId, recommendations) {
    const container = document.getElementById(containerId);
    if (!recommendations || recommendations.length === 0) {
        container.innerHTML = '<p>No hay recomendaciones disponibles</p>';
        return;
    }
    
    let html = '';
    recommendations.forEach(movie => {
        const predictedRating = movie.predicted_rating ? movie.predicted_rating.toFixed(2) : '-';
        const averageRating = movie.average_rating ? movie.average_rating.toFixed(2) : '-';
        
        html += `
            <div class="movie-card">
                <div class="movie-title">${movie.title || 'Película ' + movie.movieId}</div>
                <div class="movie-genre">${movie.genres || 'Géneros no disponibles'}</div>
                <div class="movie-rating">
                    Rating predicho: ${predictedRating}
                    <br>
                    Rating promedio: ${averageRating}
                </div>
            </div>
        `;
    });
    
    container.innerHTML = html;
}

// Función para actualizar métricas
function updateMetrics(metrics) {
    document.getElementById('accuracy').textContent = metrics.accuracy ? metrics.accuracy + '%' : '-';
    document.getElementById('precision').textContent = metrics.precision ? metrics.precision + '%' : '-';
    document.getElementById('recall').textContent = metrics.recall ? metrics.recall + '%' : '-';
    document.getElementById('f1').textContent = metrics.f1 ? metrics.f1 + '%' : '-';
}

// Función para mostrar errores
function showError(message) {
    const errorDiv = document.getElementById('errorMessage');
    errorDiv.textContent = message;
    errorDiv.style.display = 'block';
}

// Función para cargar información de usuarios
async function loadUserInfo() {
    try {
        const response = await fetch('/', {
            headers: {
                'Accept': 'application/json'
            }
        });
        const data = await response.json();
        
        if (data.user_info && data.sample_users) {
            // Mostrar rango de usuarios
            document.getElementById('userRange').textContent = 
                `${data.user_info.min_user_id} - ${data.user_info.max_user_id}`;
            
            // Mostrar total de usuarios
            document.getElementById('totalUsers').textContent = data.user_info.total_users;
            
            // Mostrar ejemplos de IDs
            document.getElementById('sampleUsers').textContent = 
                `Ejemplos: ${data.sample_users.join(', ')}`;
        }
    } catch (error) {
        console.error('Error cargando información de usuarios:', error);
        document.getElementById('userInfo').innerHTML = 
            '<p class="text-danger">Error cargando información de usuarios</p>';
    }
}

// Configurar el formulario cuando el documento esté listo
document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('searchForm');
    form.addEventListener('submit', function(e) {
        e.preventDefault();
        const userId = document.getElementById('userId').value;
        getRecommendations(userId);
    });
    
    // Mostrar pestaña de recomendaciones por defecto
    showTab('recommendations');
    
    // Cargar información de usuarios
    loadUserInfo();
}); 