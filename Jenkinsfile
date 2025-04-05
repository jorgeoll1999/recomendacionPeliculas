pipeline {
    agent any
    
    environment {
        PYTHON_VERSION = '3.13'
    }
    
    stages {
        stage('Setup') {
            steps {
                // Crear y activar entorno virtual
                sh '''
                    python -m venv venv
                    . venv/bin/activate
                '''
                
                // Instalar dependencias
                sh '''
                    pip install -r requirements.txt
                    pip install pytest pytest-cov
                '''
            }
        }
        
        stage('Test') {
            steps {
                // Ejecutar tests con cobertura
                sh '''
                    . venv/bin/activate
                    python -m pytest tests/ --cov=app --cov-report=xml -v
                '''
            }
            post {
                always {
                    // Publicar resultados de cobertura
                    cobertura coberturaReportFile: 'coverage.xml'
                }
            }
        }
        
        stage('Build Model') {
            when {
                branch 'main'  // Solo en la rama principal
            }
            steps {
                // Entrenar y guardar el modelo
                sh '''
                    . venv/bin/activate
                    python -c "from app.model import MovieRecommender; recommender = MovieRecommender(); recommender.initialize()"
                '''
            }
        }
    }
    
    post {
        always {
            // Limpiar workspace
            cleanWs()
        }
    }
} 