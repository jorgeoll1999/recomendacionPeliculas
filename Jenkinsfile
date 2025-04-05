pipeline {
    agent any

    environment {
        VENV_DIR = 'venv'
    }

    stages {
        stage('Setup') {
            steps {
                echo '🛠️ Creando entorno virtual...'
                sh '''
                    python3 -m venv venv
                    . venv/bin/activate
                    pip install -r requirements.txt
                    mkdir -p data models
                '''
            }
        }

        stage('Prepare Data') {
            steps {
                echo '📊 Copiando archivos de datos...'
                sh '''
                    cp -r data/* data/
                    ls -la data/
                '''
            }
        }

        stage('Test') {
            steps {
                echo '✅ Ejecutando pruebas...'
                sh '''
                    . venv/bin/activate
                    python -m pytest tests/ -v
                '''
            }
        }

        stage('Build Model') {
            steps {
                echo '📦 Entrenando el modelo...'
                sh '''
                    . venv/bin/activate
                    python src/train_model.py
                '''
            }
        }
    }

    post {
        always {
            echo '🧹 Limpiando workspace...'
            cleanWs()
        }
        success {
            echo '✨ Pipeline completado exitosamente!'
        }
        failure {
            echo '❌ Pipeline falló!'
        }
    }
}
