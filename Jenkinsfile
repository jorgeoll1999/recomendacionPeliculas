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
                    python3 -m venv ${VENV_DIR}
                    . ${VENV_DIR}/bin/activate
                    python -m pip install --upgrade pip setuptools wheel
                    pip install -r requirements.txt
                '''
                
                echo '📂 Preparando directorios de datos...'
                sh '''
                    mkdir -p data
                    mkdir -p models
                '''
            }
        }

        stage('Prepare Data') {
            steps {
                echo '📊 Copiando archivos de datos...'
                sh '''
                    cp -r ./* data/ || true
                    ls -la data/
                '''
            }
        }

        stage('Test') {
            steps {
                echo '✅ Ejecutando pruebas...'
                sh '''
                    . ${VENV_DIR}/bin/activate
                    python -m pytest tests/test_models.py tests/test_load.py tests/test_model.py -v
                '''
            }
        }

        stage('Build Model') {
            steps {
                echo '📦 Entrenando el modelo...'
                sh '''
                    . ${VENV_DIR}/bin/activate
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
