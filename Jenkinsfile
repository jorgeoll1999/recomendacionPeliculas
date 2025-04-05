pipeline {
    agent any

    environment {
        VENV_DIR = 'venv'
    }

    stages {
        stage('Setup') {
            steps {
                echo '🛠️ Creando entorno virtual...'
                bat '''
                    python -m venv %VENV_DIR%
                    call %VENV_DIR%\\Scripts\\activate.bat
                    python -m pip install --upgrade pip setuptools wheel
                    pip install -r requirements.txt
                '''
            }
        }

        stage('Test') {
            steps {
                echo '✅ Ejecutando pruebas...'
                bat '''
                    call %VENV_DIR%\\Scripts\\activate.bat
                    python -m pytest tests\\test_models.py tests\\test_load.py tests\\test_model.py -v
                '''
            }
        }

        stage('Build Model') {
            steps {
                echo '📦 Entrenando el modelo...'
                bat '''
                    call %VENV_DIR%\\Scripts\\activate.bat
                    python src\\train_model.py
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
