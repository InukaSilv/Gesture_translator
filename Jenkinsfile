pipeline {
    agent {
        label 'built-in' 
    }
    
    tools {
        'org.jenkinsci.plugins.shiningpanda.tools.PythonInstallation' 'Python-3.12'
    }
    
    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }
        
        stage('Setup Python') {
            steps {
                bat """
                    python --version
                    python -m venv venv
                    call venv\\Scripts\\activate
                """
            }
        }
        
        stage('Install Dependencies') {
            steps {
                bat """
                    call venv\\Scripts\\activate
                    pip install --upgrade pip
                    pip install -r requirements.txt
                """
            }
        }
        
        stage('Train Model') {
            steps {
                bat """
                    call venv\\Scripts\\activate
                    python train_model.py
                """
            }
        }
        
        stage('Archive Model') {
            steps {
                archiveArtifacts artifacts: 'asl_model.h5', fingerprint: true
            }
        }
    }
}