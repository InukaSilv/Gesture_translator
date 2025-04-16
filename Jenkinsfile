pipeline {
    agent any
    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }
        stage('Setup Python') {
            steps {
                sh 'python -m venv venv'
                sh 'source venv/bin/activate'
            }
        }
        stage('Install Dependencies') {
            steps {
                sh 'pip install -r requirements.txt'
            }
        }
        stage('Train Model') {
            steps {
                sh 'python train_model.py'
            }
        }
        stage('Archive Model') {
            steps {
                archiveArtifacts artifacts: 'asl_model.h5', fingerprint: true
            }
        }
    }
}