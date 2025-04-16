pipeline {
    agent any
    
    tools {
        python "Python-3.12" 
    }
    
    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }
        
        stage('Setup Python') {
            steps {
                script {
                    // Windows
                    bat """
                        python -m venv venv
                        call venv\\Scripts\\activate
                        python --version
                    """
                    
                    // Linux (alternative)
                    // sh '''
                    //     python3 -m venv venv
                    //     . venv/bin/activate
                    //     python --version
                    // '''
                }
            }
        }
        
        stage('Install Dependencies') {
            steps {
                bat """
                    call venv\\Scripts\\activate
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