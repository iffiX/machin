pipeline {
    agent {
        docker {
            image 'pytorch/pytorch:latest'
        }
    }
    stages {
        stage('Master branch test') {
            when {
                branch 'master'
            }
            stages {
                stage('install') {
                    bash 'python3 -m pip install virtualenv'
                    virtualenv venv
                    PATH=$WORKSPACE/venv/bin:$PATH
                    . venv/bin/activate
                    pip install -e .
                }
            }
        }
    }
}