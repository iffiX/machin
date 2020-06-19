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
                    environment {
                        PATH = '${env.WORKSPACE}/venv/bin:${env.PATH}'
                    }
                    steps {
                        bash 'python3 -m pip install virtualenv'
                        bash 'virtualenv venv'
                        bash '. venv/bin/activate'
                        bash 'pip install -e .'
                    }
                }
            }
        }
    }
}