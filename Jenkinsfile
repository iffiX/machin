pipeline {
    agent {
        docker {
            image 'pytorch/pytorch:latest'
            args '-u root:sudo'
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
                        sh 'python3 -m pip install virtualenv'
                        sh 'virtualenv venv'
                        sh '. venv/bin/activate'
                        sh 'pip install -e .'
                    }
                }
            }
        }
    }
}