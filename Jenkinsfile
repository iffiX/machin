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
                    steps {
                        bash 'python3 -m pip install virtualenv'
                        bash 'virtualenv venv'
                        PATH = '${env.WORKSPACE}/venv/bin:${env.PATH}'
                        bash '. venv/bin/activate'
                        bash 'pip install -e .'
                    }
                }
            }
        }
    }
}