pipeline {
    agent {
        docker {
            image 'pytorch/pytorch:latest'
            args '-u root:sudo'
        }
    }
    stages {
        stage('Install') {
            steps {
                sh "sed -i \'s/archive.ubuntu.com/mirrors.ustc.edu.cn/g\'" +
                   " /etc/apt/sources.list"
                sh 'apt update'
                sh 'apt install -y freeglut3-dev xvfb'
                sh 'python3 -m pip install virtualenv'
                sh 'virtualenv venv'
                sh '. venv/bin/activate'
                sh 'pip install -e .'
                sh 'pip install pytest==5.4.3'
                sh 'pip install pytest-cov==2.10.0'
                sh 'pip install allure-pytest==2.8.16'
            }
        }
        stage('Test basic API') {
            steps {
                // run basic test
                sh 'mkdir -p test_results'
                sh '''xvfb-run -s "-screen 0 1400x900x24" pytest
                      --cov-report term-missing
                      --cov=machin
                      -k "not full_train"
                      --junitxml test_results/test_basic_api.xml
                      ./test'''
                junit 'test_results/test_basic_api.xml'
            }
        }
        stage('Test full training') {
            when {
                branch 'release'
            }
            steps {
                // run full training test
                sh 'mkdir -p test_results'
                sh '''pytest
                      --cov-report term-missing
                      --cov=machin
                      -k "full_train"
                      --junitxml test_results/test_full_train.xml
                      ./test'''
                junit 'test_results/test_full_train.xml'
            }
        }
        stage('Deploy allure report') {
            when {
                branch 'release'
            }
            steps {
                // install allure and generate report
                sh "wget \"https://bintray.com/qameta/maven/download_file?" +
                   "file_path=io%2Fqameta%2Fallure%2Fallure-commandline%" +
                   "2F2.8.1%2Fallure-commandline-2.8.1.tgz\""
                sh 'tar -xvzf allure-commandline-2.8.1.tgz'
                sh 'export PATH=allure-2.8.1/bin/:$PATH'
            }
        }
        stage('Deploy PyPI package') {
            when {
                branch 'release'
            }
            steps {
                echo 'deploy'
            }
        }
    }
}