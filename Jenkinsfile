pipeline {
    agent {
        docker {
            image 'pytorch/pytorch:latest'
            args '-u root:sudo --gpus all'
        }
    }
    stages {
        stage('Install') {
            steps {
                sh 'nvidia-smi' // make sure gpus are loaded
                sh "mkdir ~/.pip && touch ~/.pip/pip.conf"
                sh "sed -i 's/archive.ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list"
                sh "echo '[global]' | tee ~/.pip/pip.conf"
                sh "echo 'index-url = https://pypi.tuna.tsinghua.edu.cn/simple' | tee -a ~/.pip/pip.conf"
                sh 'apt clean'
                sh 'rm -Rf /var/lib/apt/lists/*'
                sh 'apt update'
                //sh 'apt install -y freeglut3-dev xvfb fonts-dejavu'
                sh 'apt install -y freeglut3-dev ubuntu-desktop'
                sh 'pip install -e .'
                sh 'pip install pytest==5.4.3'
                sh 'pip install pytest-cov==2.10.0'
                sh 'pip install allure-pytest==2.8.16'
                sh 'pip install pytest-xvfb==2.0.0'
            }
        }
        stage('Test basic API') {
            steps {
                // run basic test
                sh 'mkdir -p test_results'

                // no multiline string here, will execute pytest without args
                // and will cause seg fault.
                sh "pytest --cov-report term-missing --cov=machin -k 'not full_train' --junitxml test_results/test_basic_api.xml ./test"
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
                sh "pytest --cov-report term-missing --cov=machin -k 'full_train' --junitxml test_results/test_full_train.xml ./test"
                junit 'test_results/test_full_train.xml'
            }
        }
        stage('Deploy allure report') {
            when {
                branch 'release'
            }
            steps {
                // install allure and generate report
                sh "wget 'https://bintray.com/qameta/maven/download_file?file_path=io%2Fqameta%2Fallure%2Fallure-commandline%2F2.8.1%2Fallure-commandline-2.8.1.tgz'"
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