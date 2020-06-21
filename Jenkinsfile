pipeline {
    agent {
        docker {
            image 'pytorch/pytorch:latest'
            args '-u root:sudo --gpus all'
        }
    }
    environment {
        PYPI_CREDS = credentials('pypi_username_password')
        TWINE_USERNAME = "${env.PYPI_CREDS_USR}"
        TWINE_PASSWORD = "${env.PYPI_CREDS_PSW}"
    }
    stages {
        stage('Install') {
            steps {
                sh 'nvidia-smi' // make sure gpus are loaded
                echo 'Building branch: ${env.BRANCH_NAME}'
                echo 'Building tag: ${env.TAG_NAME}'
                sh 'mkdir ~/.pip && touch ~/.pip/pip.conf'
                sh 'sed -i \'s/http:\\/\\/archive.ubuntu.com/https:\\/\\/mirr' +
                   'ors.tuna.tsinghua.edu.cn/g\' /etc/apt/sources.list'
                sh 'echo \'[global]\' | tee ~/.pip/pip.conf'
                sh 'echo \'index-url = https://pypi.tuna.tsinghua.edu.cn/simp' +
                   'le\' | tee -a ~/.pip/pip.conf'
                sh 'export PIP_DEFAULT_TIMEOUT=100'
                sh 'apt clean'
                sh 'rm -Rf /var/lib/apt/lists/*'
                sh 'apt update'
                sh 'apt install -y freeglut3-dev xvfb fonts-dejavu graphviz'
                sh 'pip install -e .'
                sh 'pip install pytest==5.4.3'
                sh 'pip install pytest-cov==2.10.0'
                sh 'pip install allure-pytest==2.8.16'
                sh 'pip install pytest-xvfb==2.0.0'
                sh 'pip install pytest-html==1.22.1'
                // This line must be included, otherwise matplotlib will
                // segfault when it tries to build the font cache.
                sh "python3 -c 'import matplotlib.pyplot as plt'"
            }
        }
        stage('Test API') {
            steps {
                // run basic test
                sh 'mkdir -p test_results'
                sh 'mkdir -p test_allure_data/api'

                // -eq 1  is used to tell jenkins to not mark
                // the test as failure when sub tests failed.
                sh 'pytest --cov-report term-missing --cov=machin ' +
                   '-k \'not full_train and not Wrapper\' ' +
                   '--junitxml test_results/test_api.xml ./test ' +
                   '--cov-report xml:test_results/cov_report.xml ' +
                   '--html=test_results/test_api.html ' +
                   '--self-contained-html ' +
                   '--alluredir="test_allure_data/api"' +
                   '|| [ $? -eq 1 ]'
                junit 'test_results/test_api.xml'
                archiveArtifacts 'test_results/test_api.html'
                archiveArtifacts 'test_results/cov_report.xml'
            }
            post {
                always {
                    step([$class: 'CoberturaPublisher',
                                   autoUpdateHealth: false,
                                   autoUpdateStability: false,
                                   coberturaReportFile: 'test_results/cov_report.xml',
                                   failNoReports: false,
                                   failUnhealthy: false,
                                   failUnstable: false,
                                   maxNumberOfBuilds: 10,
                                   onlyStable: false,
                                   sourceEncoding: 'ASCII',
                                   zoomCoverageChart: false])
                }
            }
        }
        stage('Test full training') {
            when {
                branch 'releasex'
            }
            steps {
                // run full training test
                sh 'mkdir -p test_results'
                sh 'mkdir -p test_allure_data/full_train'
                sh 'pytest --cov-report term-missing --cov=machin ' +
                   '-k \'full_train and not Wrapper\' ' +
                   '--junitxml test_results/test_full_train.xml ./test' +
                   '--alluredir="test_allure_data/full_train"' +
                   '|| [ $? -eq 1 ]'
                junit 'test_results/test_full_train.xml'
            }
        }
        stage('Deploy allure report') {
            when {
                allOf {
                    branch 'release'
                    tag pattern: 'v\\d+\\.\\d+\\.\\d+(-[a-zA-Z]+)?', comparator: "REGEXP"
                }
            }
            steps {
                // install allure and generate report
                sh 'mkdir -p test_allure_report'
                sh 'wget \'https://bintray.com/qameta/maven/download_file?fil' +
                   'e_path=io%2Fqameta%2Fallure%2Fallure-commandline%2F2.8.1%' +
                   '2Fallure-commandline-2.8.1.tgz'
                sh 'tar -xvzf allure-commandline-2.8.1.tgz'
                sh 'export PATH=allure-2.8.1/bin/:$PATH'
                sh 'allure generate test_allure_data -o test_allure_report'
            }
            post {
                always {
                    // copy this report to the server
                    sshPublisher(publishers: [sshPublisherDesc(
                        configName: 'ci.beyond-infinity.com',
                        transfers: [sshTransfer(
                            cleanRemote: false,
                            excludes: '',
                            execCommand: '',
                            execTimeout: 120000,
                            flatten: false,
                            makeEmptyDirs: false,
                            noDefaultExcludes: false,
                            patternSeparator: '[, ]+',
                            remoteDirectory: "reports/machin/${env.TAG_NAME}",
                            remoteDirectorySDF: false,
                            removePrefix: '',
                            sourceFiles: 'test_allure_report'
                        )],
                        usePromotionTimestamp: false,
                        useWorkspaceInPromotion: false, verbose: false)])
                }
            }
        }
        stage('Deploy PyPI package') {
            when {
                branch 'release'
                allOf {
                    branch 'release'
                    tag pattern: 'v\\d+\\.\\d+\\.\\d+', comparator: "REGEXP"
                }
            }
            steps {
                // build distribution wheel
                sh 'python3 -m pip install twine'
                sh 'python3 setup.py sdist bdist_wheel'
                // upload to twine
                sh 'twine upload dist/*'
            }
            post {
                always {
                    // save results for later check
                    archiveArtifacts (allowEmptyArchive: true,
                                      artifacts: 'dist/*whl',
                                      fingerprint: true)
                }
            }
        }
    }
}