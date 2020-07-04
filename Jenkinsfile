pipeline {
    agent {
        docker {
            image 'pytorch/pytorch:latest'
            args '-u root:sudo --gpus all --network=host'
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
                echo "Building branch: ${env.BRANCH_NAME}"
                echo "Building tag: ${env.TAG_NAME}"
                sh 'mkdir ~/.pip && touch ~/.pip/pip.conf'
                sh 'sed -i -r \'s|deb http://.+/ubuntu| deb [arch=amd64] http://apt.mirror.node2/ubuntu|g\' /etc/apt/sources.list'
                sh '''
                   echo '
                   [global]
                   index-url = http://pypi.mirror.node2/root/pypi/+simple/
                   trusted-host = pypi.mirror.node2
                   [search]
                   index = http://pypi.mirror.node2/root/pypi/
                   ' | tee ~/.pip/pip.conf'''
                sh 'apt clean'
                sh 'apt update'
                sh 'apt -o APT::Acquire::Retries="3" install -y wget freeglut3-dev xvfb fonts-dejavu graphviz'
                sh 'pip install -e .'
                sh 'pip install mock pytest==5.4.3 pytest-cov==2.10.0 allure-pytest==2.8.16 pytest-xvfb==2.0.0 pytest-html==1.22.1'
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
                   '-k \'not full_train\' ' +
                   '-o junit_family=xunit1 ' +
                   '--junitxml test_results/test_api.xml ./test ' +
                   '--cov-report xml:test_results/cov_report.xml ' +
                   '--html=test_results/test_api.html ' +
                   '--self-contained-html ' +
                   '--alluredir="test_allure_data/api"' +
                   '|| export test_result="failed"'
                script {
                    test_api = sh(returnStdout: true, script: 'echo $test_result').trim()
                }
                echo "$test_api"
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
                anyOf {
                    branch 'release'
                    tag pattern: 'v\\d+\\.\\d+\\.\\d+(-[a-zA-Z]+)?', comparator: "REGEXP"
                }
            }
            steps {
                // run full training test
                sh 'mkdir -p test_results'
                sh 'mkdir -p test_allure_data/full_train'
                sh 'pytest ' +
                   '-k \'full_train\' ' +
                   '-o junit_family=xunit1 ' +
                   '--junitxml test_results/test_full_train.xml ./test ' +
                   '--html=test_results/test_full_train.html ' +
                   '--self-contained-html ' +
                   '--alluredir="test_allure_data/full_train"' +
                   '|| export test_result="failed"'
                script {
                    test_full_train = sh(returnStdout: true, script: 'echo $test_result').trim()
                }
                echo "$test_full_train"
                junit 'test_results/test_full_train.xml'
                archiveArtifacts 'test_results/test_full_train.html'
            }
        }
        stage('Deploy allure report') {
            when {
                anyOf {
                    branch 'release'
                    tag pattern: 'v\\d+\\.\\d+\\.\\d+(-[a-zA-Z]+)?', comparator: "REGEXP"
                }
            }
            steps {
                // install allure and generate report
                sh 'mkdir -p test_allure_report'
                sh 'apt install -y default-jre'
                sh 'wget -O allure-commandline-2.8.1.tgz ' +
                   '\'https://bintray.com/qameta/maven/download_file?fil' +
                   'e_path=io%2Fqameta%2Fallure%2Fallure-commandline%2F2.8.1%' +
                   '2Fallure-commandline-2.8.1.tgz\''
                sh 'tar -xvzf allure-commandline-2.8.1.tgz'
                sh 'chmod a+x allure-2.8.1/bin/allure'
                sh 'allure-2.8.1/bin/allure generate test_allure_data/api ' +
                   'test_allure_data/full_train -o test_allure_report'
            }
            post {
                always {
                    // clean up remote directory and copy this report to the server
                    sshPublisher(publishers: [sshPublisherDesc(
                        configName: 'ci.beyond-infinity.com',
                        transfers: [sshTransfer(
                            cleanRemote: true,
                            excludes: '',
                            execCommand: '',
                            execTimeout: 120000,
                            flatten: false,
                            makeEmptyDirs: false,
                            noDefaultExcludes: false,
                            patternSeparator: '[, ]+',
                            remoteDirectory: "reports/machin/${env.TAG_NAME}/",
                            remoteDirectorySDF: false,
                            removePrefix: 'test_allure_report/', // remove prefix
                            sourceFiles: 'test_allure_report/**/*' // recursive copy
                        )],
                        usePromotionTimestamp: false,
                        useWorkspaceInPromotion: false, verbose: false)])
                }
            }
        }
        stage('Deploy PyPI package') {
            when {
                allOf {
                    // only version tags without postfix will be deployed
                    tag pattern: 'v\\d+\\.\\d+\\.\\d+', comparator: "REGEXP"
                    expression { test_api != "failed" && test_full_train != "failed" }
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
    post {
        always {
            // clean up workspace
            sh 'rm -rf ./*'
        }
    }
}