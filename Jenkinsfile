import hudson.model.*
import jenkins.model.*
import hudson.tasks.test.AbstractTestResultAction

pipeline {
    agent {
        node {
            label 'node1'
        }
    }
    environment {
        PYPI_CREDS = credentials('pypi_username_password')
        TWINE_USERNAME = "${env.PYPI_CREDS_USR}"
        TWINE_PASSWORD = "${env.PYPI_CREDS_PSW}"
        // See https://github.com/pytorch/pytorch/issues/37377
        MKL_SERVICE_FORCE_INTEL = "1"
    }
    stages {
        stage("All") {
            agent {
                docker {
                    image 'pytorch/pytorch:1.8.1-cuda10.2-cudnn7-runtime'
                    // host network is needed because mirrors are on the lan and resolved with /etc/hosts
                    args '-u root:sudo --shm-size=1024m --gpus all --network=host'
                    reuseNode true
                }
            }

            stages {
                stage('Install') {
                    steps {
                        sh 'nvidia-smi' // make sure gpus are loaded
                        echo "Building branch: ${env.BRANCH_NAME}"
                        echo "Building tag: ${env.TAG_NAME}"
                        sh 'mkdir -p ~/.pip && touch ~/.pip/pip.conf'
                        sh 'sed -i -r \'s|deb http://.+/ubuntu| deb [arch=amd64] http://apt.mirror.node2/ubuntu|g\' /etc/apt/sources.list'
                        sh '''
                           echo '
                           [global]
                           index-url = http://pypi.mirror.node2/root/pypi/+simple/
                           trusted-host = pypi.mirror.node2
                           [search]
                           index = http://pypi.mirror.node2/root/pypi/
                           ' | tee ~/.pip/pip.conf'''
                        retry(count: 3) {
                            sh 'apt clean'
                            sh 'apt update'
                            sh 'apt -o APT::Acquire::Retries="3" --fix-missing install -y wget freeglut3-dev xvfb fonts-dejavu graphviz cmake g++ swig'
                            // Instead of installing in development mode, we use normal to install
                            // So that import errors can be detected.
                            // sh 'pip install -e .'
                            sh 'pip install .'
                            sh 'pip install ./test_lib/multiagent-particle-envs/'
                            sh 'pip install "gym[atari, box2d, classic_control]"'
                            sh 'pip install mock pytest==6.0.1 pytest-cov==2.10.0 allure-pytest==2.8.16 pytest-xvfb==2.0.0 pytest-html==1.22.1 pytest-repeat==0.8.0'
                            // This line must be included, otherwise matplotlib will
                            // segfault when it tries to build the font cache.
                            sh "python3 -c 'import matplotlib.pyplot as plt'"
                        }
                    }
                }
                stage('Test API') {
                    steps {
                        dir('test_api') {
                            // run basic test
                            sh 'ln -s ../test test'
                            sh 'mkdir -p test_results'
                            sh 'mkdir -p test_allure_data'
                            // -eq 1  is used to tell jenkins to not mark
                            // the test as failure when sub tests failed.
                            sh 'python -m pytest ' +
                               '-s --gpu_device="cuda:1" --assert=plain ' +
                               '--cov-report term-missing --cov=machin ' +
                               '-k \'not full_train\' ' +
                               '-o junit_family=xunit1 ' +
                               '--junitxml test_results/test_api.xml ' +
                               '--cov-report xml:test_results/cov_report.xml ' +
                               '--html=test_results/test_api.html ' +
                               '--self-contained-html ' +
                               '--alluredir="test_allure_data" ' +
                               '. || [ $? -eq 1 ]'
                            junit 'test_results/test_api.xml'
                            archiveArtifacts 'test_results/test_api.html'
                            archiveArtifacts 'test_results/cov_report.xml'
                        }
                    }
                    post {
                        always {
                            step([$class: 'CoberturaPublisher',
                                           autoUpdateHealth: false,
                                           autoUpdateStability: false,
                                           coberturaReportFile: 'test_api/test_results/cov_report.xml',
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
                        dir('test_full_train') {
                            // run full training test
                            sh 'ln -s ../test test'
                            sh 'mkdir -p test_results'
                            sh 'mkdir -p test_allure_data'
                            sh 'python -m pytest ' +
                               '-s --gpu_device="cuda:1" --assert=plain -k \'full_train\' ' +
                               '-o junit_family=xunit1 ' +
                               '--junitxml test_results/test_full_train.xml ' +
                               '--html=test_results/test_full_train.html ' +
                               '--self-contained-html ' +
                               '--alluredir="test_allure_data" ' +
                               '. || [ $? -eq 1 ]'
                            junit 'test_results/test_full_train.xml'
                            archiveArtifacts 'test_results/test_full_train.xml'
                            archiveArtifacts 'test_results/test_full_train.html'
                        }
                    }
                }
                stage('Check test result') {
                    steps {
                        script {
                            def test_result_action = currentBuild.rawBuild.getAction(AbstractTestResultAction.class)
                            test_passed = true
                            if (test_result_action != null) {
                                test_passed = test_result_action.getFailCount() == 0
                            }
                            if (test_passed) {
                                println "Test passed"
                            }
                            else {
                                println "Test failed"
                            }
                        }
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
                           '\'http://file.node2/allure-commandline-2.8.1.tgz\''
                        sh 'tar -xvzf allure-commandline-2.8.1.tgz'
                        sh 'chmod a+x allure-2.8.1/bin/allure'
                        sh 'allure-2.8.1/bin/allure generate test_api/test_allure_data ' +
                           'test_full_train/test_allure_data -o test_allure_report'
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
                            expression { test_passed }
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
    }
}