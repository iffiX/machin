#!/bin/bash
# if you are using any non-standard python 3 installation (>= 3.6)
# please replace first three python with your absolute path
python --version
python -m pip install virtualenv
python -m virtualenv venv
source venv/bin/activate
pip install .
pip install ./test_lib/multiagent-particle-envs/
pip install mock pytest==6.0.1 pytest-html==1.22.1 pytest-repeat==0.8.0
python -m pytest -s --assert=plain -k "not full_train" --html=test_api.html --self-contained-html ./test/