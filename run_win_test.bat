@ECHO OFF
python --version
python -m pip install virtualenv
virtualenv venv
.\venv\Scripts\activate
pip install .
pip install .\test_lib\multiagent-particle-envs\
pip install mock pytest==6.0.0 pytest-html==1.22.1 pytest-repeat==0.8.0
python -m pytest -s --assert=plain -k "not full_train" --html=test_results\test_api.html --self-contained-html .\test\