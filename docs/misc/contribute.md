### Contribute to Machin

#### Prepare your editing environment
---
`virtualenv` package facilitates local editing, you should install it first:
```
python3 -m pip install virtualenv
``` 
Then you should clone the repository and start a new virtual environment named
in the root directory using:
```
git clone https://github.com/iffiX/machin.git
cd machin
virtualenv --no-site-packages venv
```
Finally you can switch to this new virtual environment and install the Machin 
library in local edit mode, in this way, all edits on Machin files will be 
effective immediately.
```
source venv/bin/activate
pip3 install -e .
```

#### Polish your code
All code in Machin must be **readable**, therefore we require you write your
code in the [google python style](http://google.github.io/styleguide/pyguide.html).

You also must document your code, we require you to give detailed signatures 
using the `typing` builtin library for all of your arguments as well as 
keyword arguments. A great example is the 
[google style docstring](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html)
from napoleon, an extension of sphinx the doc builder.

Finally, after so much hard work, do not forget to use pylint to check your code, 
`PEP8` style must be conformed.

#### Test your code
The great test at last! you can run the following command to run all existing
tests coming along with the Machin library. This test command also includes 
training networks.
```
pytest test --conv machin --capture=no --durations=0 -v
```
If you just want to test(train) a specific algorithm, use `-k` option, name is
`<algorithm_class> and full_train`:
```
pytest test -k "DDPG and full_train"
```
Or exclude all training by using:
```
pytest test -k "not full_train"
```

You should group your tests in a single class, like the example given by pytest,
or insert your test cases into a existing test class:
```
# content of test_class.py
class TestClass:
    def test_one(self):
        x = "this"
        assert "h" in x

    def test_two(self):
        x = "hello"
        assert hasattr(x, "check")
```
#### Submit a pull request
Submit a pull request in the end! We truly appreciate your help,
Travis will automatically test your code and we will review your code as soon as possible.

#### Build the documents
In order to build the documents, you must install `requirements.txt` in `/docs`
in your venv, and execute:
```
cd docs
make html
```
This command will build your documents in `docs/build`.