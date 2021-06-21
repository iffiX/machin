.. Machin documentation master file, created by
   sphinx-quickstart on Mon Jun  1 14:47:38 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

==================
Welcome
==================
Welcome to the main documentation of **Machin** library.

----

About
++++++++++++++++++
.. toctree::
   :maxdepth: 1

   about.rst

Installation
++++++++++++++++++
Machin is hosted on `PyPI <https://pypi.org/project/tianshou/>`_, currently it
requires:

1. python >= 3.6
2. torch >= 1.6.0

If you are using PIP to manage your python packages, you may directly type::

   pip install machin

If you are using conda to manage your python packages, you are suggested to create a
virtual environment first, to prevent PIP changes your packages without letting
conda know::

   conda create -n some_env pip
   conda activate some_env
   pip install machin

Tutorials and examples
++++++++++++++++++++++
.. toctree::
   :maxdepth: 1

   tutorials/index.rst
   advance/index.rst


API
++++++++++++++++++
.. toctree::
   :maxdepth: 2

   api/index.rst


Indices and tables
++++++++++++++++++

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
