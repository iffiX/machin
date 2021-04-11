from os import path
import re
import setuptools

root = path.abspath(path.dirname(__file__))

with open(path.join(root, "machin", "__init__.py")) as f:
    version = re.search(r"__version__ = \"(.*?)\"", f.read()).group(1)

with open("README.md", mode="r", encoding="utf8") as desc:
    long_description = desc.read()

setuptools.setup(
    name="machin",
    version=version,
    description="Reinforcement learning library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/iffiX/machin",
    author="Iffi",
    author_email="iffi@mail.beyond-infinity.com",
    license="MIT",
    python_requires=">=3.5",
    packages=setuptools.find_packages(
        exclude=[
            "test",
            "test.*",
            "test_lib",
            "test_lib.*",
            "examples",
            "examples.*",
            "docs",
            "docs.*",
        ]
    ),
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        "Development Status :: 4 - Beta",
        # Indicate who your project is intended for
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        # Pick your license as you wish (should match "license" above)
        "License :: OSI Approved :: MIT License",
        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    install_requires=[
        "gym",
        "psutil",
        "numpy",
        "torch>=1.6.0",
        "pytorch-lightning>=1.2.0",
        "torchviz",
        "moviepy",
        "matplotlib",
        "colorlog",
        "dill",
        "GPUtil",
        "Pillow",
        "tensorboardX",
    ],
)
