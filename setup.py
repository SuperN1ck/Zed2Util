"""Install script for setuptools."""

import setuptools
from os import path

# read the contents of your README file
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setuptools.setup(
    name="Zed2Utils",
    version="0.0.1",
    author="Nick Heppert",
    author_email="heppert@cs.uni-freiburg.de",
    install_requires=[
        "tyro",
        "opencv-python",
        "pynput",
    ],
    description="Zed2Utils Package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    # license="MIT",
    # url="https://github.com/chisarie/jax-agents",
    # packages=["THOR"],
    packages=setuptools.find_packages(),
    # classifiers=[
    #     "Programming Language :: Python :: 3",
    #     "License :: OSI Approved :: MIT License",
    #     "Operating System :: OS Independent",
    # ],
    python_requires=">=3.7",
)
