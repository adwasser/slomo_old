from setuptools import setup
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, "slomo/_version.py"), encoding="utf-8") as f:
    # sets version
    line = f.readline()
    __version__ = line.split("=")[1].strip("\"' \n")

with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="slomo",
    version=__version__,
    long_description=long_description,
    description="Dynamical modeling of elliptical galaxies",
    url="https://github.com/adwasser/slomo",
    author="AsherWasserman",
    author_email="adwasser@ucsc.edu",
    license="MIT",
    packages=["slomo"],
    install_requires=[
        "numpy", "scipy", "astropy", "emcee", "ruamel.yaml", "h5py", "dill",
        "multiprocess", "psutil", "colossus"
    ],
    include_package_data=True,
    scripts=["bin/slomo"])
