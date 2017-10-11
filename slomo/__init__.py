from os import path

with open(
        path.join(path.split(__file__)[0], "_version.py"),
        encoding="utf-8") as f:
    __version__ = f.readline().split("=")[1].strip("\"' \n")

__all__ = [
    "anisotropy",
    "io",
    "jeans",
    "likelihood",
    "mass",
    "models",
    "parameters",
    "pdf",
    "surface_density",
    "transforms",
    "volume_density"
]

from . import *
from .run import init, sample
from .models import DynamicalModel
