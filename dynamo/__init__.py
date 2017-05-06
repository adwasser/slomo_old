from os import path

with open(path.join(path.split(__file__)[0], "_version.py"),
          encoding="utf-8") as f:
    __version__ = f.readline().split("=")[1].strip("\"' \n")

from . import *
from .run import *
from .models import DynamicalModel
