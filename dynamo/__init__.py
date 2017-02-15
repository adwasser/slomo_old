from os import path

here = path.abspath(path.dirname(__file__))
with open(path.join(here, "../_version.py"), encoding="utf-8") as f:
    # sets version
    exec(f.readline())

from . import *
from .parser import read
from .run import sample
from .models import DynamicalModel
from .mock import mock
