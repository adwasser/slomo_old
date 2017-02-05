from os import path

here = path.abspath(path.dirname(__file__))
with open(path.join(here, "../_version.py"), encoding="utf-8") as f:
    # sets version
    exec(f.readline())

from .parser import parse

