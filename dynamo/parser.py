"""Routines for parsing the YAML config file"""

try:
    import ruamel.yaml as yaml
except ImportError as e:
    import yaml

def parse(config_file):
    """Read in the config file and construct a model to run."""
    with open(config_file) as f:
        config = yaml.load(f)
    return config
