from pathlib import Path

import yaml


def read_config():
    path = Path(__file__).parents[1]
    config_contents = yaml.safe_load(open(path / "config.yaml"))
    assert isinstance(config_contents, dict), "Config file has to be a dictionary"
    return config_contents
