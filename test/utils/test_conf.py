from machin.utils.conf import (
    Config,
    load_config_cmd,
    load_config_file,
    save_config,
    merge_config
)
from machin.utils.helper_classes import Object
from unittest import mock
from os.path import join
import os
import json


def get_config():
    c = Config()
    c.conf1 = 1
    c.conf2 = 2
    return c


@mock.patch("machin.utils.conf.argparse.ArgumentParser.parse_args",
            return_value=Object(data={
                "conf": [
                    "conf1=2",
                    "conf3=3"
                ]
            }))
def test_load_config_cmd(*_mock_classes):
    conf = load_config_cmd()
    assert conf["conf1"] == 2
    assert conf["conf2"] is None
    assert conf["conf3"] == 3

    conf = load_config_cmd(get_config())
    # configs from commandline precedes configs from the config file
    assert conf["conf1"] == 2
    assert conf["conf2"] == 2
    assert conf["conf3"] == 3


def test_load_config_file(tmpdir):
    tmp_dir = str(tmpdir.make_numbered_dir())
    with open(join(tmp_dir, "conf.json"), 'w') as config_file:
        json.dump({
            "conf1": 2,
            "conf3": 3
        }, config_file, sort_keys=True, indent=4)

    conf = load_config_file(join(tmp_dir, "conf.json"))
    assert conf["conf1"] == 2
    assert conf["conf2"] is None
    assert conf["conf3"] == 3

    conf = load_config_file(join(tmp_dir, "conf.json"), get_config())
    assert conf["conf1"] == 2
    assert conf["conf2"] == 2
    assert conf["conf3"] == 3


def test_save_config(tmpdir):
    conf = get_config()
    tmp_dir = str(tmpdir.make_numbered_dir())
    save_config(conf, join(tmp_dir, "conf.json"))
    assert os.path.exists(join(tmp_dir, "conf.json"))


def test_merge_config():
    conf = get_config()
    conf = merge_config(conf, {"conf1": 2, "conf3": 3})
    assert conf.conf1 == 2
    assert conf.conf2 == 2
    assert conf.conf3 == 3

    conf = get_config()
    conf2 = Config(
        conf1=2,
        conf3=3
    )
    conf = merge_config(conf, conf2)
    assert conf.conf1 == 2
    assert conf.conf2 == 2
    assert conf.conf3 == 3
