from os.path import join
from machin.utils.prepare import (
    prep_clear_dirs,
    prep_create_dirs,
    prep_load_state_dict,
    prep_load_model,
)

import os
import pytest
import torch as t


def create_file(file_path):
    base_dir = os.path.dirname(file_path)
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    open(file_path, "a").close()


def test_prep_clear_dirs(tmpdir):
    tmp_dir = str(tmpdir.make_numbered_dir())
    create_file(join(tmp_dir, "some_dir", "some_file"))
    create_file(join(tmp_dir, "some_file2"))
    os.symlink(join(tmp_dir, "some_dir", "some_file"), join(tmp_dir, "some_file3"))
    prep_clear_dirs([tmp_dir])
    assert not os.path.exists(join(tmp_dir, "some_dir", "some_file"))
    assert not os.path.exists(join(tmp_dir, "some_file2"))
    assert not os.path.exists(join(tmp_dir, "some_file3"))


def test_prep_create_dirs(tmpdir):
    tmp_dir = str(tmpdir.make_numbered_dir())
    prep_create_dirs([join(tmp_dir, "some_dir")])
    assert os.path.exists(join(tmp_dir, "some_dir")) and os.path.isdir(
        join(tmp_dir, "some_dir")
    )


def test_prep_load_state_dict(pytestconfig):
    model = t.nn.Linear(100, 100)
    model2 = t.nn.Linear(100, 100).to(pytestconfig.getoption("gpu_device"))
    state_dict = model2.state_dict()
    prep_load_state_dict(model, state_dict)
    assert t.all(model.weight == model2.weight.to("cpu"))
    assert t.all(model.bias == model2.bias.to("cpu"))


def test_prep_load_model(tmpdir):
    tmp_dir = str(tmpdir.make_numbered_dir())
    tmp_dir2 = str(tmpdir.make_numbered_dir())

    # create example model directory
    with t.no_grad():
        model = t.nn.Linear(100, 100, bias=False)
        model.weight.fill_(0)
        t.save(model, join(tmp_dir, "model_0.pt"))
        model.weight.fill_(1)
        t.save(model, join(tmp_dir, "model_100.pt"))

    with pytest.raises(RuntimeError, match="Model directory doesn't exist"):
        prep_load_model(join(tmp_dir, "not_exist_dir"), {"model": model})

    # load a specific version
    prep_load_model(tmp_dir, {"model": model}, version=0)
    assert t.all(model.weight == 0)

    # load a non-exist version in a directory with valid models
    # will load version 100
    prep_load_model(tmp_dir, {"model": model}, version=50)
    assert t.all(model.weight == 1)

    # load the newest version
    prep_load_model(tmp_dir, {"model": model})
    assert t.all(model.weight == 1)

    # load a non-exist version in a directory with invalid models
    # eg: cannot find the same version for all models in the model map
    with pytest.raises(RuntimeError, match="Cannot find a valid version"):
        prep_load_model(tmp_dir2, {"model": model})
    prep_load_model(tmp_dir2, {"model": model}, quiet=True)
