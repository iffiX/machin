from machin.utils.media import (
    show_image,
    create_video,
    create_video_subproc,
    create_image,
    create_image_subproc,
)
from os.path import join
import os
import pytest
import numpy as np


@pytest.fixture(scope="function")
def images():
    images = [
        np.random.randint(0, 255, size=[128, 128], dtype=np.uint8) for _ in range(120)
    ]
    return images


@pytest.fixture(scope="function")
def images_f():
    images = [np.random.rand(128, 128) for _ in range(120)]
    return images


def test_show_image(images):
    show_image(images[0], show_normalized=True)
    show_image(images[0], show_normalized=False)


def test_create_video(images, tmpdir):
    tmp_dir = str(tmpdir.make_numbered_dir())
    create_video(images, tmp_dir, "vid", extension=".gif")
    assert os.path.exists(join(tmp_dir, "vid.gif"))
    create_video(images, tmp_dir, "vid", extension=".mp4")
    assert os.path.exists(join(tmp_dir, "vid.mp4"))


def test_create_video_float(images_f, tmpdir):
    tmp_dir = str(tmpdir.make_numbered_dir())
    create_video(images_f, tmp_dir, "vid", extension=".gif")
    assert os.path.exists(join(tmp_dir, "vid.gif"))
    create_video(images_f, tmp_dir, "vid", extension=".mp4")
    assert os.path.exists(join(tmp_dir, "vid.mp4"))


def test_create_video_subproc(images, tmpdir):
    tmp_dir = str(tmpdir.make_numbered_dir())
    create_video_subproc([], tmp_dir, "empty", extension=".gif")()
    create_video_subproc(images, tmp_dir, "vid", extension=".gif")()
    assert os.path.exists(join(tmp_dir, "vid.gif"))


def test_create_image(images, tmpdir):
    tmp_dir = str(tmpdir.make_numbered_dir())
    create_image(images[0], tmp_dir, "img", extension=".png")
    assert os.path.exists(join(tmp_dir, "img.png"))


def test_create_image_float(images_f, tmpdir):
    tmp_dir = str(tmpdir.make_numbered_dir())
    create_image(images_f[0], tmp_dir, "img", extension=".png")
    assert os.path.exists(join(tmp_dir, "img.png"))


def test_create_image_subproc(images, tmpdir):
    tmp_dir = str(tmpdir.make_numbered_dir())
    create_image_subproc(images[0], tmp_dir, "img", extension=".png")()
    assert os.path.exists(join(tmp_dir, "img.png"))
