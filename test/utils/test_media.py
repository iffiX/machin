from machin.utils.media import (
    show_image, create_video, create_video_subproc,
    create_image, create_image_subproc
)
from os.path import join
import os
import pytest
import numpy as np


@pytest.fixture(scope="function")
def images():
    """
    Generate a list of images.

    Args:
    """
    images = [
        np.random.randint(0, 255, size=[128, 128], dtype=np.uint8)
        for _ in range(120)
    ]
    return images


@pytest.fixture(scope="function")
def images_f():
    """
    Returns a random image

    Args:
    """
    images = [
        np.random.rand(128, 128)
        for _ in range(120)
    ]
    return images


def test_show_image(images):
    """
    Takes an image

    Args:
        images: (todo): write your description
    """
    show_image(images[0], show_normalized=True)
    show_image(images[0], show_normalized=False)


def test_create_video(images, tmpdir):
    """
    Create a new video.

    Args:
        images: (todo): write your description
        tmpdir: (todo): write your description
    """
    tmp_dir = str(tmpdir.make_numbered_dir())
    create_video(images, tmp_dir, "vid", extension=".gif")
    assert os.path.exists(join(tmp_dir, "vid.gif"))
    create_video(images, tmp_dir, "vid", extension=".mp4")
    assert os.path.exists(join(tmp_dir, "vid.mp4"))


def test_create_video_float(images_f, tmpdir):
    """
    Create a new video float.

    Args:
        images_f: (todo): write your description
        tmpdir: (todo): write your description
    """
    tmp_dir = str(tmpdir.make_numbered_dir())
    create_video(images_f, tmp_dir, "vid", extension=".gif")
    assert os.path.exists(join(tmp_dir, "vid.gif"))
    create_video(images_f, tmp_dir, "vid", extension=".mp4")
    assert os.path.exists(join(tmp_dir, "vid.mp4"))


def test_create_video_subproc(images, tmpdir):
    """
    Create a new video.

    Args:
        images: (list): write your description
        tmpdir: (todo): write your description
    """
    tmp_dir = str(tmpdir.make_numbered_dir())
    create_video_subproc([], tmp_dir, "empty", extension=".gif")()
    create_video_subproc(images, tmp_dir, "vid", extension=".gif")()
    assert os.path.exists(join(tmp_dir, "vid.gif"))


def test_create_image(images, tmpdir):
    """
    Create a new test.

    Args:
        images: (todo): write your description
        tmpdir: (todo): write your description
    """
    tmp_dir = str(tmpdir.make_numbered_dir())
    create_image(images[0], tmp_dir, "img", extension=".png")
    assert os.path.exists(join(tmp_dir, "img.png"))


def test_create_image_float(images_f, tmpdir):
    """
    Create a new image file for image.

    Args:
        images_f: (todo): write your description
        tmpdir: (todo): write your description
    """
    tmp_dir = str(tmpdir.make_numbered_dir())
    create_image(images_f[0], tmp_dir, "img", extension=".png")
    assert os.path.exists(join(tmp_dir, "img.png"))


def test_create_image_subproc(images, tmpdir):
    """
    Create subproc of - images.

    Args:
        images: (todo): write your description
        tmpdir: (todo): write your description
    """
    tmp_dir = str(tmpdir.make_numbered_dir())
    create_image_subproc(images[0], tmp_dir, "img", extension=".png")()
    assert os.path.exists(join(tmp_dir, "img.png"))
