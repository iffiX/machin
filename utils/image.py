import numpy as np
import moviepy.editor as mpy
import matplotlib.pyplot as plt

from PIL import Image
from typing import List
from multiprocessing import get_context


def show_image(image, show_normalized=True, pause_time=0.01, title=""):
    """
    Args:
        image: A numpy array of shape [width, height, channels] or [width, height],
               datatype can be float(double) or int(int8, int16... etc.).
               When a frame is float type, its value range should be [0, 1].
               When a frame is integer type, its value range should be [0, 255].
        show_normalized: Show normalized image alongside the original one.
        pause_time: Pause time between displaying currrent image and the next one.
        title: Title of the display window.
    """
    if np.issubdtype(image.dtype, np.integer):
        image = image.astype(np.floating) / 255

    fig = plt.figure(title, clear=True)
    fig.canvas.set_window_title(title)
    if show_normalized:
        ax = fig.add_subplot("121")
        ax.set_facecolor((0.0, 0.0, 0.0))
        ax.imshow(image, vmin=np.min(image), vmax=np.max(image))

        ax2 = fig.add_subplot("122")
        ax2.set_facecolor((0.0, 0.0, 0.0))
        pix_range = np.max(image) - np.min(image)
        if pix_range == 0:
            pix_range += 0.0001
        ax2.imshow((image - np.min(image)) / pix_range, vmin=0, vmax=1)
        plt.pause(pause_time)

    else:
        ax = fig.add_subplot("111")
        ax.set_facecolor((0.0, 0.0, 0.0))
        ax.imshow(image, vmin=np.min(image), vmax=np.max(image))
        plt.pause(pause_time)


def _gif_writer(frames, path, fps):
    for f in range(len(frames)):
        if np.issubdtype(frames[f].dtype, np.integer):
            frames[f] = frames[f].astype(np.uint8)
        elif np.issubdtype(frames[f].dtype, np.floating):
            frames[f] = (frames[f] * 255).astype(np.uint8)
        if frames[f].shape == 2:
            # consider as a grey scale image
            frames[f] = np.stack([f, f, f], axis=-1)

    clip = mpy.ImageSequenceClip(frames, fps=fps)
    clip.write_gif(path + ".gif", fps=fps, verbose=False, logger=None)


def create_gif(frames: List[np.array], path, fps=15):
    """
    Args:
        frames: A list of numpy arrays of shape [width, height, channels] or [width, height],
                   datatype can be float(double) or int(int8, int16... etc.).
                   When a frame is float type, its value range should be [0, 1].
                   When a frame is integer type, its value range should be [0, 255].
        path: Path to save the gif image, without extension.
        fps: frames per second
    """
    if len(frames) == 0:
        raise RuntimeWarning("Empty frames sequence, file {}.gif skipped".format(path))
    p = get_context("spawn").Process(target=_gif_writer, args=(frames, path, fps))
    p.daemon = True
    p.start()


def _image_writer(image, path, extension):
    if np.issubdtype(image.dtype, np.integer):
        image = image.astype(np.uint8)
    elif np.issubdtype(image.dtype, np.floating):
        image = (image * 255).astype(np.uint8)
    if image.shape == 2:
        # consider as a grey scale image
        image = np.stack([image, image, image], axis=-1)
    image = Image.fromarray(image)
    image.save(path + extension)


def create_image(image: np.array, path, extension=".png"):
    """
    Args:
        image: A numpy array of shape [width, height, channels] or [width, height],
                   datatype can be float(double) or int(int8, int16... etc.).
                   When a frame is float type, its value range should be [0, 1].
                   When a frame is integer type, its value range should be [0, 255].
        path: Path to save the image, without extension.
        extension: Image extension
    """
    p = get_context("spawn").Process(target=_image_writer, args=(image, path, extension))
    p.daemon = True
    p.start()
