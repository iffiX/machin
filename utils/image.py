import numpy as np
import moviepy.editor as mpy
import matplotlib.pyplot as plt

from PIL import Image
from typing import List
from multiprocessing import get_context


def show_image(image: np.array, show_normalized=True, pause_time=0.01, title=""):
    """
    Use matplotlib to show a single image. You may repeatedly call this method with
    the same ``title`` argument to show a video or a dynamically changing image
    Args:
        image: A numpy array of shape (H, W, C) or (H, W),
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


def create_gif(frames: List[np.array], path, fps=15):
    for f in range(len(frames)):
        if np.issubdtype(frames[f].dtype, np.integer):
            frames[f] = frames[f].astype(np.uint8)
        elif np.issubdtype(frames[f].dtype, np.floating):
            frames[f] = (frames[f] * 255).astype(np.uint8)
        if frames[f].ndim == 2:
            # consider as a grey scale image
            frames[f] = np.stack([f, f, f], axis=-1)

    clip = mpy.ImageSequenceClip(frames, fps=fps)
    clip.write_gif(path + ".gif", fps=fps, verbose=False, logger=None)


def create_gif_subproc(frames: List[np.array], path, fps=15, daemon=True):
    """
    Create gif with a subprocess, since it takes a lot of time for ``moviepy`` to encode the gif
    file.

    .. seealso:: :class: `machin.utils.image.create_gif_subproc`

    Note: if ``daemon`` is true, then this function cannot be used in a daemonic subprocess,
    such as a process of ``utils.parallel.Pool``, but could be used with
    ``utils.parallel.distributed`` since those processes are launched using
    ``subprocess.Popen`` if you are using the pytorch distributed launch tool.

    Note: daemonic process will be terminated by the main process when the main process exits.

    Args:
        frames: A list of numpy arrays of shape (H, W, C) or (H, W),
                   datatype can be float(double) or int(int8, int16... etc.).
                   When a frame is float type, its value range should be [0, 1].
                   When a frame is integer type, its value range should be [0, 255].
        path: Path to save the gif image, without extension.
        fps: frames per second.
        daemon: Whether launching the saving process as a daemonic process.
    """
    if len(frames) == 0:
        raise RuntimeWarning("Empty frames sequence, file {}.gif skipped".format(path))
    p = get_context("spawn").Process(target=create_gif, args=(frames, path, fps))
    p.daemon = daemon
    p.start()


def create_image(image: np.array, path: str, extension=".png"):
    if np.issubdtype(image.dtype, np.integer):
        image = image.astype(np.uint8)
    elif np.issubdtype(image.dtype, np.floating):
        image = (image * 255).astype(np.uint8)
    if image.sndim == 2:
        # consider as a grey scale image
        image = np.stack([image, image, image], axis=-1)
    image = Image.fromarray(image)
    image.save(path + extension)


def create_image_subproc(image: np.array, path, extension=".png", daemon=True):
    """
    .. seealso:: :class: `machin.utils.image.create_gif_subproc`

    Args:
        image: A numpy array of shape [width, height, channels] or [width, height],
                   datatype can be float(double) or int(int8, int16... etc.).
                   When a frame is float type, its value range should be [0, 1].
                   When a frame is integer type, its value range should be [0, 255].
        path: Path to save the image, without extension.
        extension: Image extension.
        daemon: Whether launching the saving process as a daemonic process.
    """
    p = get_context("spawn").Process(target=create_image, args=(image, path, extension))
    p.daemon = daemon
    p.start()
