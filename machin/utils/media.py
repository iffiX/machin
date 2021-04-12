from PIL import Image
from typing import List
from machin.parallel import get_context
import os
import numpy as np
import moviepy.editor as mpy
import matplotlib.pyplot as plt


def show_image(
    image: np.ndarray,
    show_normalized: bool = True,
    pause_time: float = 0.01,
    title: str = "",
):
    """
    Use matplotlib to show a single image. You may repeatedly call this method
    with the same ``title`` argument to show a video or a dynamically changing
    image.

    Args:
        image: A numpy array of shape (H, W, C) or (H, W), and with ``dtype``
            = any float or any int.
            When a frame is float type, its value range should be [0, 1].
            When a frame is integer type, its value range should be [0, 255].
        show_normalized: Show normalized image alongside the original one.
        pause_time: Pause time between displaying current image and the next
            one.
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
        pix_range = (np.max(image) - np.min(image)) + 1e-6
        ax2.imshow((image - np.min(image)) / pix_range, vmin=0, vmax=1)
        plt.pause(pause_time)

    else:
        ax = fig.add_subplot("111")
        ax.set_facecolor((0.0, 0.0, 0.0))
        ax.imshow(image, vmin=np.min(image), vmax=np.max(image))
        plt.pause(pause_time)


def create_video(
    frames: List[np.ndarray],
    path: str,
    filename: str,
    extension: str = ".gif",
    fps: int = 15,
):
    """
    Args:
        frames: A list of numpy arrays of shape (H, W, C) or (H, W), and with
            ``dtype`` = any float or any int.
            When a frame is float type, its value range should be [0, 1].
            When a frame is integer type, its value range should be [0, 255].
        path: Directory to save the video.
        filename: File name.
        extension: File extension.
        fps: frames per second.
    """
    if frames:
        for f in range(len(frames)):
            if np.issubdtype(frames[f].dtype, np.integer):
                frames[f] = frames[f].astype(np.uint8)
            elif np.issubdtype(frames[f].dtype, np.floating):
                frames[f] = (frames[f] * 255).astype(np.uint8)
            if frames[f].ndim == 2:
                # consider as a grey scale image
                frames[f] = np.repeat(frames[f][:, :, np.newaxis], 3, axis=2)

        clip = mpy.ImageSequenceClip(frames, fps=fps)
        if extension.lower() == ".gif":
            clip.write_gif(
                os.path.join(path, filename + extension),
                fps=fps,
                verbose=False,
                logger=None,
            )
        else:
            clip.write_videofile(
                os.path.join(path, filename + extension),
                fps=fps,
                verbose=False,
                logger=None,
            )
        clip.close()


def create_video_subproc(
    frames: List[np.ndarray],
    path: str,
    filename: str,
    extension: str = ".gif",
    fps: int = 15,
    daemon: bool = True,
):
    """
    Create video with a subprocess, since it takes a lot of time for ``moviepy``
    to encode the video file.

    See Also:
         :func:`.create_video`

    Note:
        if ``daemon`` is true, then this function cannot be used in a
        daemonic subprocess.

    Args:
        frames: A list of numpy arrays of shape (H, W, C) or (H, W), and with
            ``dtype`` = any float or any int.
            When a frame is float type, its value range should be [0, 1].
            When a frame is integer type, its value range should be [0, 255].
        path: Directory to save the video.
        filename: File name.
        extension: File extension.
        fps: frames per second.
        daemon: Whether launching the saving process as a daemonic process.

    Returns:
        A wait function, once called, block until creation has finished.
    """

    def wait():
        pass

    if frames:
        p = get_context("spawn").Process(
            target=create_video, args=(frames, path, filename, extension, fps)
        )
        p.daemon = daemon
        p.start()

        def wait():
            p.join()

    return wait


def numpy_array_to_pil_image(image: np.ndarray):
    if np.issubdtype(image.dtype, np.integer):
        image = image.astype(np.uint8)
    elif np.issubdtype(image.dtype, np.floating):
        image = (image * 255).astype(np.uint8)
    if image.ndim == 2:
        # consider as a grey scale image
        image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
    image = Image.fromarray(image)
    return image


def create_image(image: np.ndarray, path: str, filename: str, extension: str = ".png"):
    """
    Args:
        image: A numpy array of shape (H, W, C) or (H, W), and with
            ``dtype`` = any float or any int.
            When a frame is float type, its value range should be [0, 1].
            When a frame is integer type, its value range should be [0, 255].
        path: Directory to save the image.
        filename: File name.
        extension: File extension.
    """
    image = numpy_array_to_pil_image(image)
    image.save(os.path.join(path, filename + extension))


def create_image_subproc(
    image: np.array,
    path: str,
    filename: str,
    extension: str = ".png",
    daemon: bool = True,
):
    """
    Create image with a subprocess.

    See Also:
         :func:`.create_image`

    Note:
        if ``daemon`` is true, then this function cannot be used in a
        daemonic subprocess.

    Args:
        image: A numpy array of shape (H, W, C) or (H, W), and with
            ``dtype`` = any float or any int.
            When a frame is float type, its value range should be [0, 1].
            When a frame is integer type, its value range should be [0, 255].
        path: Directory to save the image.
        filename: File name.
        extension: File extension.
        daemon: Whether launching the saving process as a daemonic process.

    Returns:
        A wait function, once called, block until creation has finished.
    """
    p = get_context("spawn").Process(
        target=create_image, args=(image, path, filename, extension)
    )
    p.daemon = daemon
    p.start()

    def wait():
        p.join()

    return wait
