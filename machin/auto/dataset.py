import os
import time
import tempfile
import numpy as np
import pytorch_lightning as pl
from typing import Iterable, List, Dict, Union, Any, Tuple, Callable
from torch.utils.data import IterableDataset
from pytorch_lightning.loggers.base import LoggerCollection
from machin.utils.media import create_video, numpy_array_to_pil_image


Scalar = Any


def determine_precision(models):
    dtype = set()
    for model in models:
        for k, v in model.named_parameters():
            dtype.add(v.dtype)
    dtype = list(dtype)
    if len(dtype) > 1:
        raise RuntimeError(
            "Multiple data types of parameters detected "
            f"in models: {dtype}, this is currently not supported "
            "since we need to determine the data type of your "
            "model input from your model parameter data type."
        )
    return dtype[0]


def get_loggers_as_list(module: pl.LightningModule):
    if isinstance(module.logger, LoggerCollection):
        return module.logger._logger_iterable
    else:
        return [module.logger]


def log_image(module, name, image: np.ndarray):
    for logger in get_loggers_as_list(module):
        if hasattr(logger, "log_image") and callable(logger.log_image):
            logger.log_image(name, numpy_array_to_pil_image(image))


def log_video(module, name, video_frames: List[np.ndarray]):
    # create video temp file
    fd, path = tempfile.mkstemp(suffix=".gif")
    os.close(fd)
    try:
        create_video(
            video_frames,
            os.path.dirname(path),
            os.path.basename(os.path.splitext(path)[0]),
            extension=".gif",
        )
    except Exception as e:
        print(e)
        os.remove(path)
        return

    size = os.path.getsize(path)
    while True:
        time.sleep(1)
        new_size = os.path.getsize(path)
        if size != 0 and new_size == size:
            break
        size = new_size

    for logger in get_loggers_as_list(module):
        if hasattr(logger, "log_artifact") and callable(logger.log_artifact):
            logger.log_artifact(path, name + ".gif")
    if os.path.exists(path):
        os.remove(path)


class DatasetResult:
    def __init__(
        self,
        observations: List[Dict[str, Any]] = None,
        logs: List[Dict[str, Union[Scalar, Tuple[Scalar, str]]]] = None,
    ):
        self.observations = observations or []
        self.logs = logs or []

    def add_observation(self, obs: Dict[str, Any]):
        self.observations.append(obs)

    def add_log(self, log: Dict[str, Union[Scalar, Tuple[Any, Callable]]]):
        self.logs.append(log)

    def __len__(self):
        return len(self.observations)


class RLDataset(IterableDataset):
    """
    Base class for all RL Datasets.
    """

    early_stopping_monitor = ""

    def __init__(self, **_kwargs):
        super().__init__()

    def __iter__(self) -> Iterable:
        return self

    def __next__(self):
        raise StopIteration()
