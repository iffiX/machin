import os
import shutil
from PIL.Image import Image
from matplotlib.figure import Figure
from argparse import Namespace
from typing import Union, Dict, Optional, Any
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.loggers.base import rank_zero_experiment


class LocalMediaLogger(LightningLoggerBase):
    """
    A logger designed to handle log_image and log_artifact just like the
    neptune.ai logger does. log_hyperparams and log_metrics are not handled.
    """

    def __init__(self, image_dir: str, artifact_dir: str):
        super().__init__()
        self.image_dir = image_dir
        self.artifact_dir = artifact_dir
        self._counters = {}

    @property
    def name(self):
        return "LocalMediaLogger"

    @property
    @rank_zero_experiment
    def experiment(self):
        return None

    @property
    def version(self):
        return "0.1"

    @rank_zero_only
    def log_hyperparams(self, params: Union[Dict[str, Any], Namespace]):
        """
        Not implemented.
        """
        pass

    @rank_zero_only
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int]):
        """
        Not implemented.
        """
        pass

    @rank_zero_only
    def log_artifact(self, artifact: str, destination: Optional[str] = None):
        """
        Save an artifact (file) in the ``artifact_dir``.

        Args:
            artifact: A path to the file in local filesystem.
            destination: Optional. Default is ``None``. A destination path
                in the artifact directory.
                If ``None`` is passed, an artifact file name will be used.
        """
        if destination:
            shutil.copy(
                artifact,
                self.enumerate_till_valid(os.path.join(self.artifact_dir, destination)),
            )
        else:
            shutil.copy(
                artifact,
                self.enumerate_till_valid(
                    os.path.join(self.artifact_dir, os.path.basename(artifact))
                ),
            )

    @rank_zero_only
    def log_image(
        self, log_name: str, image: Union[str, Any], step: Optional[int] = None
    ) -> None:
        """
        Log image data in the ``image_dir``.

        Args:
            log_name: The name of log, i.e. bboxes, visualisations,
                sample_images.
            image: The value of the log (data-point).
                Can be one of the following types: PIL image,
                `matplotlib.figure.Figure`,
                path to image file (str).
            step: Step number at which the metrics should be recorded,
                must be strictly increasing.
        """
        if log_name not in self._counters:
            self._counters[log_name] = 0

        if not isinstance(image, str):
            log_path = log_name + f"_{step or self._counters[log_name]}.png"
        else:
            extension = os.path.splitext(image)[1]
            log_path = log_name + f"_{step or self._counters[log_name]}{extension}"
        self._counters[log_name] += 1

        path = os.path.join(self.image_dir, log_path)
        if isinstance(image, Image):
            image.save(path)
        elif isinstance(image, Figure):
            image.savefig(path)
        elif isinstance(image, str):
            shutil.copy(image, path)
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

    @rank_zero_only
    def save(self):
        super().save()

    @rank_zero_only
    def finalize(self, status):
        pass

    @staticmethod
    def enumerate_till_valid(path):
        counter = 0
        cur_path = path
        while os.path.exists(cur_path):
            counter += 1
            names = list(os.path.splitext(path))
            names[0] += f"_{counter}"
            cur_path = "".join(names)
        return cur_path
