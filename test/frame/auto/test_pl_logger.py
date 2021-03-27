from machin.frame.auto.pl_logger import LocalMediaLogger
from pytorch_lightning.loggers.base import DummyExperiment
from PIL import Image
import os
import matplotlib.pyplot as plt


class TestLocalMediaLogger:
    def test_all(self, tmpdir):
        tmp_dir = str(tmpdir.make_numbered_dir())
        lm_logger = LocalMediaLogger(tmp_dir, tmp_dir)

        assert lm_logger.name == "LocalMediaLogger"
        assert isinstance(lm_logger.experiment, DummyExperiment)
        assert lm_logger.version == "0.1"

        # nothing happens
        lm_logger.log_hyperparams({"a": 0.1})
        lm_logger.log_metrics({"b": 0.1}, step=1)
        lm_logger.save()
        lm_logger.finalize("")

        # test logging artifact
        artifact_path = str(os.path.join(tmp_dir, "test.txt"))
        new_artifact_path = str(os.path.join(tmp_dir, "test1.txt"))

        with open(artifact_path, "w") as f:
            f.write("1" * 1000)
        lm_logger.log_artifact(artifact_path)
        assert os.path.exists(artifact_path)
        os.remove(artifact_path)

        with open(artifact_path, "w") as f:
            f.write("1" * 1000)
        lm_logger.log_artifact(artifact_path, "test1.txt")
        assert not os.path.exists(artifact_path)
        assert os.path.exists(new_artifact_path)
        os.remove(new_artifact_path)

        # test logging image
        image_path = str(os.path.join(tmp_dir, "test_1.png"))

        # PIL Image test
        im = Image.new("RGB", (100, 100))
        lm_logger.log_image("test", im, step=1)
        assert os.path.exists(image_path)
        os.remove(image_path)

        # Matplotlib figure test
        fig = plt.figure()
        lm_logger.log_image("test", fig, step=1)
        assert os.path.exists(image_path)
        os.remove(image_path)

        # Image file test
        source_path = str(os.path.join(tmp_dir, "x.jpg"))
        target_path = str(os.path.join(tmp_dir, "test_1.jpg"))
        im.save(source_path)
        lm_logger.log_image("test", source_path, step=1)
        assert os.path.exists(target_path)
        os.remove(target_path)
