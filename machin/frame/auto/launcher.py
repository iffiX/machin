import pytorch_lightning as pl
from .dataset import *


class Launcher(pl.LightningModule):
    def __init__(self, config, env_dataset_creator):
        super(Launcher, self).__init__()
        self.algorithm = algorithm(
            *models
        )

    def train_dataloader(self):
        pass

    def test_dataloader(self):
        pass

    def training_step(self, batch, _batch_idx):
        pass

    def test_step(self, batch, _batch_idx):
        pass

    def configure_optimizers(self):
        pass


def launch(config, env_dataset_creator):
    pass