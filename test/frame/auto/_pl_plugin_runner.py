from machin.parallel.distributed import get_world, get_cur_rank
from torch.utils.data import DataLoader, TensorDataset
import os
import sys
import pickle
import torch as t
import torch.nn as nn
import pytorch_lightning as pl


class NNModule(nn.Module):
    def __init__(self):
        super(NNModule, self).__init__()
        self.fc1 = nn.Linear(10, 10)

    def forward(self, x):
        return t.sum(x)


class ParallelModule(pl.LightningModule):
    def __init__(self):
        super(ParallelModule, self).__init__()
        self.nn_model = NNModule()

    def train_dataloader(self):
        return DataLoader(
            dataset=TensorDataset(t.ones([5, 10])),
            collate_fn=lambda x: x
        )

    def training_step(self, batch, _batch_idx):
        model_properly_inited = isinstance(self.nn_model, NNModule)
        properly_inited = get_world() is not None

        if properly_inited and get_cur_rank() == 0:
            with open(os.environ["TEST_SAVE_PATH"], "wb") as f:
                pickle.dump([model_properly_inited, properly_inited], f)
        return None

    def configure_optimizers(self):
        return None


if __name__ == "__main__":
    os.environ["WORLD_SIZE"] = "3"
    trainer = pl.Trainer(
        gpus=0,
        num_nodes=1,
        num_processes=3,
        limit_train_batches=1,
        max_steps=1,
        accelerator="ddp" if sys.argv[1] == "ddp" else "ddp_spawn"
    )
    model = ParallelModule()
    trainer.fit(model)
