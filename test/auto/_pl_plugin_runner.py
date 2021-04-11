from machin.parallel.distributed import get_world, get_cur_rank
from machin.utils.helper_classes import Object
from torch.utils.data import DataLoader, TensorDataset
import os
import sys
import pickle
import torch as t
import torch.nn as nn
import pytorch_lightning as pl

# necessary to patch PL DDP plugins
import machin.auto


class NNModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 10)

    def forward(self, x):
        return t.sum(x)


class ParallelModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.nn_model = NNModule()
        self.frame = Object({"optimizers": None, "lr_schedulers": None})

    def train_dataloader(self):
        return DataLoader(
            dataset=TensorDataset(t.ones([5, 10])), collate_fn=lambda x: x
        )

    def training_step(self, batch, _batch_idx):
        world_inited = get_world() is not None
        model_inited = isinstance(self.nn_model, NNModule)

        if world_inited and get_cur_rank() == 0:
            with open(os.environ["TEST_SAVE_PATH"], "wb") as f:
                pickle.dump([model_inited], f)
        if not world_inited:
            raise RuntimeError("World not initialized.")
        return None

    def init_frame(self):
        pass

    def configure_optimizers(self):
        return None


if __name__ == "__main__":
    os.environ["WORLD_SIZE"] = "3"
    print(os.environ["TEST_SAVE_PATH"])
    trainer = pl.Trainer(
        gpus=0,
        num_nodes=1,
        num_processes=3,
        limit_train_batches=1,
        max_steps=1,
        accelerator="ddp" if sys.argv[1] == "ddp" else "ddp_spawn",
    )
    model = ParallelModule()
    trainer.fit(model)
