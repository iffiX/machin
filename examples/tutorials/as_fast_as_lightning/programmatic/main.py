from machin.auto.config import (
    generate_algorithm_config,
    generate_env_config,
    generate_training_config,
    launch,
)

import torch as t
import torch.nn as nn


class SomeQNet(nn.Module):
    def __init__(self, state_dim, action_num):
        super().__init__()

        self.fc1 = nn.Linear(state_dim, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, action_num)

    def forward(self, state):
        a = t.relu(self.fc1(state))
        a = t.relu(self.fc2(a))
        return self.fc3(a)


if __name__ == "__main__":
    config = generate_algorithm_config("DQN")
    config = generate_env_config("openai_gym", config)
    config = generate_training_config(
        root_dir="trial", episode_per_epoch=10, max_episodes=10000, config=config
    )
    config["frame_config"]["models"] = ["SomeQNet", "SomeQNet"]
    config["frame_config"]["model_kwargs"] = [{"state_dim": 4, "action_num": 2}] * 2
    launch(config)
