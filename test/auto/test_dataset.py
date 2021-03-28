from machin.auto.dataset import determine_precision, DatasetResult
import pytest
import torch as t
import torch.nn as nn


class QNet(nn.Module):
    def __init__(self, state_dim, action_num):
        super(QNet, self).__init__()

        self.fc1 = nn.Linear(state_dim, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, action_num)

    def forward(self, state):
        a = t.relu(self.fc1(state))
        a = t.relu(self.fc2(a))
        return self.fc3(a)


def test_determine_precision():
    assert determine_precision([QNet(10, 2)]) == t.float32
    mixed_qnet = QNet(10, 2)
    mixed_qnet.fc2 = mixed_qnet.fc2.type(t.float64)
    with pytest.raises(RuntimeError, match="Multiple data types of parameters"):
        determine_precision([mixed_qnet])


class TestDatasetResult:
    def test_add_observation(self):
        dr = DatasetResult()
        dr.add_observation({})
        assert len(dr.observations) == 1
        assert len(dr) == 1

    def test_add_log(self):
        dr = DatasetResult()
        dr.add_log({})
        assert len(dr.logs) == 1
