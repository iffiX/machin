from machin.parallel.pickle import dump_tensor
import dill
import torch as t


def test_dump_tensor():
    tensor = t.ones([10])
    assert t.all(
        dill.loads(dump_tensor(tensor, reduce_as_reference=False)) == tensor
    )
    assert t.all(
        dill.loads(dump_tensor(tensor, reduce_as_reference=True)) == tensor
    )
