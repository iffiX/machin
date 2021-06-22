from machin.frame.transition import Transition
from machin.frame.buffers import Buffer

import dill
import pytest
import torch as t


class TestBuffer:
    BUFFER_SIZE = 1
    SAMPLE_BUFFER_SIZE = 10

    ########################################################################
    # Test for Buffer.append
    ########################################################################
    param_test_append = [
        (
            [
                {
                    "state": {"state_1": t.zeros([1, 2])},
                    "action": {"action_1": t.zeros([1, 3])},
                    "next_state": {"next_state_1": t.zeros([1, 2])},
                    "reward": 1,
                    "terminal": True,
                    "some_custom_attr": None,
                },
                Transition(
                    state={"state_1": t.zeros([1, 2])},
                    action={"action_1": t.zeros([1, 3])},
                    next_state={"next_state_1": t.zeros([1, 2])},
                    reward=1,
                    terminal=True,
                    some_custom_attr=None,
                ),
            ],
            ("state", "action", "next_state", "reward", "terminal"),
            None,
            None,
        ),
        (
            [
                {
                    "state": {"state_1": t.zeros([1, 2])},
                    "action": {"action_1": t.zeros([1, 3])},
                    "next_state": {"next_state_1": t.zeros([1, 2])},
                    "reward": 1,
                    "terminal": True,
                    "some_custom_attr": None,
                }
            ],
            (
                "state",
                "action",
                "next_state",
                "reward",
                "terminal",
                "some_special_attr",
            ),
            ValueError,
            "Transition object missing attributes",
        ),
        (
            [
                {
                    "state": {"state_1": t.zeros([1, 2])},
                    "action": {"action_1": t.zeros([1, 3])},
                    "next_state": {"next_state_1": t.zeros([1, 2])},
                    "reward": 1,
                    "terminal": True,
                    "some_custom_attr": None,
                },
                {
                    "state": {"state_1": t.zeros([1, 2])},
                    "action": {"action_1": t.zeros([1, 3])},
                    "next_state": {"next_state_1": t.zeros([1, 2])},
                    "reward": 1,
                    "terminal": True,
                    "some_other_custom_attr": None,
                },
            ],
            ("state", "action", "next_state", "reward", "terminal"),
            ValueError,
            "Transition object has different attributes",
        ),
    ]

    @pytest.mark.parametrize(
        "trans_list,require_attr," "exception,match", param_test_append
    )
    def test_append(self, trans_list, require_attr, exception, match):
        buffer = Buffer(self.BUFFER_SIZE)
        if exception is not None:
            with pytest.raises(exception, match=match):
                for trans in trans_list:
                    buffer.append(trans, require_attr)
        else:
            for trans in trans_list:
                buffer.append(trans, require_attr)

    ########################################################################
    # Test for Buffer.clear
    ########################################################################
    param_test_clear = [
        [
            {
                "state": {"state_1": t.zeros([1, 2])},
                "action": {"action_1": t.zeros([1, 3])},
                "next_state": {"next_state_1": t.zeros([1, 2])},
                "reward": 1,
                "terminal": True,
                "some_custom_attr": None,
            }
        ]
    ]

    @pytest.mark.parametrize("trans_list", param_test_clear)
    def test_clear(self, trans_list):
        buffer = Buffer(self.BUFFER_SIZE)
        for trans in trans_list:
            buffer.append(trans)
        buffer.clear()
        assert buffer.size() == 0

    ########################################################################
    # Test for Buffer.sample_batch
    ########################################################################
    def t_eq(self, a: t.Tensor, b: t.Tensor):
        if a.shape != b.shape:
            return False
        b = b.to(a.device)
        return t.all(a == b)

    @pytest.fixture(scope="class")
    def const_buffer(self, pytestconfig):
        data = [
            {
                "state": {"state_1": t.zeros([1, 2])},
                "action": {"action_1": t.zeros([1, 3])},
                "next_state": {"next_state_1": t.zeros([1, 4])},
                "reward": 10.0,
                "terminal": True,
                "data_index": i,
                "not_concatenable": (i, "some_str"),
            }
            for i in range(self.SAMPLE_BUFFER_SIZE)
        ]
        buffer = Buffer(
            buffer_size=self.SAMPLE_BUFFER_SIZE,
            buffer_device=pytestconfig.getoption("gpu_device"),
        )
        for d in data:
            buffer.append(d)
        return buffer

    @pytest.mark.parametrize(
        "batch_size",
        [0, int(SAMPLE_BUFFER_SIZE / 2), SAMPLE_BUFFER_SIZE, SAMPLE_BUFFER_SIZE * 2],
    )
    @pytest.mark.parametrize("concat", [True, False])
    @pytest.mark.parametrize("dev", [None, "cpu"])  # buffer already on gpu
    @pytest.mark.parametrize(
        "sample_method", ["random", "random_unique", "all", "some_invalid_method"]
    )
    @pytest.mark.parametrize(
        "sample_attrs,concat_attrs,should_be_attrs",
        [
            # Case 0
            (
                None,
                None,
                [
                    "state",
                    "action",
                    "next_state",
                    "reward",
                    "terminal",
                    "data_index",
                    "not_concatenable",
                ],
            ),
            # Case 1
            (
                ["state", "action", "next_state", "reward", "terminal"],
                [],
                ["state", "action", "next_state", "reward", "terminal"],
            ),
            # Case 2
            (
                ["state", "action", "next_state", "reward", "terminal", "data_index"],
                ["data_index"],
                ["state", "action", "next_state", "reward", "terminal", "data_index"],
            ),
            # Case 3
            (
                ["state", "action", "next_state", "reward", "terminal", "*"],
                ["data_index"],
                ["state", "action", "next_state", "reward", "terminal", "*"],
            ),
            # Case 4
            (
                ["state", "action", "next_state", "reward", "terminal", "*"],
                ["not_concatenable"],
                ["state", "action", "next_state", "reward", "terminal", "*"],
            ),
        ],
    )
    def test_sample(
        self,
        const_buffer,
        batch_size,
        concat,
        dev,
        sample_method,
        sample_attrs,
        concat_attrs,
        should_be_attrs,
    ):
        sample_not_empty = batch_size != 0 or sample_method == "all"
        will_concat_custom = (
            isinstance(concat_attrs, list) and concat and sample_not_empty
        )

        if sample_method == "some_invalid_method":
            with pytest.raises(
                RuntimeError, match="Cannot find specified sample method"
            ):
                const_buffer.sample_batch(
                    batch_size,
                    concatenate=concat,
                    device=dev,
                    sample_method=sample_method,
                    sample_attrs=sample_attrs,
                    additional_concat_attrs=concat_attrs,
                )
        elif will_concat_custom and "not_concatenable" in concat_attrs:
            with pytest.raises(ValueError, match="Batch not concatenable"):
                const_buffer.sample_batch(
                    batch_size,
                    concatenate=concat,
                    device=dev,
                    sample_method=sample_method,
                    sample_attrs=sample_attrs,
                    additional_concat_attrs=concat_attrs,
                )
        else:
            bsize, b = const_buffer.sample_batch(
                batch_size,
                concatenate=concat,
                device=dev,
                sample_method=sample_method,
                sample_attrs=sample_attrs,
                additional_concat_attrs=concat_attrs,
            )

            # Check form of sample
            if bsize == 0:
                assert b is None
            else:
                assert len(b) == len(should_be_attrs)
                for data, attr in zip(b, should_be_attrs):
                    if attr == "state":
                        if concat:
                            assert self.t_eq(data["state_1"], t.zeros([bsize, 2]))
                        else:
                            assert (
                                isinstance(data["state_1"], list)
                                and len(data["state_1"]) == bsize
                                and self.t_eq(data["state_1"][0], t.zeros([1, 2]))
                            )
                    elif attr == "action":
                        if concat:
                            assert self.t_eq(data["action_1"], t.zeros([bsize, 3]))
                        else:
                            assert (
                                isinstance(data["action_1"], list)
                                and len(data["action_1"]) == bsize
                                and self.t_eq(data["action_1"][0], t.zeros([1, 3]))
                            )
                    elif attr == "next_state":
                        if concat:
                            assert self.t_eq(data["next_state_1"], t.zeros([bsize, 4]))
                        else:
                            assert (
                                isinstance(data["next_state_1"], list)
                                and len(data["next_state_1"]) == bsize
                                and self.t_eq(data["next_state_1"][0], t.zeros([1, 4]))
                            )
                    elif attr == "reward":
                        if concat:
                            assert self.t_eq(data, t.full([bsize, 1], 10.0))
                        else:
                            assert (
                                isinstance(data, list)
                                and len(data) == bsize
                                and data[0] == 10
                            )
                    elif attr == "terminal":
                        if concat:
                            assert self.t_eq(
                                data, t.full([bsize, 1], True, dtype=t.bool)
                            )
                        else:
                            assert (
                                isinstance(data, list)
                                and len(data) == bsize
                                and data[0] is True
                            )
                    elif attr == "data_index":
                        if will_concat_custom:
                            assert list(data.shape) == [bsize, 1]
                        else:
                            assert (
                                isinstance(data, list)
                                and len(data) == bsize
                                and isinstance(data[0], int)
                            )
                    elif attr == "not_concatenable":
                        assert (
                            isinstance(data, list)
                            and len(data) == bsize
                            and isinstance(data[0], tuple)
                        )

    ########################################################################
    # Test for Buffer.__reduce__
    ########################################################################
    def test_reduce(self, const_buffer):
        str = dill.dumps(const_buffer)
        _buffer = dill.loads(str)
