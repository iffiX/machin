from machin.frame.transition import TransitionBase, Transition

import pytest
import torch as t


class TestTransitionBase:
    ########################################################################
    # Test for TransitionBase.__init__
    ########################################################################
    param_test_init = [
        (
            [("ma1", {"ma1_1": None})],
            [("sa1", 1)],
            [("ca", None)],
            ValueError,
            "is a invalid tensor",
        ),
        (
            [("ma1", {"ma1_1": t.zeros([])})],
            [("sa1", 1)],
            [("ca", None)],
            ValueError,
            "is a invalid tensor",
        ),
        (
            [("ma1", {"ma1_1": t.zeros([1, 2])}), ("ma2", {"ma2_1": t.zeros([2, 3])})],
            [("sa1", 1)],
            [("ca", None)],
            ValueError,
            "has invalid batch size",
        ),
        (
            [("ma1", {"ma1_1": t.zeros([1, 2])}), ("ma2", {"ma2_1": t.zeros([1, 3])})],
            [("sa1", 1)],
            [("ca", None)],
            None,
            None,
        ),
        (
            [("ma1", {"ma1_1": t.zeros([1, 2])}), ("ma2", {"ma2_1": t.zeros([2, 3])})],
            [("sa1", 1)],
            [("ca", None)],
            ValueError,
            "has invalid batch size",
        ),
        (
            [("ma1", {"ma1_1": t.zeros([2, 2])}), ("ma2", {"ma2_1": t.zeros([2, 3])})],
            [("sa1", t.zeros([2, 4]))],
            [("ca", None)],
            None,
            None,
        ),
        (
            [("ma1", {"ma1_1": t.zeros([2, 2])}), ("ma2", {"ma2_1": t.zeros([2, 3])})],
            [("sa1", 1)],
            [("ca", None)],
            ValueError,
            "Transition sub attribute .+ is a scalar, " "but batch size is",
        ),
        (
            [("ma1", {"ma1_1": t.zeros([2, 2])}), ("ma2", {"ma2_1": t.zeros([2, 3])})],
            [("sa1", t.zeros([]))],
            [("ca", None)],
            ValueError,
            "Transition sub attribute .+ is a invalid tensor",
        ),
        (
            [("ma1", {"ma1_1": t.zeros([2, 2])}), ("ma2", {"ma2_1": t.zeros([2, 3])})],
            [("sa1", t.zeros([3, 4]))],
            [("ca", None)],
            ValueError,
            "Transition sub attribute .+ has invalid batch size",
        ),
        (
            [("ma1", {"ma1_1": t.zeros([2, 2])}), ("ma2", {"ma2_1": t.zeros([2, 3])})],
            [("sa1", None)],
            [("ca", None)],
            ValueError,
            "Transition sub attribute .+ requires scalar or tensor",
        ),
    ]

    @pytest.mark.parametrize("major,sub,custom,exception,match", param_test_init)
    def test_init(self, major, sub, custom, exception, match):
        if exception is not None:
            with pytest.raises(exception, match=match):
                _ = TransitionBase(
                    major_attr=[m[0] for m in major],
                    sub_attr=[s[0] for s in sub],
                    custom_attr=[c[0] for c in custom],
                    major_data=[m[1] for m in major],
                    sub_data=[s[1] for s in sub],
                    custom_data=[c[1] for c in custom],
                )
        else:
            _ = TransitionBase(
                major_attr=[m[0] for m in major],
                sub_attr=[s[0] for s in sub],
                custom_attr=[c[0] for c in custom],
                major_data=[m[1] for m in major],
                sub_data=[s[1] for s in sub],
                custom_data=[c[1] for c in custom],
            )

    ########################################################################
    # Test for TransitionBase.__len__
    ########################################################################
    param_test_len = [
        (
            [("ma1", {"ma1_1": t.zeros([2, 2])}), ("ma2", {"ma2_1": t.zeros([2, 3])})],
            [("sa1", t.zeros([2, 4]))],
            [("ca", None)],
            4,
        ),
    ]

    @pytest.mark.parametrize("major,sub,custom,length", param_test_len)
    def test_len(self, major, sub, custom, length):
        tb = TransitionBase(
            major_attr=[m[0] for m in major],
            sub_attr=[s[0] for s in sub],
            custom_attr=[c[0] for c in custom],
            major_data=[m[1] for m in major],
            sub_data=[s[1] for s in sub],
            custom_data=[c[1] for c in custom],
        )
        assert len(tb) == length

    ########################################################################
    # Test for TransitionBase.__setattr__, __setitem__ and __getitem__
    ########################################################################
    param_test_set_get = [
        (
            [("ma1", {"ma1_1": t.zeros([2, 2])}), ("ma2", {"ma2_1": t.zeros([2, 3])})],
            [("sa1", t.zeros([2, 4]))],
            [("ca", None)],
            "sa1",
            t.zeros([2, 4]),
        ),
    ]

    @pytest.mark.parametrize("major,sub,custom,key,value", param_test_set_get)
    def test_set_get(self, major, sub, custom, key, value):
        tb = TransitionBase(
            major_attr=[m[0] for m in major],
            sub_attr=[s[0] for s in sub],
            custom_attr=[c[0] for c in custom],
            major_data=[m[1] for m in major],
            sub_data=[s[1] for s in sub],
            custom_data=[c[1] for c in custom],
        )
        assert t.all(tb[key] == value)
        assert t.all(getattr(tb, key) == value)
        tb[key] = value
        assert t.all(tb[key] == value)
        assert t.all(getattr(tb, key) == value)

    def test_dynamic_set_get(self):
        tb = TransitionBase(
            major_attr=["ma1"],
            sub_attr=["sa1"],
            custom_attr=["ca"],
            major_data=[{"ma1_1": t.zeros([2, 2])}],
            sub_data=[t.zeros([2, 4])],
            custom_data=[None],
        )
        with pytest.raises(RuntimeError, match="You cannot dynamically set"):
            tb["some_attr"] = 1
        with pytest.raises(RuntimeError, match="You cannot dynamically set"):
            tb.some_attr = 1

    ########################################################################
    # Test for TransitionBase.major_attr, custom_attr, keys, has_keys
    # and items
    ########################################################################
    param_test_attr = [
        (
            [("ma1", {"ma1_1": t.zeros([2, 2])}), ("ma2", {"ma2_1": t.zeros([2, 3])})],
            [("sa1", t.zeros([2, 4]))],
            [("ca", None)],
        ),
    ]

    @pytest.mark.parametrize("major,sub,custom", param_test_attr)
    def test_attr(self, major, sub, custom):
        tb = TransitionBase(
            major_attr=[m[0] for m in major],
            sub_attr=[s[0] for s in sub],
            custom_attr=[c[0] for c in custom],
            major_data=[m[1] for m in major],
            sub_data=[s[1] for s in sub],
            custom_data=[c[1] for c in custom],
        )
        assert tb.major_attr == [m[0] for m in major]
        assert tb.sub_attr == [s[0] for s in sub]
        assert tb.custom_attr == [c[0] for c in custom]
        assert tb.keys() == (
            [m[0] for m in major] + [s[0] for s in sub] + [c[0] for c in custom]
        )
        all_attr = {k: v for k, v in major + sub + custom}
        for k, v in tb.items():
            assert k in all_attr
            if t.is_tensor(v) and t.is_tensor(all_attr[k]):
                assert t.all(all_attr[k] == v)
            else:
                assert all_attr[k] == v
        assert tb.has_keys(tb.keys())
        assert not tb.has_keys(["cSHxn3pyd1", "53D0dape5r"])

    param_test_to = [
        (
            [("ma1", {"ma1_1": t.zeros([2, 2])}), ("ma2", {"ma2_1": t.zeros([2, 3])})],
            [("sa1", t.zeros([2, 4]))],
            [("ca", None)],
        ),
    ]

    ########################################################################
    # Test for TransitionBase.to
    ########################################################################
    @pytest.mark.parametrize("major,sub,custom", param_test_attr)
    def test_to(self, major, sub, custom, pytestconfig):
        tb = TransitionBase(
            major_attr=[m[0] for m in major],
            sub_attr=[s[0] for s in sub],
            custom_attr=[c[0] for c in custom],
            major_data=[m[1] for m in major],
            sub_data=[s[1] for s in sub],
            custom_data=[c[1] for c in custom],
        )
        from colorlog import getLogger

        logger = getLogger("")
        tb.to(pytestconfig.getoption("gpu_device"))


class TestTransition:
    ########################################################################
    # Test for Transition.__init__
    ########################################################################
    param_test_init = [
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
            "reward": t.zeros([1]),
            "terminal": True,
            "some_custom_attr": None,
        },
        {
            "state": {"state_1": t.zeros([1, 2])},
            "action": {"action_1": t.zeros([1, 3])},
            "next_state": {"next_state_1": t.zeros([1, 2])},
            "reward": t.zeros([1, 4]),
            "terminal": True,
            "some_custom_attr": None,
        },
        {
            "state": {"state_1": t.zeros([2, 2])},
            "action": {"action_1": t.zeros([2, 3])},
            "next_state": {"next_state_1": t.zeros([2, 2])},
            "reward": t.zeros([2, 4]),
            "terminal": t.ones([2, 1], dtype=t.bool),
            "some_custom_attr": None,
        },
    ]

    @pytest.mark.parametrize("trans", param_test_init)
    def test_init(self, trans, pytestconfig):
        if t.is_tensor(trans["reward"]) and trans["reward"].shape[0] > 1:
            with pytest.raises(ValueError, match="must be 1"):
                _ = Transition(**trans)
        else:
            tb = Transition(**trans)
            tb.to(pytestconfig.getoption("gpu_device"))
