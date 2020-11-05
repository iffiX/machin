from machin.utils.helper_classes import (
    Counter, Trigger, Timer, Switch, Object
)
import pytest


class TestCounter(object):
    def test_counter(self):
        """
        Test counter counter.

        Args:
            self: (todo): write your description
        """
        c = Counter(start=0, step=1)
        c.count()
        assert c.get() == 1
        c.reset()
        assert c.get() == 0
        assert c < 1
        assert c <= 1
        assert c == 0
        assert c > -1
        assert c >= -1
        str(c)


class TestSwitch(object):
    def test_switch(self):
        """
        Test if the switch.

        Args:
            self: (todo): write your description
        """
        s = Switch()
        s.on()
        assert s.get()
        s.off()
        assert not s.get()
        s.flip()
        assert s.get()


class TestTrigger(object):
    def test_trigger(self):
        """
        Set the trigger to the trigger.

        Args:
            self: (todo): write your description
        """
        t = Trigger()
        t.on()
        assert t.get()
        assert not t.get()


class TestTimer(object):
    def test_timer(self):
        """
        Test if the timer is_timer.

        Args:
            self: (todo): write your description
        """
        t = Timer()
        t.begin()
        t.end()


class TestObject(object):
    def test_init(self):
        """
        Initialize a test object.

        Args:
            self: (todo): write your description
        """
        obj = Object()
        assert obj.data == {}
        obj = Object({"a": 1})
        assert obj.data == {"a": 1}

    def test_call(self):
        """
        Called when a test.

        Args:
            self: (todo): write your description
        """
        obj = Object()
        obj("original_call")
        obj.call = lambda _: "pong"
        assert obj("ping") == "pong"

    def test_get_attr(self):
        """
        Get the test attribute.

        Args:
            self: (todo): write your description
        """
        obj = Object({"a": 1})
        with pytest.raises(AttributeError, match="Failed to find"):
            _ = obj.__some_invalid_special_attr__
        assert obj.a == 1

    def test_get_item(self):
        """
        Get the test item.

        Args:
            self: (todo): write your description
        """
        obj = Object({"a": 1})
        assert obj["a"] == 1

    def test_set_attr(self):
        """
        Sets the test attributes.

        Args:
            self: (todo): write your description
        """
        # set data keys
        obj = Object({"a": 1, "const": 0}, const_attrs={"const"})
        obj.a = 1
        assert obj.a == 1
        obj.b = 1
        assert obj.b == 1

        # set const keys
        with pytest.raises(RuntimeError, match="is const"):
            obj.const = 1

        # set .call attribute
        obj.call = lambda _: "pong"
        assert obj("ping") == "pong"

        # set .data attribute
        obj.data = {}
        assert obj.a is None
        with pytest.raises(ValueError, match="must be a dictionary"):
            obj.data = None

        # set other attributes
        with pytest.raises(RuntimeError, match="should not set"):
            obj.__dict__ = {}

    def test_set_item(self):
        """
        Sets the test item as a test.

        Args:
            self: (todo): write your description
        """
        obj = Object({"a": 1})
        obj["a"] = 2
        assert obj.a == 2
