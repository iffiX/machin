import os
import re
import datetime
import torch as t


class Archive:
    def __init__(self, path=None, find_in=None, match=None):
        if path is not None:
            self.path = path
        elif find_in is not None and match is not None:
            for f in os.listdir(find_in):
                if (
                    os.path.isfile(os.path.join(find_in, f))
                    and re.match(match, f) is not None
                ):
                    self.path = os.path.join(find_in, f)
                    break
            else:
                raise ValueError(f"Could not find a file in {find_in} matching {match}")
        else:
            raise ValueError(
                "You can either specify a path, or a find path and match pattern."
            )
        self.data = {}

    def add_item(self, key, obj):
        self.data[key] = obj
        return self

    def load(self):
        self.data = t.load(self.path)
        return self

    def save(self):
        t.save(self.data, self.path, pickle_protocol=3)
        return self

    def item(self, key):
        return self.data[key]


def get_time_string():
    return datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
