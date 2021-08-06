from . import generators, ROOT
from .archive import Archive
import os
import re


def first(iterable, condition=lambda x: True):
    """
    Returns the first item in the `iterable` that
    satisfies the `condition`.

    If the condition is not given, returns the first item of
    the iterable.

    Raises `StopIteration` if no item satisfying the condition is found.
    """
    return next(x for x in iterable if condition(x))


def generate_all():
    print("Generating all needed data...")
    os.makedirs(os.path.join(ROOT, "generated"), exist_ok=True)
    for gen in dir(generators):
        method = getattr(getattr(generators, gen), "generate", None)
        name = getattr(getattr(generators, gen), "generated_name", None)
        if method is not None and name is not None:
            match = [
                f if re.match(name, f) is not None else None
                for f in os.listdir(os.path.join(ROOT, "generated"))
            ]
            try:
                file = first(match, lambda m: m is not None)
                print(f"Skipping {gen} because file {file} already exists.")
            except StopIteration:
                print(f"Generating {gen}...")
                method()
        else:
            print(
                f"Skipping {gen} because its method({method}) "
                f"or generated name({name}) is None"
            )


def get_all():
    archives = {}
    os.makedirs(os.path.join(ROOT, "generated"), exist_ok=True)
    for gen in dir(generators):
        method = getattr(getattr(generators, gen), "generate", None)
        name = getattr(getattr(generators, gen), "generated_name", None)
        if method is not None and name is not None:
            match = [
                f if re.match(name, f) is not None else None
                for f in os.listdir(os.path.join(ROOT, "generated"))
            ]
            try:
                file = first(match, lambda m: m is not None)
                archives[name] = Archive(path=os.path.join(ROOT, "generated", file))
            except StopIteration:
                raise ValueError(
                    f"Missing generated file for {gen}, please re-run generate_all"
                )
    return archives
