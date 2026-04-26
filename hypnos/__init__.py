__version__ = "0.1.0"


def __getattr__(name):
    if name == "Hypnos":
        from hypnos.core import Hypnos
        return Hypnos
    raise AttributeError(f"module 'hypnos' has no attribute {name!r}")
