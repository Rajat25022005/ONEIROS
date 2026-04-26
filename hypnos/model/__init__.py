__version__ = "0.1.0"


def __getattr__(name):
    if name == "MambaBackbone":
        from hypnos.model.backbone import MambaBackbone
        return MambaBackbone
    if name == "ThoughtBlock":
        from hypnos.model.thought_block import ThoughtBlock
        return ThoughtBlock
    if name == "EMATeacher":
        from hypnos.model.ema_teacher import EMATeacher
        return EMATeacher
    if name == "LatentDecoder":
        from hypnos.model.decoder import LatentDecoder
        return LatentDecoder
    raise AttributeError(f"module has no attribute {name!r}")
