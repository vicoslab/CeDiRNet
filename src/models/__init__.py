from models.FPN import FPN
from models.center_estimator import CenterDirEstimator
from models.center_augmentator import CenterDirAugmentator

def get_model(name, model_opts):
    if name == "fpn":
        model = FPN(**model_opts)
    else:
        raise RuntimeError("model \"{}\" not available".format(name))

    return model

def get_center_model(name, model_opts):
    return CenterDirEstimator(model_opts)

def get_center_augmentator(name, model_opts):
    return CenterDirAugmentator(model_opts)
