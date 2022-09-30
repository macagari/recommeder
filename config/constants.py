import json
from enum import Enum
from pathlib import Path
from typing import NewType

SKU = NewType('SKU', str)

#with open(Path.cwd().parent.parent / "ppom.json") as ppom:
#    PPOM = json.load(ppom)

with open(Path.cwd().parent/"ppom.json") as ppom:
    PPOM = json.load(ppom)

VERSION = PPOM["version"]


class ModelName(str, Enum):
    # hModelNamettps://fastapi.tiangolo.com/tutorial/path-params/#create-an-enum-class
    ALEXNET = "alexnet"
    EFFICIENTNET_B1 = "efficientnet_b1"
    GOOGLENET = "googlenet"
    INCEPTION_V3 = "inception_v3"
    RESNET18 = "resnet18"
    SWINTRANSFORMER = "swintransformer"
    VGG16 = "vgg16"
    BIT = "bit"


class MetricName(str, Enum):
    # https://fastapi.tiangolo.com/tutorial/path-params/#create-an-enum-class
    ANGULAR = "angular"
    DOT = "dot"
    EUCLIDEAN = "euclidean"
    HAMMING = "hamming"
    MANHATTAN = "manhattan"

class Columns(str, Enum):

    user = 'users'
    item = 'items'
    order = 'orders'
    datetime = 'datetime'

