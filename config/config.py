import torchvision.models as models
from environs import Env
import os
from config.constants import ModelName
from models.torch import Identity

env = Env()
env.read_env()  # read .env file, if it exists
MONGO_URL = env("DB", "recommender_mongo")
#MONGO_URL = env("DB", "localhost")
MONGO_PORT = env.int("PORT", 27017)
#os.environ["TORCH_HOME"] = "../torch_models"




EMBEDDING_DIMENSIONS = {
    ModelName.ALEXNET:         9216,
    ModelName.EFFICIENTNET_B1: 1280,
    ModelName.GOOGLENET:       1024,
    ModelName.INCEPTION_V3:    2048,
    ModelName.RESNET18:        512,
    ModelName.SWINTRANSFORMER: 768,
    ModelName.VGG16:           25088,
}

IMAGE_SIZES = {
    ModelName.ALEXNET:         224,
    ModelName.EFFICIENTNET_B1: 224,
    ModelName.GOOGLENET:       224,
    ModelName.INCEPTION_V3:    299,
    ModelName.RESNET18:        224,
    ModelName.SWINTRANSFORMER: 224,
    ModelName.VGG16:           224,
}

MODELS = {
    #ModelName.ALEXNET:         models.alexnet(pretrained=True, progress=False),
    ModelName.EFFICIENTNET_B1: models.efficientnet_b1(pretrained=True, progress=False),
    #ModelName.GOOGLENET:       models.googlenet(pretrained=True, progress=False),
    #ModelName.INCEPTION_V3:    models.inception_v3(pretrained=True, progress=False),
    #ModelName.RESNET18:        models.resnet18(pretrained=True, progress=False),
    # ModelName.SWINTRANSFORMER: None,
    #ModelName.VGG16:           models.vgg16(pretrained=True, progress=False),

}

#MODELS[ModelName.ALEXNET].classifier = Identity()
MODELS[ModelName.EFFICIENTNET_B1].classifier = Identity()
#MODELS[ModelName.GOOGLENET].fc = Identity()
#MODELS[ModelName.INCEPTION_V3].fc = Identity()
#MODELS[ModelName.RESNET18].fc = Identity()
#MODELS[ModelName.VGG16].classifier = Identity()
