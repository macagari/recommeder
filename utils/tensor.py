from typing import Iterable, Optional

import torch
from PIL import Image
from bidict import bidict
from torch import Tensor
from torch.nn import Module
from torchvision import transforms as T

from config.constants import SKU


def _preprocess_image(image: Image.Image, image_size: int) -> Tensor:
    """Preprocesses an Image, resizing and turning it into a Tensor."""
    # image_data can have 4 channels (RGB+alpha transparency channel), but we need
    # just RGB.
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # resize image and convert to tensor
    transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    return transform(image).unsqueeze(0)


def _create_features(image_tensor: Tensor, model: Module) -> Tensor:
    model.eval()
    with torch.no_grad():
        return model(image_tensor)


def process_image(image: Image.Image, image_size: int, model: Module) -> Tensor:
    t = _preprocess_image(image=image, image_size=image_size)
    return _create_features(image_tensor=t, model=model)


def create_embedding(
        image_data: Iterable[tuple[SKU, Image.Image]],
        image_size: int,
        model: Module,
        previous_embedding: Optional[Tensor] = None,
        previous_idx_sku_bidict: Optional[bidict[int, SKU]] = None,
) -> tuple[bidict[int, SKU], Tensor]:

    idx_sku_bidict = bidict() if previous_idx_sku_bidict is None else previous_idx_sku_bidict
    embedding = torch.Tensor() if previous_embedding is None else previous_embedding
    start = max(idx_sku_bidict.keys() or [0])  # handle when the dictionary is empty
    for img_idx, img_data in enumerate(image_data, start=start + 1):
        sku, img = img_data
        idx_sku_bidict[img_idx] = sku
        features = process_image(image=img, image_size=image_size, model=model)
        embedding = torch.cat([embedding, features], dim=0)
    return idx_sku_bidict, embedding


def featuresxRS(image: Image.Image, image_size: int, model: Module) -> list[float]:
    t = _preprocess_image(image=image, image_size=image_size)
    # create embedding for RS
    return model(t).view(-1).tolist()
