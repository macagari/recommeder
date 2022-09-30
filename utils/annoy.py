from PIL import Image
from annoy import AnnoyIndex
from pymongo.collection import Collection
from torch import Tensor
from torch.nn import Module

from config.constants import MetricName
from models.mongo import CollectionMongo
from utils.tensor import process_image


def create_annoy_index(
        embedding: Tensor, embedding_dimension: int, metric: MetricName
) -> AnnoyIndex:
    t = AnnoyIndex(embedding_dimension, metric.value)
    for idx in range(embedding.size(0)):
        t.add_item(idx, embedding[idx, :])
    t.build(-1)
    return t


def find_k_nn(
        image: Image.Image,
        max_items: int,
        image_size: int,
        model: Module,
        embedding_dimension: int,
        annoy_index: AnnoyIndex,
) -> list[int]:

    features = process_image(image=image, image_size=image_size, model=model)
    # TODO: insert features
    vec = features.view(embedding_dimension)
    return annoy_index.get_nns_by_vector(vec, max_items)

def find_k_nn_(
        sku : str,
        max_items: int,
        annoy_index: AnnoyIndex,
) -> list[int]:

        return annoy_index.get_nns_by_item(1, max_items)

if __name__ == '__main__':
    pass


