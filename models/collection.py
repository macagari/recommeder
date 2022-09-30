from __future__ import annotations

import io
from datetime import datetime
from tempfile import NamedTemporaryFile
from typing import Iterable, Optional

import torch
from PIL.Image import Image
from annoy import AnnoyIndex
from bidict import bidict
from pydantic import BaseModel, Field
from torch import Tensor

from config.config import EMBEDDING_DIMENSIONS, IMAGE_SIZES, MODELS
from config.constants import MetricName, ModelName, SKU
from models.mongo import CollectionMongo
from utils.annoy import create_annoy_index
from utils.tensor import create_embedding


class Collection(BaseModel):
    collection_name: str
    model_name: ModelName
    annoy_metric: MetricName
    idx_sku_bidict: Optional[bidict[int, SKU]] = None
    annoy_index: Optional[AnnoyIndex] = None
    embedding: Optional[Tensor] = None
    created_date: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def create(
            cls,
            collection_name: str,
            # collection_data: Iterable[tuple[SKU, Image]],  # todo: remove
            model_name: ModelName = ModelName.EFFICIENTNET_B1,
            annoy_metric: MetricName = MetricName.EUCLIDEAN,
            # previous_embedding: Optional[Tensor] = None,  # todo: remove
            # previous_idx_sku_bidict: Optional[bidict[int, SKU]] = None,  # todo: remove

    ) -> Collection:
        # idx_sku_bidict, embedding = create_embedding(
        #     image_data=collection_data,
        #     image_size=IMAGE_SIZES[model_name],
        #     model=MODELS[model_name],
        #     previous_embedding=previous_embedding,
        #     previous_idx_sku_bidict=previous_idx_sku_bidict,
        # )
        #
        # annoy_index = create_annoy_index(
        #     embedding=embedding,
        #     embedding_dimension=EMBEDDING_DIMENSIONS[model_name],
        #     metric=annoy_metric,
        # )
        return cls(
            collection_name=collection_name,
            model_name=model_name,
            annoy_metric=annoy_metric,
            idx_sku_bidict=bidict(),
        )
        # return cls(
        #     collection_name=collection_name,
        #     model_name=model_name,
        #     idx_sku_bidict=idx_sku_bidict,
        #     annoy_metric=annoy_metric,
        #     annoy_index=annoy_index,
        #     embedding=embedding,
        # )

    @classmethod
    def load(cls, collection: CollectionMongo) -> Collection:
        # convert dict saved in mongo to bidict and convert keys back to int
        idx_sku_dict = bidict({int(k): SKU(v) for k, v in collection.idx_sku_dict.items()})
        # load embedding
        embedding = torch.load(io.BytesIO(collection.embedding_file.read()))
        # Loading AnnoyIndex only possible with filesystem, workaround described here:
        # https://arxiv.org/pdf/2204.07922.pdf
        # https://medium.com/abnormal-security-engineering-blog/lazily-loading-ml
        # -models-for-scoring-with-pyspark-a167d4deed3c#d384
        annoy = AnnoyIndex(EMBEDDING_DIMENSIONS[collection.model_name], collection.annoy_metric)
        with NamedTemporaryFile(suffix='.ann') as tf:
            tf.write(collection.annoy_index_file.read())
            annoy.load(tf.name)

        return cls(
            collection_name=collection.collection_name,
            model_name=collection.model_name,
            idx_sku_bidict=idx_sku_dict,
            annoy_metric=collection.annoy_metric,
            annoy_index=annoy,
            embedding=embedding,
            created_date=collection.created_date,
        )

    def update(self, collection_data: Iterable[tuple[SKU, Image]]) -> Collection:

        idx_sku_bidict, embedding = create_embedding(
            image_data=collection_data,
            image_size=IMAGE_SIZES[self.model_name],
            model=MODELS[self.model_name],
            previous_embedding=self.embedding,
            previous_idx_sku_bidict=self.idx_sku_bidict,
        )

        annoy_index = create_annoy_index(
            embedding=embedding,
            embedding_dimension=EMBEDDING_DIMENSIONS[self.model_name],
            metric=self.annoy_metric,
        )

        self.idx_sku_bidict = idx_sku_bidict
        self.embedding = embedding
        self.annoy_index = annoy_index

        return self

    def export(self, collection: Optional[CollectionMongo] = None) -> CollectionMongo:
        collection = collection or CollectionMongo()

        # convert bidict to dict and convert keys to str
        idx_sku_dict = {str(k): str(v) for k, v in self.idx_sku_bidict.items()}

        collection.collection_name = self.collection_name
        collection.model_name = self.model_name
        collection.annoy_metric = self.annoy_metric
        collection.idx_sku_dict = idx_sku_dict
        collection.created_date = self.created_date
        # export embedding (using filesystem -- file buffer wasn't working)
        with NamedTemporaryFile(suffix='.pt') as tf:
            torch.save(self.embedding, tf.name)
            with open(tf.name, "rb") as f:
                collection.embedding_file.replace(f, content_type='application/x-binary')
        # Saving AnnoyIndex only possible with filesystem, workaround described here:
        # https://medium.com/abnormal-security-engineering-blog/lazily-loading-ml
        # -models-for-scoring-with-pyspark-a167d4deed3c#d384
        with NamedTemporaryFile(suffix='.ann') as tf:
            self.annoy_index.save(tf.name)
            with open(tf.name, "rb") as f:
                collection.annoy_index_file.replace(f, content_type='application/x-binary')

        return collection
