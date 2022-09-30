from enum import Enum
from pprint import pprint
from typing import Literal, Optional

import numpy as np
import pandas as pd
from annoy import AnnoyIndex
from bidict import bidict

from config.config import EMBEDDING_DIMENSIONS
from config.constants import MetricName, SKU, Columns
from dao.general_dao import Dao
from models.collection import Collection
from models.mongo import CollectionMongo

metrics = Literal['euclidean', 'manhattan', 'dot', 'hammings', 'angular']
methods = Literal["global", "centroid"]


class MethodName(str, Enum):
    GLOBAL = "global"
    CENTROID = "centroid"


class SimilarityRecommender:
    def __init__(
            self,
            collection_name:str,
            threshold: Optional[int] = 2,
            train_months_duration: Optional[int] = 5,
            # modname: str = 'resnet18',
            method: str = 'global',
            # metric: metrics = 'euclidean',
    ):

        self.dao = Dao()
        assert method in ['global', 'centroid'] #'method should be either global or centroid.'
        # assert metric in ['euclidean', 'manhattan', 'dot', 'hammings', 'angular'],
        # 'selected incorrect metric.'
        # self.method = method
        # self.metric = metric
        # self.ife = Imfeatex(modname)
        # self.features_df = self.ife.load_features()
        # self.features_dict = {}
        # for row in self.features_df.iterrows():
        #     self.features_dict[row[1][0]] = row[1][1:].to_numpy()
        self.tot_df, self.train_df, self.test_df = self.dao.temporal_split(
        threshold=threshold, train_months_duration=train_months_duration)

        # self.train_users = self.train_df[COLUMNS().user].unique()
        # self.test_users = self.test_df[COLUMNS().user].unique()

    def _build_annoy_index(
            self,
            user_history: list[SKU],
            embedding_dimension: int,
            metric_name: MetricName,
            features_dict: dict[SKU, list[float]],
    ) -> AnnoyIndex:

        print("features_dict",features_dict)
        t = AnnoyIndex(embedding_dimension, metric=metric_name.value)
        for i, sku in enumerate(features_dict):
            if sku in user_history:
                continue
            t.add_item(i=i, vector=features_dict[sku])
        t.build(n_trees=100, n_jobs=-1)
        return t

    def _build_annoy_index_new(
            self,
            embedding_dimension: int,
            metric_name: MetricName,
            features_dict: dict[int, list[float]],  # annoy_id from Product, features from Photo
    ) -> AnnoyIndex:
        t = AnnoyIndex(embedding_dimension, metric=metric_name.value)
        for annoy_id, features in features_dict.items():
            t.add_item(i=annoy_id, vector=features)
        t.build(n_trees=100, n_jobs=-1)
        return t


    def _skus2recommendations(
            self,
            user_history: list[SKU],
            embedding_dimension: int,
            metric_name: MetricName,
            method_name: MethodName,
            features_dict: dict[SKU, list[float]],
            idx_sku_bidict: bidict[int, SKU],
            k: Optional[int] = 10,
    ) -> list[SKU]:
        t = self._build_annoy_index(
            user_history=user_history,
            embedding_dimension=embedding_dimension,
            features_dict=features_dict,
            metric_name=metric_name,
        )

        if method_name is MethodName.CENTROID:
            user_history_features = [features_dict[sku] for sku in user_history]
            centroid_vector = np.array(pd.DataFrame(user_history_features).mean())
            similar_items = t.get_nns_by_vector(vector=centroid_vector, n=k)
            return [idx_sku_bidict[idx] for idx in similar_items]

        elif method_name is MethodName.GLOBAL:
            recommendation_df = []
            for sku in user_history:
                embedding = features_dict[sku]
                similar_items = t.get_nns_by_vector(
                    vector=embedding, n=k, include_distances=True
                )
                recommendation_df.extend(
                    [idx_sku_bidict[similar_items[0][i]], similar_items[1][i]]
                    for i in range(k)
                )
            df = pd.DataFrame(
                recommendation_df, columns=['items', 'score']
            ).sort_values(by='score')
            df.drop_duplicates(['items'], inplace=True)  # <--------- safe?
            return list(df['items'][: k])

    # todo
    def recommend(self,
                  username: str,
                  embedding_dimension: int,
                  metric_name: str,
                  method_name: str,
                  features_dict: dict[SKU, list[float]],
                  idx_sku_bidict: bidict[int, SKU],
                  k: int = 10):
        user_history = list(self.train_df.loc[self.train_df[
                                                  Columns.user.value] == username,
                                              Columns.item.value])
        #recommended = self._skus2recommendations(user_history=user_history, k=k)
        print(user_history)

        recommended = self._skus2recommendations(user_history=user_history, k=k,
                                                    embedding_dimension= embedding_dimension,
                                                    metric_name=metric_name,
                                                    method_name=method_name,
                                                    features_dict=features_dict,
                                                    idx_sku_bidict = idx_sku_bidict
                                                )
        return recommended



if __name__ == '__main__':
    _collection_mongo = CollectionMongo.objects.get(collection_name="chiara_ferragni")
    collection = Collection.load(_collection_mongo)
    s = SimilarityRecommender(collection_name="chiara_ferragni")

    s.recommend(username="3dd26dbab276aab40740d8443188e319",
                embedding_dimension= EMBEDDING_DIMENSIONS[collection.model_name],
                metric_name=collection.annoy_metric,
                method_name=MethodName.GLOBAL,
                features_dict=collection.embedding,
                idx_sku_bidict = collection.idx_sku_bidict,
                )
    #print(Columns.user)