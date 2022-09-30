from pprint import pprint

from bidict import bidict
from fastapi import HTTPException
from mongoengine import DoesNotExist

from config.constants import Columns
from models.mongo import Orders, CollectionMongo, RecommendationCache
import pandas as pd
from typing import List, Optional
from dateutil.relativedelta import relativedelta
import numpy as np


class Dao:
    def __init__(self, collection_name='luisa_spagnoli'):
        self.dict_users_items = orders_to_bidict(collection_name=collection_name)
        self.collection_name = collection_name
        #print(self.dict_users_items)

    def id2user(self,
                idx: int):
        return self.dict_users_items['users'][idx]

    def id2item(self,
                idx: int):
        return self.dict_users_items['items'][idx]

    def user2id(self,
                username: str):
        return self.dict_users_items['users'].inverse[username]

    def item2id(self,
                sku: str):
        # sku refers to the product identifier
        return self.dict_users_items['items'].inverse[sku]

    def get_df(self,
               ids: Optional[bool] = False
               ):

        try:
            orders = Orders.objects(related_collection=self.collection_name)
        except DoesNotExist:
            raise HTTPException(status_code=415, detail='Orders for that collection do not exist!')

        df = pd.DataFrame([i.to_mongo().to_dict() for i in orders])
        df = df.rename(columns={
            "md5email": "users",
            "sku": "items",
            "_id": "orders",
            "created_at": "datetime"
        })
        if ids:
            for col, op in [(Columns.user.value, self.user2id),
                            (Columns.item.value, self.item2id)]:
                df[col] = df[col].apply(op)
        print(f'{df.shape=}')
        # df['datetime'] = pd.to_datetime(df['datetime'],
        #                                 format="'%Y-%m-%dT%H:%M:%SZ",
        #                                 errors="coerce").fillna(
        #                  pd.to_datetime(df['datetime'],
        #                                 format="%Y-%m-%d %H:%M:%S",
        #                                 errors="coerce"))
        #
        # print(min(df['datetime']))

        return df[[i.value for i in list(Columns)]]

    def temporal_split(self,
                       threshold: Optional[int] = 1,
                       train_months_duration: Optional[int] = 10):
        assert threshold >= 0, 'threshold should be positive'
        assert threshold <= 30, 'no users with more than 31 sales'

        # select the users with a minimum number of sales.
        tmp = self.get_df()[['users', 'items']]
        tmp = tmp.groupby(['users']).apply(dict).reset_index()
        tmp.loc[:, 'sales'] = 'NaN'
        for row in tmp.iterrows():
            # print(row[1][0])
            tmp.loc[row[0], 'sales'] = len(row[1][0].get('items').values)
        users_to_use = tmp.loc[tmp['sales'] >= threshold, 'users'].tolist()

        df = self.get_df()[['users', 'items', 'datetime']]
        df = df[df['users'].isin(users_to_use)]

        # Temporal split:
        train_start = min(df['datetime'])
        train_start = pd.to_datetime(train_start)
        test_end = max(df['datetime'])
        test_end = pd.to_datetime(test_end)
        train_end = train_start + relativedelta(months=train_months_duration)
        test_start = train_end + relativedelta(days=1)

        train_selection = df[(df['datetime'] >= train_start) & (df['datetime'] <= train_end)]
        test_selection = df[(df['datetime'] >= test_start) & (df['datetime'] <= test_end)]
        train_users = list(train_selection['users'].unique())
        test_users = list(test_selection['users'].unique())

        # Here we select clients with sales in both train and tesT period
        both_users = list(set(train_users) & set(test_users))
        # print("Number of users both in train and in test: %i" % len(both_users), "\n")
        selected_df = self.get_df(ids=False)[['users', 'items', 'datetime']]

        selected_df = selected_df[selected_df['users'].isin(both_users)]
        df_train = selected_df[(selected_df['datetime'] >= train_start) & (selected_df['datetime'] <= train_end)]
        df_test = selected_df[(selected_df['datetime'] >= test_start) & (selected_df['datetime'] <= test_end)]

        print(f'{selected_df.shape=}, {df_train.shape=}, {df_test.shape=}')
        return selected_df, df_train, df_test


def orders_to_bidict(collection_name: str):
    try:
        orders = Orders.objects(related_collection=collection_name)
    except DoesNotExist:
        raise HTTPException(status_code=415, detail='Orders for that collection do not exist!')

    id2data = dict()
    count = 0
    id2data['users'] = bidict()
    for j, i in enumerate(orders):
        if i.md5email not in id2data['users'].values():
            id2data['users'][j] = i.md5email
            count += 1


    id2data['items'] = bidict({int(k): v for k, v in enumerate(set([i.sku for i in orders]))})

    return id2data  # pprint(id2data)


if __name__ == '__main__':
    pass
    #a = Dao()
    #print(list(a.dict_users_items['users'].values()))
    #print(orders_to_bidict(collection_name='luisa_spagnoli'))
    #print(a.item2id(sku="531141_020200020000"))
    #a.temporal_split()
    #print(a.get_df().head())
    # cache_users = RecommendationCache.objects.filter(collection_name='luisa_spagnoli')
    # users = [i.id for i in cache_users]
    # print(users)