from __future__ import annotations  # enable self-referencing type annotations

import datetime

from mongoengine import connect, DateTimeField, DictField, Document, EmbeddedDocument, \
    EmbeddedDocumentListField, EnumField, \
    FileField, FloatField, ListField, MapField, ReferenceField, StringField, IntField, EmbeddedDocumentField

from config.config import MONGO_PORT, MONGO_URL
from config.constants import MetricName, ModelName

connect(host=MONGO_URL, port=MONGO_PORT)


class CollectionMongo(Document):
    collection_name = StringField(primary_key=True)
    model_name = EnumField(required=True, enum=ModelName)
    idx_sku_dict = DictField(required=True)
    annoy_metric = EnumField(required=True, enum=MetricName)
    embedding_file = FileField(required=True)
    annoy_index_file = FileField(required=True)
    created_date = DateTimeField(default=datetime.datetime.now)


# Based on order.csv file
class Orders(Document):
    related_collection = StringField(required=True)
    md5email = StringField(required=True)  # user id
    order_id = StringField(primary_key=True, unique_with='related_collection')
    sku = StringField(required=True)
    country = StringField(required=True)
    zip = StringField(required=True)
    created_at = DateTimeField(required=True)


class User(Document):
    md5email = StringField(primary_key=True)
    annoy_id = IntField(required=True, unique=True)
    created_date = DateTimeField(default=datetime.datetime.now)
    suggested_annoy_index_file = FileField()


class CollectionxRS(Document):
    collection_name = StringField(primary_key=True)
    model_name = EnumField(required=True, enum=ModelName)
    annoy_metric = EnumField(required=True, enum=MetricName)
    created_date = DateTimeField(default=datetime.datetime.now)


class RecommendationCache(Document):
    collection_name = StringField(required=True)
    user_id = StringField(primary_key=True, required=True)
    recommendations = ListField(required=True)
    last_update = DateTimeField(required=True)

#
# class Photo(EmbeddedDocument):
#     image = FileField(required=True)
#     features = ListField(FloatField())
#     created_date = DateTimeField(default=datetime.datetime.now)
#
#
# class Product(Document):
#     sku = StringField(primary_key=True)
#     collection = ReferenceField(CollectionxRS)
#     annoy_id = IntField(required=True, unique=True)
#     photo = EmbeddedDocumentField(Photo)
#     created_date = DateTimeField(default=datetime.datetime.now)
#
#
# class Orders(Document):
#     order_id = StringField(primary_key=True)
#     products_sku = ListField(ReferenceField(Product))
#     purchase_date = DateTimeField(required=True)
#     country = StringField(required=True)
#     zip = StringField(required=True)
#     user = ReferenceField(User, required=True)
#     created_date = DateTimeField(default=datetime.datetime.now)
#
