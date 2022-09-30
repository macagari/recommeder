from models.collection import Collection
from models.mongo import CollectionMongo
from bidict import bidict


_collection_mongo: CollectionMongo = CollectionMongo.objects.get(collection_name='chiara_ferragni')
collection = Collection.load(_collection_mongo)
sku = '0HU4007_CFCOOLEYE_BLUEFLUO'
annoy_index = collection.annoy_index

idx_sku = collection.idx_sku_bidict
idx_ = idx_sku.inverse[sku]
print(idx_)
print(annoy_index.get_nns_by_item(idx_-1, 10))
indec= annoy_index.get_nns_by_item(idx_-1, 10)
print([idx_sku[i+1] for i in indec])





