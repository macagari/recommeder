import codecs
import csv
import json
import zipfile
from io import BytesIO
from pathlib import Path
from typing import IO, List, Set, Dict
from zipfile import ZipInfo
import re
from bidict import bidict
from fastapi import File, HTTPException, UploadFile
from mongoengine import DoesNotExist, NotUniqueError, ObjectIdField, MultipleObjectsReturned, ValidationError
from PIL import Image, UnidentifiedImageError

from apps.app import app
from config.config import EMBEDDING_DIMENSIONS, IMAGE_SIZES, MODELS
from config.constants import MetricName, ModelName, SKU
from models.collection import Collection
from models.mongo import CollectionMongo, Orders, RecommendationCache
from models.responses import CheckSimilarityResponse, CollectionResponse, \
    CreateCollectionLoadImageResult, CreateCollectionResponse, ResponseListCollections, CreateOrdersFile, \
    UserRecommendation, AvailableUsers, SimilarityResponseFromSKU, DeleteResponse
from recommenders.hybrid_recommender import HybridRecommender
from utils.annoy import find_k_nn
from utils.chunk import chunked
from utils.create_update_agg import update_collection
from utils.logger_utils import stream_logger
from utils.zip import get_zip_files
import datetime
from fastapi import BackgroundTasks, FastAPI
import logging

# logger = logging.getLogger()
logger = stream_logger(__name__)


@app.get("/collections_", tags=['Collection CRUD'])
async def get_collections():
    return [collection.collection_name
            for collection in CollectionMongo.objects
            ]


@app.post("/collections", tags=['Collection CRUD'])
async def get_collections(example_body: Dict):
    logger.debug(example_body)
    return ResponseListCollections(collection_list=[
        CollectionResponse(
            collection_name=collection.collection_name,
            model_name=collection.model_name,
            metric_name=collection.annoy_metric,
            total_item=len(list(collection.idx_sku_dict.values())),
        )
        for collection in CollectionMongo.objects
    ])


@app.post("/create_collection", tags=['Collection CRUD'], response_model=CreateCollectionResponse)
async def create_collection(
        file: UploadFile = File(...),
        args: UploadFile = File(...),
):
    model_name = ModelName.EFFICIENTNET_B1
    metric_name = MetricName.EUCLIDEAN

    args = json.loads((await args.read()).decode())
    collection_name = args.get("new_collection_name")
    logger.debug(collection_name)
    # check if the collection exists
    try:
        _: CollectionMongo = CollectionMongo.objects.get(collection_name=collection_name)
    except DoesNotExist:
        pass
    else:
        raise HTTPException(status_code=415, detail='Collection already exist, use the update endpoint')
    # create a pydantic model to be later exported and saved on mongo
    collection: Collection = Collection.create(
        collection_name=collection_name,
        model_name=model_name,
        annoy_metric=metric_name,

    )

    try:
        # handle images processing in batch
        _zip_archive_bytes = BytesIO(await file.read())
        max_size_batch = 10  # batch  to update the database
        results: list[CreateCollectionLoadImageResult] = []

        for image_data in chunked(get_zip_files(_zip_archive_bytes), max_size_batch):
            results += update_collection(bytes_data=image_data, collection=collection)

    except zipfile.BadZipFile as e:
        raise HTTPException(status_code=415, detail=str(e))

    collection.export().save()

    # --------------------------------------------------
    # RETURN RESULTS
    # --------------------------------------------------
    return CreateCollectionResponse(
        file_name=file.filename,
        total_success=sum(1 for r in results if r.success),
        total_error=sum(1 for r in results if not r.success),
        errors=[r for r in results if not r.success],
        collection_name=collection_name,
        model_name=model_name,
        metric_name=metric_name,
    )


@app.post("/update_collection", tags=['Collection CRUD'], response_model=CreateCollectionResponse)
async def update(
        file: UploadFile = File(...),
        args: UploadFile = File(...),
):
    args = json.loads((await args.read()).decode())
    collection_name = args.get("collection_name")
    logger.debug(collection_name)

    try:
        previous_collection_mongo: CollectionMongo = CollectionMongo.objects.get(collection_name=collection_name)
    except DoesNotExist as e:
        raise HTTPException(status_code=415, detail=str("The collection do not exist, create one first!"))
    else:
        _collection = Collection.load(previous_collection_mongo)

    try:
        # handle images processing in batch
        _zip_archive_bytes = BytesIO(await file.read())
        max_size_batch = 10
        results: list[CreateCollectionLoadImageResult] = []

        for image_data in chunked(get_zip_files(_zip_archive_bytes), max_size_batch):
            results += update_collection(bytes_data=image_data, collection=_collection)

    except zipfile.BadZipFile as e:
        raise HTTPException(status_code=415, detail=str(e))

    _collection.export().save()

    # --------------------------------------------------
    # RETURN RESULTS
    # --------------------------------------------------
    return CreateCollectionResponse(
        file_name=file.filename,
        total_success=sum(1 for r in results if r.success),
        total_error=sum(1 for r in results if not r.success),
        errors=[r for r in results if not r.success],
        collection_name=collection_name,
        model_name=_collection.model_name,
        metric_name=_collection.annoy_metric,
    )


@app.post("/delete_collection", tags=['Collection CRUD'], response_model=DeleteResponse)  # HTTP 204: no content
async def delete_collection(args: Dict):
    # --------------------------------------------------
    # LOAD SEASON
    # --------------------------------------------------

    collection_name = args.get("args").get("collection_name")
    logger.debug(collection_name)
    try:
        collection_mongo: CollectionMongo = CollectionMongo.objects.get(collection_name=collection_name)
    except DoesNotExist:
        raise HTTPException(status_code=422, detail="Collection does not exist!")

    # --------------------------------------------------
    # DELETE GRIDFS FILES, THEN DELETE DOCUMENT
    # --------------------------------------------------
    collection_mongo.embedding_file.delete()
    collection_mongo.annoy_index_file.delete()
    collection_mongo.delete()
    logger.debug("fine")
    return DeleteResponse(msg=f'{collection_name} collection removed!')


@app.post("/find_similar_images", tags=["Search"], response_model=CheckSimilarityResponse)
async def find_similar_images(
        file: UploadFile = File(...),
        args: UploadFile = File(...),
):
    # --------------------------------------------------
    # LOAD SEASON
    # --------------------------------------------------
    args = json.loads((await args.read()).decode())
    collection_name = args.get("collection_name")
    logger.debug(collection_name)
    max_items = int(args.get("max_items"))

    logger.debug(collection_name)
    # logger.debug((await file.read()).decode())
    try:
        _collection_mongo: CollectionMongo = CollectionMongo.objects.get(collection_name=collection_name)
    except DoesNotExist:
        raise HTTPException(status_code=422, detail="Collection does not exist!")
    else:
        collection = Collection.load(_collection_mongo)

    # --------------------------------------------------
    # LOAD IMAGE
    # --------------------------------------------------
    try:
        image = Image.open(BytesIO(await file.read()))
    except UnidentifiedImageError:
        raise HTTPException(status_code=415, detail="File must be an image!")

    # --------------------------------------------------
    # FIND K NEAREST NEIGHBORS
    # --------------------------------------------------
    ids = find_k_nn(
        image=image,
        max_items=max_items,
        image_size=IMAGE_SIZES[collection.model_name],
        model=MODELS[collection.model_name],
        embedding_dimension=EMBEDDING_DIMENSIONS[collection.model_name],
        annoy_index=collection.annoy_index,
    )

    logger.debug(f'ids ---> {ids}')
    # --------------------------------------------------
    # RETURN K SIMILAR SKUS
    # --------------------------------------------------
    return CheckSimilarityResponse(
        file_name=file.filename,
        similar_skus=[collection.idx_sku_bidict[i] for i in ids],
    )


@app.post("/collections/{collection_name}/similar/{sku}", tags=["Search"], response_model=SimilarityResponseFromSKU)
async def find_similar_images_from_sku(
        sku: str = "0HU4007_CFCOOLEYE_BLUEFLUO",
        collection_name: str = "chiara_ferragni",
        max_items: int = 10,
):
    # --------------------------------------------------
    # LOAD COLLECTION
    # --------------------------------------------------
    try:
        _collection_mongo: CollectionMongo = CollectionMongo.objects.get(collection_name=collection_name)

    except DoesNotExist:
        raise HTTPException(status_code=422, detail="Collection does not exist!")

    if sku not in _collection_mongo.idx_sku_dict.values():
        raise HTTPException(status_code=422, detail=f'Sku not in {collection_name} collection ')

    else:
        collection = Collection.load(_collection_mongo)

        idx_ = collection.idx_sku_bidict.inverse[sku]
        return SimilarityResponseFromSKU(
            sku=sku,
            similar_skus=[collection.idx_sku_bidict[i + 1] for i in
                          collection.annoy_index.get_nns_by_item(idx_ - 1, max_items + 1)[1:]]
        )


#############################REC SYSTEM #####################################

def update_cache(
        collection_name: str,
        k: int = 20,
        months_training: int = 5,
        min_item_allowed: int = 2,
):
    ####################################################
    # CALL RECOMMENDER TO GET SUGGESTIONS
    ###################################################
    hr = HybridRecommender(collection_name=collection_name,
                           cf_recommenders=["ALS", "LMF", "BPR"],  # available recommenders
                           cf_quotas=[0.3, 0.5, 0.2],
                           train_month_duration=months_training,
                           threshold=min_item_allowed)

    recommendations_: Dict = hr.recommend_all_users(k=k)
    #########################################################
    # PREPARE SUGGESTIONS TO SAVE AND SAVE CACHE IN DB
    ########################################################
    for user in recommendations_:
        n_items = len(recommendations_[user])
        union_rec = recommendations_[user]
        if n_items < k:
            diff = k - n_items
            # complete the suggestion in the cache with similar images
            rec_similarity: CheckSimilarityResponse = find_similar_images(
                collection_name=collection_name,
                k=diff,
            ).similar_skus

            union_rec += rec_similarity  # add

        cache_rec = RecommendationCache(collection_name=collection_name,
                                        user_id=user,
                                        recommendations=union_rec,
                                        last_update=datetime.datetime.now)

        cache_rec.save()


@app.post("/recsys/orders", tags=["Recommender System"],
          response_model=CreateOrdersFile)  # HTTP 204: no content
async def load_orders(
        file: UploadFile = File(...),
        args: UploadFile = File(...),
):
    logger.debug(f'________________>{file.filename}')
    args = json.loads((await args.read()).decode())
    collection_name = args.get("collection_name_")
    months_training = int(args.get("months_training"))
    min_item_allowed = int(args.get("min_item_allowed"))
    ###########################################################################
    # CHECK IF THE COLLECTION WITH IMAGES HAVE BEEN PREVIOUSLY CREATED
    ##########################################################################
    try:
        orders_ = Orders.objects(related_collection=collection_name)
        # print([i.idx_sku_dict for i in collection][0].values())
    except DoesNotExist:
        logger.debug(f'the collection {collection_name} does not exist!')
        HTTPException(status_code=402, detail=f'{collection_name} does not exist!')
    #################################################################################
    # SET ERROR COUNTERS
    #############################################################################À
    tot_orders = 0
    errors_ = 0
    parsed = 0
    images_excluded = 0
    ###############################################################################
    # INSERT DATASET FROM CSV INTO MONGO DB
    ############################################################################
    # TODO: add check to file csv
    logger.debug(file.file)
    for row in csv.DictReader(codecs.iterdecode(file.file, 'utf-8')):
        logger.debug("---------_>", row)
        tot_orders += 1  # Track orders
        # CHECK THERE IS AT LEAST A VALID DATE
        if not re.search(r'[0-9]{4}-[0-9]{2}-[0-9]{2}', row['created_at']):
            logger.debug(f'date can not be parsed {row["created_at"]=}')
            parsed += 1
            continue
        # CHECK IF THE SKU BELONGS TO THE IMAGE COLLECTION AS WELL
        # TODO: check later if needed
        # if row['sku'] not in set([i.sku for i in orders_]):
        #     images_excluded += 1
        #     logger.warning(f'item can not be added, it is not in the images collection {row["sku"]=}')
        #     continue

        # create record in mongo
        order_name = Orders(md5email=row['md5email'],
                            order_id=row['order_id'] + '_' + row['sku'],
                            sku=row['sku'],
                            country=row['country'],
                            zip=row['zip'],
                            created_at=row['created_at'],
                            related_collection=collection_name)
        try:
            order_name.save()
        except ValidationError:
            HTTPException(detail="""Check content in the file, 'created_at' 
                                  field could be not a correct data format""",
                          status_code=422)
            logger.debug(msg=f'mongo engine was not able to save the record: {row}')
            errors_ += 1
    logger.info(msg=f'Total element without a regular date parsing YYYY-mm-dd{parsed}')
    logger.info(msg=f'Total errors given by MongoDB no admissible {errors_}')
    logger.info(msg=f'Total sku without a corresponding image {images_excluded}')
    logger.info(msg=f'Total orders received {tot_orders}')
    ############################################################################
    # CACHE UPDATE WITH RECOMMENDATIONS
    ###########################################################################
    logger.info("File with orders ok, wait for the cache updating")

    # TODO: CHECK IF THE COLLECTION ORDERS IN MONGO HAS BEEN MODIFIED, IF IT IS NOT THEN DO NOT START THE UPDATE
    update_cache(collection_name=collection_name,
                 months_training=months_training,
                 min_item_allowed=min_item_allowed)

    return CreateOrdersFile(file_name=file.filename,
                            total_orders=tot_orders,
                            fail_date_parsing=parsed,
                            errors_db=errors_,
                            sku_no_images=images_excluded,
                            )


@app.post("/delete/orders", tags=["Recommender System"])  # HTTP 204: no content
async def delete_collection_from_orders(args: Dict):
    collection_name = args.get("args").get("collection_name_")

    try:
        Orders.objects.get(related_collection=collection_name)
        RecommendationCache.objects.get(collection_name=collection_name)
        orders: Orders = Orders.objects(related_collection=collection_name).delete()
        cache: RecommendationCache = RecommendationCache.objects(collection_name=collection_name).delete()
        return f'Orders from the {collection_name} collection were removed!'
    except DoesNotExist:
        HTTPException(status_code=204, detail=f'There are no orders from the {collection_name} collection')
        return f'There are no records from {collection_name} collection!'


@app.post("/recsys/get_user_id", tags=["Recommender System"],
          response_model=AvailableUsers)
async def available_users(args: Dict):
    collection_name = args.get("args").get("collection_name_")

    try:
        cache_users = RecommendationCache.objects.filter(collection_name=collection_name)
        users = [i.id for i in cache_users]
        return AvailableUsers(collection_name=collection_name, users=users)
    except DoesNotExist:
        HTTPException(status_code=402, detail=f'There are no user for {collection_name} collection')


@app.post("/recsys/recommend", tags=["Recommender System"],
          response_model=UserRecommendation)  # HTTP 204: no content
async def recommend_by_user(args: Dict):
    collection_name = args.get("args").get("collection_name_")
    user_id = args.get("args").get("user_id")
    k = int(args.get("args").get("max_items_"))

    ###############################################################################
    # CHECK ERROR CAUSES WITH DB AND INPUT PARAMETERS
    ##############################################################################
    if k <= 0:
        raise HTTPException(status_code=404, detail="k has to be greater than 0")

    # check there are all the information needed in the db
    try:
        _collection_mongo: CollectionMongo = CollectionMongo.objects.get(collection_name=collection_name)
    except DoesNotExist:
        raise HTTPException(status_code=422, detail="Collection does not exist")
    try:
        user_in_orders: Orders = Orders.objects.get(md5email=user_id)
    except DoesNotExist:
        raise HTTPException(status_code=422, detail="User ID does not exist")
    except MultipleObjectsReturned:
        pass
    ##################################################################################
    # PROVIDE SUGGESTION FROM THE CACHE
    #################################################################################
    try:
        _cache_recommendations: RecommendationCache = RecommendationCache.objects.get(collection_name=collection_name,
                                                                                      user_id=user_id)
    except DoesNotExist:
        raise HTTPException(status_code=422, detail="The combination of collection and userID is not correct!")

    return UserRecommendation(user_id=user_id, items=_cache_recommendations.recommendations[:k])


