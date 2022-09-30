from datetime import datetime

from pydantic import BaseModel, Field

from config.constants import MetricName, ModelName, SKU


class CollectionResponse(BaseModel):
    collection_name: str
    model_name: ModelName
    metric_name: MetricName
    total_item: int
    created_date: datetime = Field(default_factory=datetime.utcnow)


class ResponseListCollections(BaseModel):
    collection_list: list[CollectionResponse]


class CheckSimilarityResponse(BaseModel):
    file_name: str
    similar_skus: list[str]

class SimilarityResponseFromSKU(BaseModel):
    sku: str
    similar_skus: list[str]


class CreateCollectionLoadImageResult(BaseModel):
    file_name: str
    sku: SKU
    success: bool = True
    message: str = "OK"


class CreateCollectionResponse(BaseModel):
    collection_name: str
    model_name: ModelName
    metric_name: MetricName
    file_name: str
    total_success: int
    total_error: int
    errors: list[CreateCollectionLoadImageResult]


class CreateOrdersFile(BaseModel):
    file_name: str
    total_orders: int
    fail_date_parsing: int
    errors_db: int
    sku_no_images: int


class UserRecommendation(BaseModel):
    user_id: str
    items: list[str]


class AvailableUsers(BaseModel):
    collection_name: str
    users: list[str]
