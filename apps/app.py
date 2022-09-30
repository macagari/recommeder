# metadata for services
from fastapi import FastAPI

from config.constants import VERSION

tags_metadata = [
    {
        "name":        "Collection CRUD",
        "description": "Operations to handle CRUD for fashion collections (Ex: Luisa "
                       "Spagnoli, Chiara Ferragni)",
    },
    {
        "name":        "Search",
        "description": "Returns similar images from a given fashion collection",
    },
    {
        "name":         "Recommender System",
        "description":  "Operations to handle the FAIRE Hybrid recommender system",
    }
]

app = FastAPI(
    title="FAIRE Search Tool",
    version="0.0.1",
    description="",
    openapi_tags=tags_metadata,
)
