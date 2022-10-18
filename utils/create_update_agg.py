from pathlib import Path
from typing import IO
from zipfile import ZipInfo

from PIL import Image, UnidentifiedImageError

from config.constants import SKU
from models.collection import Collection
from models.responses import CreateCollectionLoadImageResult


def update_collection(bytes_data: list[tuple[ZipInfo, IO[bytes]]], collection: Collection) -> list[CreateCollectionLoadImageResult]:
    results: list[CreateCollectionLoadImageResult] = []
    image_data: list[tuple[SKU, Image.Image]] = []
    skus: dict[SKU, str] = {}  # keep track of duplicate SKUs -> filename

    for zip_info, file_bytes in bytes_data:
        # check for duplicate SKUs (a SKU is just the filename without extension)
        _sku = SKU(Path(zip_info.filename).stem)  # remove extension and file path
        if _sku in skus:
            results.append(CreateCollectionLoadImageResult(
                file_name=zip_info.filename,
                sku=_sku,
                success=False,
                message=f"Duplicate SKU found in files! Conflict with: '{skus[_sku]}'.",
            ))
            continue
        elif _sku in collection.idx_sku_bidict.inverse.keys():
            results.append(CreateCollectionLoadImageResult(
                file_name=zip_info.filename,
                sku=_sku,
                success=False,
                message=f"Duplicate SKU found in db: '{_sku}'.",
            ))
            continue
        else:
            skus[_sku] = zip_info.filename

        # try to load image
        try:
            _image = Image.open(file_bytes)
            image_data.append((_sku, _image))
            results.append(CreateCollectionLoadImageResult(
                file_name=zip_info.filename,
                sku=_sku,
                success=True,
            ))
        except UnidentifiedImageError as e:
            results.append(CreateCollectionLoadImageResult(
                file_name=zip_info.filename,
                sku=_sku,
                success=False,
                message=str(e),
            ))

    collection.update(collection_data=image_data)
    return results



