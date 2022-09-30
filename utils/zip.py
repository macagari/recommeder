import zipfile
from io import BytesIO
from os import PathLike
from typing import IO, Iterator, Tuple, Union


def get_zip_files(
        zip_file: Union[str, PathLike[str], IO[bytes]]
) -> Iterator[Tuple[zipfile.ZipInfo, IO[bytes]]]:
    with zipfile.ZipFile(zip_file) as _zip_archive:
        for _zip_file_info in _zip_archive.infolist():
            if not _zip_file_info.is_dir():
                with _zip_archive.open(_zip_file_info) as _zip_file_bytes:
                    yield _zip_file_info, BytesIO(_zip_file_bytes.read())
