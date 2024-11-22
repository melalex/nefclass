import logging
from pathlib import Path
import zipfile
import kaggle

from pandas import DataFrame, Series
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

from lib.data.panda_ds import PandasDs


def download_dataset(owner: str, name: str, dest: Path, logger: logging.Logger) -> Path:
    dest.mkdir(parents=True, exist_ok=True)

    file_path = dest / f"{name}.zip"

    if file_path.is_file():
        logger.info(
            "Found [ %s ] dataset in [ %s ]. Skipping download...", name, file_path
        )
    else:
        logger.info("Downloading [ %s ] dataset to [ %s ]", name, file_path)
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(dataset=f"{owner}/{name}", path=dest)

    return file_path


def unzip_file(archive: Path, logger: logging.Logger) -> Path:
    archive_name = archive.stem
    dest_file = archive.parent / archive_name

    if dest_file.exists():
        logger.info("[ %s ] is already unzipped. Skipping ...", dest_file)
    else:
        logger.info("Unzipping [ %s ] to [ %s ]", archive, dest_file)
        with zipfile.ZipFile(archive, "r") as zip_ref:
            zip_ref.extractall(dest_file)

    return dest_file


def dataset_split(
    x: DataFrame,
    y: Series,
    test_train_ratio: float,
    valid_train_ratio: float,
    random_seed: int,
    batch_size: int = 256,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    all_x_train, x_test, all_y_train, y_test = train_test_split(
        x, y, test_size=test_train_ratio, random_state=random_seed
    )

    x_train, x_valid, y_train, y_valid = train_test_split(
        all_x_train, all_y_train, test_size=valid_train_ratio, random_state=random_seed
    )

    return (
        DataLoader(PandasDs(x_train, y_train), batch_size=batch_size),
        DataLoader(PandasDs(x_valid, y_valid), batch_size=batch_size),
        DataLoader(PandasDs(x_test, y_test), batch_size=batch_size),
    )
