import logging

import pandas as pd
from lib.data.kaggle import download_dataset, unzip_file
from lib.data.util import scale_df
from src.definitions import EXTERNAL_DATA_FOLDER


def create_loan_default_df(
    logger: logging.Logger = logging.getLogger(__name__),
):
    archive = download_dataset(
        "architsharma01",
        "loan-approval-prediction-dataset",
        EXTERNAL_DATA_FOLDER,
        logger,
    )
    data_path = unzip_file(archive, logger)

    df = pd.read_csv(data_path / "loan_approval_dataset.csv")
    df.columns = df.columns.str.replace(" ", "")
    df = df.drop(columns=["loan_id"])

    bool_to_int = {"Yes": 1, "No": 0}
    status_to_int = {"Approved": 1, "Rejected": 0}
    education_to_int = {"Graduate": 1, "Not Graduate": 0}

    df["education"] = df["education"].str.strip().map(education_to_int)
    df["self_employed"] = df["self_employed"].str.strip().map(bool_to_int)
    df["loan_status"] = df["loan_status"].str.strip().map(status_to_int)

    return df


def scale_loan_default_df(df: pd.DataFrame) -> pd.DataFrame:
    return scale_df(df=df, ignore_columns=["education", "self_employed", "loan_status"])
