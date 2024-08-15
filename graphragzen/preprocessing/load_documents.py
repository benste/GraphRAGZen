import os
from collections import defaultdict
from typing import Any

import pandas as pd

from .typing import LoadTextDocumentsConfig


def load_text_documents(**kwargs: Any) -> pd.DataFrame:
    """loads files from folder path and subfolders.

    Args:
        raw_documents_folder (str): Folder to search for text documents
        raw_content_column (str, optional): Name of the dataframe column to store each document's
            content. Defaults to 'content'.
    Returns:
        pd.DataFrame: Includes the columns 'document_path' and 'document_id'
    """
    config = LoadTextDocumentsConfig(**kwargs)  # type: ignore

    # Walk the folder path, find text files and load them
    folder_path = config.raw_documents_folder
    df = defaultdict(list)
    file_id = 0
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".txt"):
                df["document_path"].append(os.path.join(root, file))
                df[config.raw_content_column].append(open(df["document_path"][-1], "r").read())  # type: ignore  # noqa: E501
                df["document_id"].append(str(file_id))
                file_id += 1

    return pd.DataFrame(df)
