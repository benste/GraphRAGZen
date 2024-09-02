import os
from collections import defaultdict
from typing import Optional

import pandas as pd


def load_text_documents(
    raw_documents_folder: str,
    raw_content_column: Optional[str] = "content",
) -> pd.DataFrame:
    """loads files from folder path and subfolders as raw text.

    Args:
        raw_documents_folder (str): Folder to search for text documents
        raw_content_column (str, optional): Name of the dataframe column to store each document's
            content. Defaults to 'content'.
    Returns:
        pd.DataFrame: Includes the columns 'document_path' and 'document_id'
    """
    # config = LoadTextDocumentsConfig(**kwargs)  # type: ignore

    # Walk the folder path, find text files and load them
    folder_path = raw_documents_folder
    df = defaultdict(list)
    file_id = 0
    for root, _, files in os.walk(folder_path):
        for file in files:
            try:
                df["document_path"].append(os.path.join(root, file))
                df[raw_content_column].append(open(df["document_path"][-1], "r").read())  # type: ignore  # noqa: E501
                df["document_id"].append(str(file_id))
                file_id += 1
            except Exception as e:
                del df["document_path"][-1]
                print(f"Could not load {os.path.join(root, file)}: {e}")

    return pd.DataFrame(df)
