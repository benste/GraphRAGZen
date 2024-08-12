import os
from collections import defaultdict

import pandas as pd

from graphragzen.llm.base_llm import LLM
from graphragzen.preprocessing import utils
from graphragzen.typing import preprocessing


def raw_documents(config: preprocessing.RawDocumentsConfig) -> pd.DataFrame:
    """
      loads files from folder path and subfolders.
    """
    
    # Walk the folder path, find text files and load them
    folder_path = config.raw_documents_folder
    df = defaultdict(list)
    file_id = 0
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".txt"):
                df["document_path"].append(os.path.join(root, file))
                df[config.raw_content_column].append(open(df["document_path"][-1], "r").read())
                df["document_id"].append(str(file_id))
                file_id += 1
        
    return pd.DataFrame(df)

def chunk_documents(dataframe: pd.DataFrame, llm: LLM, config: preprocessing.ChunkConfig) -> pd.DataFrame:
    results_column = config.results_column
    len_column = config.results_column + '_len'
    id_column = config.results_column + '_id'
    
    # Apply chunking per document, also saving the number of tokens in each chunk
    dataframe[results_column], dataframe[len_column] = zip(*dataframe[config.column_to_chunk].apply(lambda c: utils.chunk_tokens(c, llm, config.window_size, config.overlap)))
    
    # Map each chunk back to the correct row
    dataframe = dataframe.explode([results_column, len_column])
    
    # Give each chunk a unique ID
    dataframe[id_column] = list(range(len(dataframe)))
    
    # TODO: drop content column to save space?
    
    return dataframe
    