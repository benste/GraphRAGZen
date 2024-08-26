import warnings
from typing import List, Optional, Union

import numpy as np
from sentence_transformers import SentenceTransformer


class BaseEmbedder:

    def embed(
        self, text: Union[str, List[str]], task: Optional[str] = "embed_document"
    ) -> np.ndarray:
        """Text embed strings

        Args:
            text (Union[str, List[str]]): String(s) to embed
            task (Optional[str], optional): Some embedding models create different vectors for
                different tasks.

        Returns:
            np.array
        """
        return np.array([])


class NomicTextEmbedder(BaseEmbedder):
    def __init__(self, model_path_or_huggingface_URI: str = "nomic-ai/nomic-embed-text-v1.5"):

        print(f"loading {model_path_or_huggingface_URI}")
        self.model = SentenceTransformer(model_path_or_huggingface_URI, trust_remote_code=True)
        self.task_prefix = {
            "embed_document": "search_document: ",
            "embed_query": "search_query: ",
            "clustering": "clustering: ",
            "classification": "classification: ",
        }

    def embed(
        self, text: Union[str, List[str]], task: Optional[str] = "embed_document"
    ) -> np.ndarray:
        """Text embed strings for a specific task.

        Args:
            text (Union[str, List[str]]): String(s) to embed
            task (Optional[str], optional): Should be any of the following, the nomic embedder will
                create vector appropriate to the task:
                ["embed_document", "embed_query", "clustering", "classification"]. Defaults to
                "embed_document".

        Returns:
            np.array
        """

        if isinstance(text, str):
            text = [text]

        # Add prefix to the texts to embed
        prefix = self.task_prefix.get(task, "")  # type: ignore
        text = [f"{prefix}{t}" if t else t for t in text]

        if not prefix:
            warnings.warn(
                f"""
task {task} has no prefix defined.
When embedding with nomic-text-embedder it's recommended to add a prefix to obtain the task
specific embedding. See https://huggingface.co/nomic-ai/nomic-embed-text-v1.5
"""
            )

        return self.model.encode(text)  # type: ignore
