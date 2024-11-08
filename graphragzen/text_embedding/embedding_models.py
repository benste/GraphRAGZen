import warnings
from abc import ABC, abstractmethod
from typing import List, Optional, Union

import numpy as np
from sentence_transformers import SentenceTransformer


class BaseEmbedder(ABC):

    vector_size: int = 0

    @abstractmethod
    def embed(
        self,
        text: Union[str, List[str]],
        task: Optional[str] = "embed_document",
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        """Text embed strings

        Args:
            text (Union[str, List[str]]): String(s) to embed
            task (str, optional): Some embedding models create different vectors for
                different tasks.
            show_progress_bar (bool, optional): If True shows a progress bar. Defaults to False.

        Returns:
            np.array
        """
        return np.array([])


class NomicTextEmbedder(BaseEmbedder):

    vector_size: int = 768

    def __init__(self, huggingface_URI: str = "nomic-ai/nomic-embed-text-v1.5"):
        """Initialize the nomic text embedder.

        note: Either or both `model_storage_path` or `huggingface_URI` must be set. When both are
        set `model_storage_path` takes precedence.

        Args:
            model_storage_path (str, optional): Path to the model on the local filesystem
            huggingface_URI (str, optional): Huggingface URI of the model.
                Defaults to HF URI "nomic-ai/nomic-embed-text-v1.5".
        """

        print(f"loading {huggingface_URI}")
        self.model = SentenceTransformer(huggingface_URI, trust_remote_code=True)
        self.task_prefix = {
            "embed_document": "search_document: ",
            "embed_query": "search_query: ",
            "clustering": "clustering: ",
            "classification": "classification: ",
        }

    def embed(
        self,
        text: Union[str, List[str]],
        task: Optional[str] = "embed_document",
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        """Text embed strings for a specific task.

        Args:
            text (Union[str, List[str]]): String(s) to embed
            task (str, optional): Should be any of the following, the nomic embedder will
                create vector appropriate to the task:
                ["embed_document", "embed_query", "clustering", "classification"]. Defaults to
                "embed_document".
            show_progress_bar (bool, optional): If True shows a progress bar. Defaults to False.

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

        return self.model.encode(text, show_progress_bar=show_progress_bar)  # type: ignore
