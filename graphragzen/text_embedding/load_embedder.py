from typing import Any, Union

from .embedding_models import NomicTextEmbedder
from .typing import EmbedderLoadingConfig


def load_nomic_embed_text(**kwargs: Union[dict, EmbedderLoadingConfig, Any]) -> NomicTextEmbedder:
    """Load the nomic text embedder.

    note: Either or both `model_storage_path` or `huggingface_URI` must be set. When both are
    set `model_storage_path` takes precedence.

    Args:
        model_storage_path (str, optional): Path to the model on the local filesystem
        huggingface_URI (str, optional): Huggingface URI of the model.
            Defaults to "nomic-ai/nomic-embed-text-v1.5".

    Returns:
        NomicTextEmbedder
    """
    config = EmbedderLoadingConfig(**kwargs)  # type: ignore

    # Get the local model path, and if not provided the huggingface URI
    model_URI_path = config.model_storage_path or config.huggingface_URI

    return NomicTextEmbedder(model_path_or_huggingface_URI=model_URI_path)
