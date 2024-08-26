from typing import Any, Union

from .embedding_models import NomicTextEmbedder
from .typing import EmbedderLoadingConfig


def load_nomic_embed_text(**kwargs: Union[dict, EmbedderLoadingConfig, Any]) -> NomicTextEmbedder:
    config = EmbedderLoadingConfig(**kwargs)  # type: ignore

    # Get the local model path, and if not provided the huggingface URI
    model_URI_path = config.model_storage_path or config.huggingface_URI

    return NomicTextEmbedder(model_path_or_huggingface_URI=model_URI_path)
