from pydantic import ConfigDict

from ..typing.MappedBaseModel import MappedBaseModel


class ChatNames(MappedBaseModel):
    """The ChatNames the LLM expects in the prompt

    Args:
        user (str, optional): Name of the user. Defaults to 'user'.
        model (str, optional): Name of the model Defaults to 'assistant'.
    """

    user: str = "user"
    model: str = "assistant"


class LlmLoadingConfig(MappedBaseModel):
    """Config for loading local LLM

    Args:
        model_storage_path (str): Path to the model on the local filesystem
        tokenizer_URI (str): URI for the tokenizer
        context_size (int, optional): Size of the context window in tokens. Defaults to 8192
    """

    model_config = ConfigDict(protected_namespaces=())

    model_storage_path: str
    tokenizer_URI: str
    context_size: int = 8192
