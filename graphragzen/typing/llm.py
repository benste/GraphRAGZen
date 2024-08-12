from pydantic import BaseModel

class ChatNames(BaseModel):
    user: str = "user"
    model: str = "assistant"
    
class LlmLoadingConfig(BaseModel):
    """Config for loading local LLM

    Args:
        model_storage_path (str): Path to the model on the local filesystem
        tokenizer_URI (str): URI for the tokenizer
        context_size (int, optional): Size of the context window in tokens. Defaults to 8192
    """

    model_storage_path: str
    tokenizer_URI: str
    context_size: int = 8192
