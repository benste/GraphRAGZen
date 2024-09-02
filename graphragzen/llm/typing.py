from typing import Optional

from pydantic import ConfigDict

from ..typing.MappedBaseModel import MappedBaseModel


class ChatNames(MappedBaseModel):
    """The ChatNames the LLM expects in the prompt

    Args:
        user (str, optional): Name of the user. Defaults to 'user'.
        model (str, optional): Name of the model Defaults to 'assistant'.
    """

    system: str = "system"
    user: str = "user"
    model: str = "model"


class LlmLoadingConfig(MappedBaseModel):
    """Config for loading local LLM

    Args:
        model_storage_path (str): Path to the model on the local filesystem
        tokenizer_URI (str): Huggingface URI for the tokenizer
        context_size (int, optional): Size of the context window in tokens. Defaults to 8192
        use_cache (bool, optional): Use a cache to find output for previously processed inputs in
            stead of re-generating output from the input. Default to True.
        cache_persistent (bool, optional): Append the cache to a file on disk so it can be re-used
            between runs. If False will use only in-memory cache. Default to True
        persistent_cache_file (str, optional): The file to store the persistent cache.
            Defaults to './llm_persistent_cache.yaml'.
    """

    model_config = ConfigDict(protected_namespaces=())

    model_storage_path: str
    tokenizer_URI: str
    context_size: int = 8192
    use_cache: bool = True
    cache_persistent: bool = True
    persistent_cache_file: str = "./llm_persistent_cache.yaml"
    
class LlmAPIClientConfig(MappedBaseModel):
    """Config for initiating a client that communicates with an API that is compatible with the
    openAI API.
    
    Note on tokenizers - The client tries to initiate a tokenizer in the following order, only
        moving on if the previous step failed:
        - Load tokenizer from HF using hf_tokenizer_URI
        - Try to tokenize and de-tokenize using the API endpoints selfbase_url/tokenize
            and base_url/detokenize
        - Try to initiate tiktoken, getting encoding from
            tiktoken.encoding_for_model(model)

    Args:
        base_url (str, optional): url with API endpoints. Not needed if using openAI. Defaults to 
            None.
        model (str, optional): Name of the model to use. Required when using openAI API.
            Defaults to "placeholder_model_name".
        context_size (int): Context size of the model. Defaults to 8192.
        api_key_env_variable (str): Environment variable to read the openAI API key from. 
            Defaults to "OPENAI_API_KEY".
        openai_organization_id (str, optional): Organization ID to use when querying the openAI API.
            Defaults to None.
        openai_project_id (str, optional): Project ID to use when querying the openAI API.
            Defaults to None.
        hf_tokenizer_URI (str, optional): The URI to a tokenizer on HuggingFace. If not provided 
            the API will be tested on the ability to tokenize. If that also fails a tiktoken is 
            initiated.
        max_retries (optional, int): Number of times to retry on timeout. Defaults to 2.
        use_cache (bool, optional): Use a cache to find output for previously processed inputs in
            stead of re-generating output from the input. Default to True.
        cache_persistent (bool, optional): Append the cache to a file on disk so it can be re-used
            between runs. If False will use only in-memory cache. Default to True
        persistent_cache_file (str, optional): The file to store the persistent cache.
            Defaults to './llm_persistent_cache.yaml'.
    """
    
    base_url: Optional[str] = None
    model: Optional[str] = "placeholder_model_name"
    context_size: int = 8192
    api_key_env_variable: str = "OPENAI_API_KEY"
    openai_organization_id: Optional[str] = None
    openai_project_id: Optional[str] = None
    hf_tokenizer_URI: Optional[str] = None
    max_retries: Optional[int] = 2
    use_cache: bool = True
    cache_persistent: bool = True
    persistent_cache_file: str = "./phi35_mini_persistent_cache.yaml"
    