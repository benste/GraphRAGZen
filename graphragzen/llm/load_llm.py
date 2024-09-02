from typing import Optional

from graphragzen.llm import llama_cpp_models, openAI_API_client


def load_gemma2_gguf(
    model_storage_path: str,
    tokenizer_URI: str,
    context_size: int = 8192,
    use_cache: bool = True,
    cache_persistent: bool = True,
    persistent_cache_file: str = "./llm_persistent_cache.yaml",
) -> llama_cpp_models.Gemma2GGUF:
    """Load gguf version of Gemma 2

    Args:
        model_storage_path (str): Path to the model on the local filesystem
        tokenizer_URI (str): URI for the tokenizer
        context_size (int, optional): Size of the context window in tokens. Defaults to 8192
        use_cache (bool, optional): Use a cache to find output for previously processed inputs in
            stead of re-generating output from the input. Default to True.
        cache_persistent (bool, optional): Append the cache to a file on disk so it can be re-used
            between runs. If False will use only in-memory cache. Default to True
        persistent_cache_file (str, optional): The file to store the persistent cache.
            Defaults to './llm_persistent_cache.yaml'.

    Returns:
        Gemma2GGUF: see `graphragzen.llm.gemma2.Gemma2GGUF`
    """

    return llama_cpp_models.Gemma2GGUF(
        model_storage_path=model_storage_path,
        tokenizer_URI=tokenizer_URI,
        context_size=context_size,
        use_cache=use_cache,
        cache_persistent=cache_persistent,
        persistent_cache_file=persistent_cache_file,
    )


def load_phi35_mini_gguf(
    model_storage_path: str,
    tokenizer_URI: str,
    context_size: int = 8192,
    use_cache: bool = True,
    cache_persistent: bool = True,
    persistent_cache_file: str = "./llm_persistent_cache.yaml",
) -> llama_cpp_models.Phi35MiniGGUF:
    """Load gguf version of Phi 3.5 mini

    Args:
        model_storage_path (str): Path to the model on the local filesystem
        tokenizer_URI (str): URI for the tokenizer
        context_size (int, optional): Size of the context window in tokens. Defaults to 8192
        use_cache (bool, optional): Use a cache to find output for previously processed inputs in
            stead of re-generating output from the input. Default to True.
        cache_persistent (bool, optional): Append the cache to a file on disk so it can be re-used
            between runs. If False will use only in-memory cache. Default to True
        persistent_cache_file (str, optional): The file to store the persistent cache.
            Defaults to './llm_persistent_cache.yaml'.

    Returns:
        Phi35MiniGGUF: see `graphragzen.llm.phi35.Phi35MiniGGUF`
    """

    return llama_cpp_models.Phi35MiniGGUF(
        model_storage_path=model_storage_path,
        tokenizer_URI=tokenizer_URI,
        context_size=context_size,
        use_cache=use_cache,
        cache_persistent=cache_persistent,
        persistent_cache_file=persistent_cache_file,
    )


def load_openAI_API_client(
    base_url: Optional[str] = None,
    model_name: str = "placeholder_model_name",
    context_size: int = 8192,
    api_key_env_variable: str = "OPENAI_API_KEY",
    openai_organization_id: Optional[str] = None,
    openai_project_id: Optional[str] = None,
    hf_tokenizer_URI: Optional[str] = None,
    max_retries: int = 2,
    use_cache: bool = True,
    cache_persistent: bool = True,
    persistent_cache_file: str = "./phi35_mini_persistent_cache.yaml",
) -> openAI_API_client.OpenAICompatibleClient:
    """Initiate a client that can communicate with OpenAI compatible API endpoints.
    e.g. llama.cpp server is mostly OpenAI API compatible.

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
        model_name (str, optional): Name of the model to use. Required when using openAI API.
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

    Returns:
        openAI_API_client.OpenAICompatibleClient
    """

    return openAI_API_client.OpenAICompatibleClient(
        base_url,
        model_name,
        context_size,
        api_key_env_variable,
        openai_organization_id,
        openai_project_id,
        hf_tokenizer_URI,
        max_retries,
        use_cache,
        cache_persistent,
        persistent_cache_file,
    )


def load_gemma2_huggingface() -> None:
    """NOT YET IMPLEMENTED"""
    # TODO: implement
    pass


def load_openAI() -> None:
    """NOT YET IMPLEMENTED"""
    # TODO: implement
    pass
