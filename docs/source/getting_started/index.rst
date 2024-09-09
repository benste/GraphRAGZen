Getting Started
===================================

Installation
------------

.. code-block:: python

    pip install graphragzen

Usage examples
---------------

These examples are rather intuitive and should get you started fast (click on source)

:func:`graphragzen.examples.autotune_custom_prompts.create_custom_prompts`

:func:`graphragzen.examples.generate_entity_graph.entity_graph_pipeline`

:func:`graphragzen.examples.query.question`


LLM
----

**GraphRAGZen** relies on an LLM to create a graph from documents. 

Two methods are supported to interact with an LLM:

1. With an LLM running on a server through an openAI API compatible endpoint.
2. By loading the model locally in-memory; good for testing and development.

It's advised to communicate with an LLM running on a server when deploying in production. The 
local in-memory variant does not support GPU's very well and the models are too slow for practical 
usage other than development locally.

OpenAI API compatible client
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Under the hood **GraphRAGZen** uses the openai python library, which can communicate not only with
the openAI API, but any API that has compatible endpoints:

.. code-block:: python

    from graphragzen.llm import load_openAI_API_client

    # Using OpenAI's API
    llm = load_openAI_API_client(
        api_key_env_variable = "OPENAI_API_KEY"  # the env variable, not the actual key!
        model_name="gpt-4o-mini",
        context_size = 32768,
        # Caches responses locally, saving time and OpenAI credits if the same request is made twice
        use_cache=True,
        # Append the cache to a file on disk so it can be re-used between runs.
        cache_persistent=True,
        # The file to store the persistent cache.
        persistent_cache_file="./OpenAI_persistent_cache.yaml"
    )

    # If you're running your own server running an LLM
    llm = load_openAI_API_client(
        base_url = "http://localhost:8081",
        context_size = 32768,
        # Caches responses locally, saving time and server load if the same request is made twice
        use_cache=True,
        # Append the cache to a file on disk so it can be re-used between runs.
        cache_persistent=True,
        # The file to store the persistent cache.
        persistent_cache_file="./phi35_mini_server_persistent_cache.yaml"
    )

Loading models in-memory
^^^^^^^^^^^^^^^^^^^^^^^^

Out of the box **GraphRAGZen** can load on your local machine `Phi 3.5 mini` and `gemma2` models in gguf format using Llama CPP python. If you want to implement a different model, it's not hard to write your own class for local loading (see below).

.. code-block:: python

    from graphragzen.llm import load_gemma2_gguf

    model_storage_path="path/to/model.gguf"
    tokenizer_URI="google/gemma-2-2b-it" # HF URI, adjust according to your model

    llm = load_gemma2_gguf(
                model_storage_path=model_storage_path,
                tokenizer_URI=tokenizer_URI,
            )
Phi 3.5 mini instruct gave the best results in my tests, but the domain of your documents might show different results. I would advice to extract entities from a very small set of documents, check if the extraction makes sense, and try a different model if it doens't. Pay attention that not just quality nodes are extracted, but also a good amount of edges.

`Phi 3.5 mini instruct Q4 K M <https://huggingface.co/bartowski/Phi-3.5-mini-instruct-GGUF/blob/main/Phi-3.5-mini-instruct-Q4_K_M.gguf>`_

`Gemma 2 2B it Q4 M <https://huggingface.co/bartowski/gemma-2-2b-it-GGUF/blob/main/gemma-2-2b-it-Q4_K_M.gguf>`_

`Gemma 2 9B it Q4 XS <https://huggingface.co/bartowski/gemma-2-9b-it-GGUF/blob/main/gemma-2-9b-it-IQ4_XS.gguf>`_

Implementing your own local LLM instance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can load any LLM you want and **GraphRAGZen** can use it, as long as your implementation defines the following:

(See :func:`graphragzen.llm.llama_cpp_models.BaseLlamCpp` for an example.)

.. code-block:: python

    from graphragzen.llm.base_llm import LLM

    class MyLlmImplementation(LLM):
        def __init__(self) -> None:
            """Write your init as you like, but end it with super()__init__()"""

            super().__init__()

            
        def __call__(
            self, input: Any, output_structure: Optional[BaseModel] = None, **kwargs: Any
        ) -> Any:
            """Call the LLM as you would llm(input), but allow to force an output structure.
            
            If your implementation does not support forcing output structures, simply disregard
            the variable 'output_structure'. 

            Args:
                input (Any): Any input you would normally pass to llm(input, kwargs)
                output_structure (BaseModel): Output structure to force. e.g. grammars from llama.cpp.
                    This SHOULD NOT be an instance of the pydantic model, just the reference.
                    Correct = BaseLlamCpp("some text", MyPydanticModel)
                    Wrong = BaseLlamCpp("some text", MyPydanticModel())
                kwargs (Any): Any keyword arguments you would normally pass to llm(input, kwargs)

            Returns:
                Any
            """

        def run_chat(
            self,
            chat: List[dict],
            max_tokens: int = -1,
            output_structure: Optional[BaseModel] = None,
            stream: bool = False,
        ) -> str:
            """Runs a chat through the LLM

            If your implementation does not support forcing output structures, simply disregard
            the variable 'output_structure'. 

            Args:
                chat (List[dict]): in form [{"role": ..., "content": ...}, {"role": ..., "content": ...
                max_tokens (int, optional): Maximum number of tokens to generate. Defaults to -1.
                output_structure (BaseModel): Output structure to force. e.g. grammars from llama.cpp.
                stream (bool, optional): If True, streams the results to console. Defaults to False.

            Returns:
                str: Generated content
            """

        def tokenize(self, content: str) -> List[str]:
            """Tokenize a string

            Args:
                content (str): String to tokenize

            Returns:
                List[str]: Tokenized string
            """

        def untokenize(self, tokens: List[str]) -> str:
            """Generate a string from a list of tokens

            Args:
                tokens (List[str]): Tokenized string

            Returns:
                str: Untokenized string
            """
            
        def num_chat_tokens(self, chat: List[dict]) -> int:
            """Return the length of the tokenized chat

            Args:
                chat (List[dict]): in form [{"role": ..., "content": ...}, {"role": ..., "content": ...

            Returns:
                int: number of tokens
            """

