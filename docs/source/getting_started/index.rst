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

OpenAI API compatible client
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Under the hood **GraphRAGZen** uses the openai python library, which can communicate not only with
the openAI API, but any API that has compatible endpoints:

.. code-block:: python

    from graphragzen.llm import OpenAICompatibleClient

    # Using OpenAI's API
    llm = OpenAICompatibleClient(
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
    llm = OpenAICompatibleClient(
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
^^^^^^^^^^^^^^^^^^^^^^^^^

Out of the box, without deplopying a local server, **GraphRAGZen** can load GGUF models on your
local machine using Llama CPP python.
`Phi 3.5 mini` and `gemma2` models have been tested, but you can try any GGUF model.
If you want to implement a different model type (e.g. driectly from HF using transformers), it's not
hard to write your own class for local loading (see below).

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


Implementing your own local LLM class
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**GraphRAGZen** expects certain methods when calling an LLM. The abstract base class `LLM` defines
the required methods using @abstractmethod; you should inherit from this class when implementing your own LLM implementation.

See :py:class:`graphragzen.llm.base_llm.LLM` (click source)
