.. _llm_interaction_label:

.. role:: raw-html(raw)
    :format: html

LLM interaction
----------------

**GraphRAGZen** uses an LLM for the following:

* Creating custom prompts for extracting graphs. These prompts are specific to the domain of your documents . 
* Extracting a graph from your documents.
* Summarizing a node or edge feature that has multiple descriptions of that feature. :raw-html:`<br />` (this happens when a node or edge is found multiple times in your documents. The features assigned to this entity are concatenated when creating the graph, where the entity exists only once.)
* Creating descriptions of graph clusters.
* Querying while adding context from the graph.

Two methods are supported to interact with an LLM:

#. By loading a model locally in-memory.
#. With an LLM running on a server through an openAI API compatible endpoint.

A server can be remote or deployed locally depending on your own preference.

Loading a model in-memory uses llama-cpp-python and is unlikely to use your GPU unless configured well. Thus, using locally in-memory is good for development and testing, but for production deployment it is recommended to communicate with an LLM that is properly set-up on a server.

These examples show how the different ways to initiate interaction with an LLM

.. code-block:: python

    from graphragzen.llm import OpenAICompatibleClient
    from graphragzen.llm import Phi35MiniGGUF


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
        model_name = "joeAI/phi3.5:latest"
        context_size = 32768,
        # Caches responses locally, saving time and server load if the same request is made twice
        use_cache=True,
        # Append the cache to a file on disk so it can be re-used between runs.
        cache_persistent=True,
        # The file to store the persistent cache.
        persistent_cache_file="./phi35_mini_instruct_server_persistent_cache.yaml"
    )
    
    # Load model locally in-memory using llama-cpp-python
    llm = Phi35MiniGGUF(
        model_storage_path="path/to/Phi-3.5-mini-instruct-Q4_K_M.gguf",
        tokenizer_URI="microsoft/Phi-3.5-mini-instruct",
        context_size=32786,
        persistent_cache_file="./phi35_mini_instruct_persistent_cache.yaml",
    )

Which model to use
^^^^^^^^^^^^^^^^^^^

The latest OpenAI models give very good results, but might be costly; even with a moderate amount of documents the number of LLM calls to create a graph get large. 

For choosing an open-source model, I found Phi 3.5 mini instruct to give good results in my tests. It's a relatively small model so deployment should be easy and inference fast. When querying your knowledge graph you can switch to a larger, more capable model if desired. 

Though Phi 3.5 mini instruct worked well in my tests, the domain of your documents might show different results. I would advice to extract entities from a small set of your documents, check if the extraction makes sense, and try a different model if it doens't. Pay attention that not just quality nodes are extracted, but also a good amount of edges.

`Phi 3.5 mini instruct Q4 K M <https://huggingface.co/bartowski/Phi-3.5-mini-instruct-GGUF/blob/main/Phi-3.5-mini-instruct-Q4_K_M.gguf>`_

`Gemma 2 2B it Q4 M <https://huggingface.co/bartowski/gemma-2-2b-it-GGUF/blob/main/gemma-2-2b-it-Q4_K_M.gguf>`_

`Gemma 2 9B it Q4 XS <https://huggingface.co/bartowski/gemma-2-9b-it-GGUF/blob/main/gemma-2-9b-it-IQ4_XS.gguf>`_


Async LLM calls
^^^^^^^^^^^^^^^^

When interacting with an LLM on a server, aync LLM calls can significantly speed up the graph generation process.

Functions that make LLM calls have a `async_llm_calls` parameter that, when set to True, will call the LLM async.

The follow functions support this feature:

* extract_raw_entities (:func:`graphragzen.entity_extraction.extract_entities.extract_raw_entities`)
* describe_clusters (:func:`graphragzen.clustering.describe.describe_clusters`)
* generate_entity_relationship_examples (:func:`graphragzen.prompt_tuning.entities.generate_entity_relationship_examples`)


Of the LLM classes, only the `OpenAICompatibleClient` has async implemented, since it's the only class that interacts with an LLM on a server.

When calling it directly for text completion, i.e. llm('input text to complete'), the class checks if it is called in an async context and calls the server async or sync accordingly.

For chat functionality there is an async version, see :func:`graphragzen.llm.openAI_API_client.OpenAICompatibleClient.a_run_chat`


Implementing your own local LLM class
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If the server does not have an OpenAI API compatible endpoint, or you want to load an LLM locally without using llama-cpp-python, you can implement a custom LLM class.

**GraphRAGZen** expects certain methods when calling an LLM. The abstract base class `LLM` defines
the required methods using @abstractmethod; you should inherit from this class when implementing your own LLM implementation.

See :py:class:`graphragzen.llm.base_llm.LLM` (click source)