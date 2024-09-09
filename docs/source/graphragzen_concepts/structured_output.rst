Structured Output
------------------

**GraphRAGZen** relies heavily on valid json strings to be produced by the LLM to extract 
information from the provided documents. 

The default prompts asks the LLM to produce a json string, but the chances of this being a valid
json string can be significantly increased by forcing the LLM to adhere it's output to a schema

JSON Schema
^^^^^^^^^^^^

Many inference solutions allow a schema to be submitted with the prompt to the LLM. The engine than
tries to restrict the token generation such that the output adheres to the schema.

See 
`here <https://platform.openai.com/docs/guides/structured-outputs/examples>`_ for examples using the openAI API

In **GraphRAGZen** both locally loaded LLM's and the API client can be provided with a pydantic
class as an output structure for the LLM to adhere to.

.. note::

   The output structure should NOT be passed as an instance of the pydantic model, just the
    reference.
        Correct = LLM("some text", output_structure=MyPydanticModel)
        Wrong = LLM("some text", output_structure=MyPydanticModel())

Loading an LLM locally or an API client:

.. code-block:: python

    from graphragzen.llm import load_phi35_mini_gguf, load_openAI_API_client

    # Load LLM locally
    llm = load_phi35_mini_gguf(
        model_storage_path="/path/to/Phi-3.5-mini-instruct-Q4_K_M.gguf",
        tokenizer_URI="microsoft/Phi-3.5-mini-instruct",
        persistent_cache_file="./phi35_mini_persistent_cache.yaml",
        context_size=32786,
    )

    # Communicate with OpenAI API endpoint 
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
        persistent_cache_file="./openAI_persistent_cache.yaml"
    )

Extract raw entities using a custom output structure:
(By default graphragzen.entity_extraction.extract_raw_entities() already uses a build-in structure
see :py:class:`graphragzen.entity_extraction.llm_output_structures.ExtractedEntities`)

.. code-block:: python

    from typing import List

    from graphragzen import entity_extraction
    from pydantic import BaseModel


    class ExtractedNode(BaseModel):
        type: str = "node"
        name: str
        category: str
        description: str
        relevance: float
        source_sentence: str

    class ExtractedEdge(BaseModel):
        type: str = "edge"
        source: str
        target: str
        description: str
        weight: float
        source_sentence: str


    class ExtractedEntities(BaseModel):
        extracted_nodes: List[ExtractedNode]
        extracted_edges: List[ExtractedEdge]


    raw_entities = entity_extraction.extract_raw_entities(
        chunked_documents, llm, output_structure: ExtractedEntities
    )