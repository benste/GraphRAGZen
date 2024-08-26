Functions
----------

**GraphRAGZen** utilized pure python functions to maintain a modular and intuitive library.

All function inputs are organized according to 
.. code-block:: python
    def somefunction(
        data_from_pipeline: type-hint,
        other_data_from_pipeline: type-hint,
        **kwargs: Union[dict, PydanticClass, Any],
        )

    config = PydanticClass(**kwargs)

:Dissecting this structure we can see that 
1. The first *n* inputs are always data as expected from a data-pipeline (loaded documents, LLM
instance, extracted graph, etc.)
2. The last input is always *kwargs*. These are the parameters that determing how the function
operates.
3. The *kwargs* is type-hinted with a *dict* AND a *Pydantic class*
4. The first this a function does is passing the *kwargs* to the pydantic class

**Some explanation on points 3 and 4**
By structuring the function this way, parameters can be passed as keyword arguments, or in a single
pydantic class instance (but no longer possitional)

Providing a pydantic class instance has 1 MAJOR advantage; wrongly typed parameters are caucht
before a pipeline starts proccessing.

Secondly, setting parameters this way allows from **GraphRAGZen** to be easily integration in
pipeline frameworks like *Kedro*, which often pass parameters as a {keyword: argument}
dictionary.

.. code-block:: python
    from graphragzen.llm import load_gemma2_gguf
    from graphragzen.llm.typing import LlmLoadingConfig

    # Load using simple keyword arguments
    llm = load_gemma2_gguf(
        model_storage_path="/home/bens/projects/DemystifyGraphRAG/models/gemma-2-2b-it-Q4_K_M.gguf",
        tokenizer_URI="google/gemma-2-2b-it",
    )

    # Load using Pydantic class instance
    # The correct Pydantic class can be found in the function's type hint:
    # `(function) load_gemma2_gguf(**kwargs: Union[dict, LlmLoadingConfig, Any]) -> Gemma2GGUF`
    config = LlmLoadingConfig(
        model_storage_path="/home/bens/projects/DemystifyGraphRAG/models/gemma-2-2b-it-Q4_K_M.gguf",
        tokenizer_URI="google/gemma-2-2b-it",
    )
    llm = load_gemma2_gguf(config=config)