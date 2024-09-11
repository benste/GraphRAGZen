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

For initializing interaction with an LLM see :ref:`llm_interaction_label` 

Extract raw entities using a custom output structure:
(By default graphragzen.entity_extraction.extract_raw_entities() already uses a build-in output structure
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