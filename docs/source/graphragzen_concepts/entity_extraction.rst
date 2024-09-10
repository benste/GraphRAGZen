Entity extraction
------------------

RAG relies on retrieving documents related to a query, and adding those to the query to create a final prompt to be send to an LLM.
GraphRAG can also retrieve concepts and relations that are extracted from the documents to add those to the query. These concepts and relations contain denser information and can handle more complex queries.

A graph consists of entities. Entities are the nodes (concepts) and edges (relationships between the concepts). These are extracted from the documents in advance, not during query-time.

To extract a graph, documents are fed into an LLM with a prompt that asks it to:

1. Extract the entities in the document
2. For each entity say if it's a node or and relationship (edge)
3. Give the entity a name
4. Give each entity a category (e.g. concept, location or organization)
5. Give the entity a description
6. Format this all in a json string

These steps are in a single prompt. **GraphRAGZen** does query the LLM a few times per document
to check if all entities have been extracted, so multple json strings can be returned per document.

The output strings are parsed to an actual graph by simply loading the json strings, doing some
simple checks on the contents, and using is as input to networkx.

.. collapse:: example output

    .. code-block:: python

        """"
        [
            {{
                "type": "node",
                "name": "WASHINGTON",
                "category": "LOCATION",
                "description": "Washington is a location where communications are being received, indicating its importance in the decision-making process."
            }},
            {{
                "type": "node",
                "name": "OPERATION: DULCE",
                "category": "MISSION",
                "description": "Operation: Dulce is described as a mission that has evolved to interact and prepare, indicating a significant shift in objectives and activities."
            }},
            {{
                "type": "node",
                "name": "THE TEAM",
                "category": "ORGANIZATION",
                "description": "The team is portrayed as a group of individuals who have transitioned from passive observers to active participants in a mission, showing a dynamic change in their role."
            }},
            {{
                "type": "edge",
                "source": "THE TEAM",
                "target": "WASHINGTON",
                "descripton": "The team receives communications from Washington, which influences their decision-making process.",
                "weight": 1.0
            }},
            {{
                "type": "edge",
                "source": "THE TEAM",
                "target": "OPERATION: DULCE",
                "descripton": "The team is directly involved in Operation: Dulce, executing its evolved objectives and activities.",
                "weight": 1.0
            }}
        ]"""



The prompt used for extraction can be found here :ref:`entity_extraction_prompt_label`

document size
^^^^^^^^^^^^^

While extracting a graph, each document is quickly to large to feed to the LLM in go.
To overcome this we first split the document into smaller chunks and extract the entities
from each chunk. 
It's a good idea to have overlap between the chunks so no entities spanning two chunks are missed.
