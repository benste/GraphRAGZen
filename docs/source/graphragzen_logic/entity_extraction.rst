Entity extraction
------------------

Entities are the nodes and edges of a graph. 

Documents are fed into an LLM with a prompt that asks it to:

1. Extract the entities in the document
2. For each entity say if it's a node or and relationship (edge)
3. Give the entity a name
4. Give each entity a type (e.g. concept or organization)
5. Give the entity a description
6. Format this all in a structured manner

These steps are in a single prompt and a single string is returned per document.

The output strings are parsed to an actual graph by simply splitting on the delimiters and using it 
as an input to networkx.

.. collapse:: example output

    .. code-block:: python

        """"##\n("entity"<|>"Machine Learning"<|>"concept"<|>"Machine Learning is a subset of Artificial Intelligence, focusing on developing algorithms that enable computers to perform tasks without explicit instructions.")##\n("entity"<|>"Artificial Intelligence"<|>"concept"<|>"Artificial Intelligence is a field of study focused on developing intelligent machines that can perform tasks that typically require human intelligence.")<|COMPLETE|>"""



The prompt used for extraction can be found here :ref:`entity_extraction_prompt_label`

**document size**

Typically you cannot feed a whole document into the LLM due to the size of the document. To 
circumvent we first split the document into smaller parts and extract the entities from each part. 
It's a good idea to have overlap between the splits when splitting the documents, so no entities 
span two document chunks are missed.
