GraphRAGZen ideology
=========================

Functions: first class citizens
----------------------------------

The python function is king, and the functions are named and located intuitively. 

- Each function has a clear purpose and should be easy to read.
- Required inputs and produced outputs are easily understood; flow of data can be easily traced.
- No wacky behind the scenes magic.

**GraphRAGZen** utilizes semi-pure python functions to maintain a modular and intuitive library.

This means that:

- It does not modifying global variables.
- It does not mutate input.
- If no LLM is not used in a function, the same output is guaranteed for the same input.
    - If an LLM is used this no longer holds (hence semi-pure)

All function inputs are organized according to:

.. code-block:: python

    def somefunction(
            data_from_pipeline: type-hint,
            other_data_from_pipeline: type-hint,
            parameter_1: str = "some_string",
            parameter_2: bool = True,
        )

1. The first *n* inputs are always data as expected from a data-pipeline (loaded documents, LLM
instance, extracted graph, etc.)

2. The later inputs are always parameters. These are the parameters that determing how the function
operates.

3. The parameters all sane default values (if possible, e.g. raw_documents_path cannot have a default)

Modularity
------------
Modularity is achieved by adhering to the following rules

1. Functions should not presume a specific use-case. 

e.g. there is no **load_documents** function that presumes documents are in text format, rather 
there is a **load_text_documents** function.

2. Any interaction with backends (LLM, Embedding Model, Vector Database, etc.) goes through an AbstractBaseClass. 
This way any backend not supported out of the box can be easily implemented by inheriting from the relevant AbstractBaseClass and writing custrom versions of the abstracmethods.

Intuitive
------------
To be intuitive **GraphRAGZen** is organized according to the steps one takes to implement Graph RAG

- load documents
    - load text documents
    - load PDF files (to be implemented)
- preprocessing
    - clean strings
    - chunk documents
- make graph
    - optionally make custom prompts for the domain of the documents
    - extract entities (nodes and edges) using LLM
    - parse extracted entities into a graph
- post-process graph
    - Merge features of entities found multiple times in the documents
    - Cluster graph
    - Describe each cluster
- Embeddings
    - Text Embed the node and edge descriptions
    - Text Embed the cluster describtions
- Query graph
    - Load the Graph and optionally cluster report and source documents
    - Retrieve context relevant to a user input
    - Add context to the user input in a prompt for an LLM