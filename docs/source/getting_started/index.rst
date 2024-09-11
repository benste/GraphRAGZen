Getting Started
===================================

Installation
------------

.. code-block:: console

    pip install graphragzen

CLI
----

**GraphRAGZen** is made with developers in mind; it's modularity allows integration with any backend
and parts of the library to be used as needed. Nevertheless, if you just want to create a knowledge graph of your documents and query them there is a CLI to do that.

Create a graph from your documents

.. code-block:: console

    python -m graphragzen.CLI.make_graph --documents_folder "path/to/folder/with/text/files" --project_folder "graphtest"

Query with context from the graph

.. code-block:: console

    python -m graphragzen.CLI.query --project_folder "graphtest"


Using **GraphRAGZen** as a library
-----------------------------------

The following examples are rather intuitive and should get you started (click on source)

Create prompts for extracting graphs that are specific to the domain of your documents
:func:`graphragzen.examples.autotune_custom_prompts.create_custom_prompts`

Create a graph, graph clusters and text embedding vectors
:func:`graphragzen.examples.generate_entity_graph.entity_graph_pipeline`

Query the graph
:func:`graphragzen.examples.query.question`


LLM
----

**GraphRAGZen** relies on an LLM to create a graph from documents. 

Two methods are supported to interact with an LLM:

1. With an LLM running on a server through an openAI API compatible endpoint.
    This server can be remote or deployed locally depending on your own preference.
2. By loading the model locally in-memory.

Loading a model in-memory uses llama-cpp-python and unlikely uses your GPU unless configured well. Thus using in-memory is good for development and testing, but for production deployment it is recommended to communicate with an LLM that is properly set-up on a server.

For more information see :ref:`llm_interaction_label` 

