Text Embedding
---------------

In order to find the part of the graph that's relevant to a query text embeddings are used.

Any feature of a node and edge that is text can be embedded, and during querying these embeddings are compared to the query embedding. The nodes and edges coupled to the *k* embeddings closest to the query embedding are used to build the final context that's inject into the prompt.

Although any feature of a node and edge that is text can be embedded, for retrieving the relevant entities often the embeddings of entity descriptions are used.

See :func:`graphragzen.text_embedding.embed.embed_graph_features` for embedding entity features.

Vector Database
^^^^^^^^^^^^^^^^

After a graph is created, and before querying, the vectors of the entity feature embedding are stored in a vector database. **GraphRAGZen** implements local Qdrant as a vector database backend out of the box, but any backend can be used. 

In order to use your own backend, simply make a class that inherits from :py:class:`graphragzen.text_embedding.vector_databases.VectorDatabase` and implement the methods that are @abstractmethods in the VectorDatabase class (click source to see the methods you'll need to implement)

