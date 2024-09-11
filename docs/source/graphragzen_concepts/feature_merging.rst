Feature merging
----------------

When entities are extracted from the documents the same entity (node or edge) can be found multiple
times in different documents.
This means that the same entity will have multiple versions of it's features (e.g. multiple 
descriptions).

It's a good idea to consolidate these multiplicated features for the final graph.

**GraphRAGZen** can 

- Summarize a list of features
- Return the most occuring from a list
- Average a list of features if it's numeric

For descriptions we can ask the LLM to summarize them, edge weights we can simply average, etc. 

See :func:`graphragzen.feature_merging.merge_features.merge_graph_features`