Feature merging
----------------

When entities are extracted from the documents the same entity (node or edge) can be found multiple
times.
This means that the same entity will have multiple versions of it's features (e.g. multiple 
descriptions).

It's a good idea to consolidate these multiplicated features for the final graph.

**GraphRAGZen** can 

- Summarize a list of features
- Average a list of features if it's numeric
- Retutn the most occuring from a list

For descriptions we can ask the LLM to summarize them, edge weights we can simply average, etc. 