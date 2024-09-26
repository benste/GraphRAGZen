Querying: Local Search
-----------------------

The Graph RAG library provides a robust mechanism for querying and retrieving context-relevant information by leveraging graph data, vector searches, and optionally source documents and cluster reports. The querying process is designed to find the most relevant information from a heterogeneous mix of data sources and to format this information into a coherent prompt that's ready to send to an LLM.

Querying can be devided in local search (specific information from a small part of the graph) and global search (information from summaries of the whole graph)

see the following example on how to query using your graph for added context (click on source)
:func:`graphragzen.examples.query.question`

Overview of the Local Search Process
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The querying for local serach process in **GraphRAGZen** integrates several components to construct a comprehensive response to a user's query. This process involves:

- Semantic Vector Search: A semantic search is performed using vector representations of entities and the query. This step retrieves entities from the vector database that are most similar to the query based on their embeddings. The vector search is controlled by parameters like the number of similar entities to retrieve (top_k_similar_entities) and a score threshold (score_threshold) that filters out entities below a certain similarity score.

- Graph Context Retrieval: Once the initial set of similar entities is identified, additional contextual information can be optionally retrieved from the graph:

    - Inside Edges: Edges whose both nodes are already present among the similar entities. This helps in identifying strong internal relationships within the group of entities that are relevant to the query.

    - Outside Edges: Edges where exactly one node is present among the similar entities. This step helps to explore peripheral relationships that might add contextual relevance to the retrieved entities.

- Source Document Integration: If source documents are provided, they sources of the entities retrieved in the *Semantic Vector Search* step are added to the context. The number of such texts to include can be controlled by the `top_k_source_documents` parameter.

- Cluster Report Summaries: These are summarizations of clustered graph nodes. These summaries can help in understanding broader trends or patterns that relate to the query and the retrieved entities. If available, cluster summaries coupled to the entities retrieved in the *Semantic Vector Search* step are added to the context. The number of cluster descriptions to include can be set using the `top_k_cluster_descriptions`` parameter.

Prompt Construction for Local Search
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

After gathering all relevant data, a final prompt is constructed. The prompt combines:

- Node Descriptions: Information about nodes retrieved from the graph, including their category and a brief description.

- Relationship Descriptions: Descriptions of edges (relationships) within and outside the set of similar entities.

- Specific Source Contexts (optional): Direct excerpts or references from source documents.

- Global Source Contexts (optional): Summaries from cluster reports.

The prompt is formatted using a base prompt template, which is customized with the contextual data and the original query. This ensures that the constructed prompt is rich with relevant context, enhancing the accuracy and relevance of any subsequent operations, such as generating responses or extracting insights.
