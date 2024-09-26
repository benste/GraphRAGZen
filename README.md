# GraphRAGZen
**GraphRAGZen** is a functional, common sense, library for developing GraphRAG pipelines.

A big **thank you** to the team at microsoft that created [GraphRAG](https://github.com/microsoft/graphrag), which is the foundation for this project.

[You can find the documentation here](https://benste.github.io/GraphRAGZen/)

# Motivation
[GraphRAG from Microsoft](https://github.com/microsoft/graphrag) makes it easy to get started with 
automated Graph RAG.

Sadly, the codebase is hard to read, difficult to adapt, and dataflow
in the pipelines near impossible to follow.

The work done by the GraphRAG team shouldn’t be in vain, but should
allow for developers to create GraphRAG applications that are
maintainable, extendable and intuitive.

And so GraphRAGZen was born; the logic given by GraphRAG in a
functional, common sense library...

## But that's just another Graph RAG library, right?
[XKCD says it well](https://xkcd.com/927/)

It's early days for GraphRAG technology, and there's no standard implementation method accepted at 
large. This shows that there's still a lack of high quality, easy to implement solutions in the 
OSS space.

From experience I found that most libraries available either provide a limited set
of abilities with the bulk hiding behind convoluted code (i.e. GraphRAG)
or are unintuitive and rely heavily on extended documentation.

This should be easily mitigated by writing a functional library. That
is, the python function is king, and the functions are named and located intuitively.

When that is established we have a toolbox that can be used as seen fit
by developers. It is modular, extendable, maintainable and intuitive.


# Features
**Knowledge graph generation**
- Extract nodes and edges from your supplied documents
- Merge the nodes and edges that are found multiple times over your documents
- Create clusters of nodes and describe each cluster

**Text Embedding**
- Embed any graph feature that is a string, this is used to find the parts of the graph that are relevant to a query
- Qdrant local vector database for storing and vector search

**LLM interaction**
- Local in-memory LLM's using llama-cpp-python
- LLM's running on servers with OpenAI API compatible endpoints
- Async calls to LLM's running on servers
- Structured output, i.e. force json output, optionally with a specific structure ([these](https://platform.openai.com/docs/guides/structured-outputs/examples) examples explains it well)

**Querying**
- Create a prompt from the nodes and edges relevant to a query
- Optionally add cluster descriptions relevant to the query
- Optionally add the source documents used to create the most relevant nodes
