# GraphRAGZen
Logic of GraphRAG in a functional, non-nonsense library.

A big thank you to the team at microsoft that created [GraphRAG](https://github.com/microsoft/graphrag), which is the foundation for this project.

# Motivation
[GraphRAG from Microsoft](https://github.com/microsoft/graphrag) makes it easy to get started with automated RAG using Graphs.

Sadly, the codebase is hard to read, difficult to addapt, and dataflow in the pipelines near impossible to follow.

The work done by the GraphRAG team shouldn't be in vain, but should allow for developers to create GraphRAG applications that are maintainable, extendable and intuitive.

And so GraphRAGZen was born; the logic given by GraphRAG in a functional, common-sense library.

## But that's just another implementation of GraphRAG, there are so many already
[XKCD says it well](https://xkcd.com/927/)

It's early days for this technology, and there's not yet a standard accepted at large. 

From experience I found that most libraries either provide a limited set of abilities with the bulk hiding behind convoluted code (i.e. GraphRAG) or are unintuitive and rely heavily on extended documentation. 

This should be easily mittigated by writing a functional library. That is, the python function is king, and the functions are organized intuitively. 
- A good python function has a clear purpose, is not too long and easy to read.
- Required inputs and produced outputs are easily understood.
- Flow of data can be easily traced.
- No wacky behind the scenes magic.
  
When that is established we have a toolbox that can be used as seen fit by developers. It is modular, extendable, maintainable and intuitive.

# Installation
TODO

# Getting started
TODO

# Kedro
TODO

# TODO:
## necasary for v0.1.0
✔ Everything docstringed!

✔ Coding style tests (flake8, black, isort and mypy)

✔ Make functions more 'functional' and move typing out of the big typing folder
- Unit tests
- Github tasks (automatic tests, push to pypi pipeline)
- Finish examples (and a function that can spawn the examples in the working dir)
- make automatic documentation that updates when new version is pushed to pypi (portray?)

## Needed for full version
- community detection
- graph embeddings (node2vec, text embeddings)
    - Find semantically similar nodes and combine them
- graph querying 
- test with GPU

## quality improvements
- OpenAI communication 
- Async
- graph visualization 
- structured output, e.g. force Json (https://til.simonwillison.net/llms/llama-cpp-python-grammars)
- different chunking strategies
- extract claims coupled to graph entities 
