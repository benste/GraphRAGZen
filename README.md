# GraphRAGZen
the logic given by GraphRAG in a functional, non-nonsense library.

Thank you to the team at microsoft that created [GraphRAG](https://github.com/microsoft/graphrag), which is the foundation for this project.

# Motivation
[GraphRAG from Microsoft](https://github.com/microsoft/graphrag) makes it easy to get started with automated RAG using Graphs.

Sadly, the codebase is hard to read, difficult to addapt, and dataflow in the pipelines near impossible to follow.

The work done by the GraphRAG team shouldn't be in vain, but should allow for developers to create GraphRAG applications that are maintainable, extendable and intuitive.

And so GraphRAGZen was born; the logic given by GraphRAG in a functional, non-nonsense library.

## But that's just another implementation of GraphRAG, there are so many already
[XKCD says it well](https://xkcd.com/927/)

It's early days for this technology, and there's not yet a standard accepted at large. 

From experience I found that most libraries either provide a limited set of abilities with the bulk hiding behind convoluted code (i.e. GraphRAG) or are unintuitive and rely heavily on extended documentation. 

This should be easily mittigated by writing a functional library. That is, the python function is king, and the functions are organized intuitively. 
- A good python function has a clear purpose, is not too long and easy to read.
- Required inputs and outputs are easily understood.
- Flow of data can be easily traced.
- No wacky behind the scenes magic.
  
When that is established we have a toolbox that can be used as seen fit by developers. It is maintainable, extendable and intuitive.

# Installation
TODO

# Getting started
TODO

# Kedro
TODO

# TODO:
- Everything docstringed!
- Unit tests
- Coding style tests (flake8, black, isort and mypy)
- Test for duplicate typing
- Github tasks (automatic tests, push to pypi pipeline)
- Finish examples (and a function that can spawn the examples in the working dir)
