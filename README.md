# GraphRAGZen
**GraphRAGZen** is a functional, common sense, library for developing GraphRAG applications.
[Documentation found here](https://benste.github.io/GraphRAGZen/)

A big **thank you** to the team at microsoft that created [GraphRAG](https://github.com/microsoft/graphrag), which is the foundation for this project.

# Motivation
[GraphRAG from Microsoft](https://github.com/microsoft/graphrag) makes it easy to get started with 
automated RAG using Graphs.

Sadly, the codebase is hard to read, difficult to addapt, and dataflow
in the pipelines near impossible to follow.

The work done by the GraphRAG team shouldn’t be in vain, but should
allow for developers to create GraphRAG applications that are
maintainable, extendable and intuitive.

And so GraphRAGZen was born; the logic given by GraphRAG in a
functional, common sense library..

## But that's just another implementation of GraphRAG, there are so many already
[XKCD says it well](https://xkcd.com/927/)

It's early days for GraphRAG technology, and there's no standard implementation method accepted at 
large. This shows that there's still a lack of high quality, easy to implement, solutions in the 
OSS space.

From experience I found that most libraries available either provide a limited set
of abilities with the bulk hiding behind convoluted code (i.e. GraphRAG)
or are unintuitive and rely heavily on extended documentation.

This should be easily mittigated by writing a functional library. That
is, the python function is king, and the functions are named and located intuitively.

When that is established we have a toolbox that can be used as seen fit
by developers. It is modular, extendable, maintainable and intuitive.

[Documentation found here](https://benste.github.io/GraphRAGZen/)
