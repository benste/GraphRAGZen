Motivation
==========

`GraphRAG from Microsoft <https://github.com/microsoft/graphrag>`_
makes it easy to get started with automated RAG using Graphs.

Sadly, the codebase is hard to read, difficult to addapt, and dataflow
in the pipelines near impossible to follow.

The work done by the GraphRAG team shouldn’t be in vain, but should
allow for developers to create GraphRAG applications that are
maintainable, extendable and intuitive.

And so GraphRAGZen was born; the logic given by GraphRAG in a
functional, common sense library.

But that’s just another implementation of GraphRAG, there are so many already
-----------------------------------------------------------------------------

`XKCD says it well <https://xkcd.com/927/>`_

It’s early days for this technology, and there’s not yet a standard
accepted at large.

From experience I found that most libraries either provide a limited set
of abilities with the bulk hiding behind convoluted code (i.e. GraphRAG)
or are unintuitive and rely heavily on extended documentation.

This should be easily mittigated by writing a functional library. That
is, the python function is king, and the functions are named and located intuitively.

When that is established we have a toolbox that can be used as seen fit
by developers. It is modular, extendable, maintainable and intuitive.