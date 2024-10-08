Motivation
==========

If you are not familiar with Graph RAG yet, I found `this to be a good write-up <https://www.analyticsvidhya.com/blog/2024/07/graph-rag/>`_

`GraphRAG from Microsoft <https://github.com/microsoft/graphrag>`_
makes it easy to get started with automated Graph RAG.

Sadly, the codebase is hard to read, difficult to adapt, and dataflow in the pipelines near impossible to follow.

The work done by the GraphRAG team shouldn't be in vain, but should allow for developers to create GraphRAG applications that are maintainable, extendable and intuitive.

And so **GraphRAGZen** was born; the logic given by GraphRAG in a functional, common sense library...



But that's just another Graph RAG library, right?
-----------------------------------------------------------------------------

`XKCD says it well <https://xkcd.com/927/>`_

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