GraphRAGZen ideology
=========================

Functions: first class citizens
----------------------------------

The python function is king, and the functions are named and located intuitively. 

- A well written python function has a clear purpose, is not too long, and easy to read.
- Required inputs and produced outputs are easily understood.
- Flow of data can be easily traced.
- No wacky behind the scenes magic.

Modularity
------------
In order to be modular functions should not presume a specific use-case. 

e.g. there is no **load_documents** function that presumes documents are in text format, rather 
there is a **load_text_documents** function.

Intuition
------------
To be intuitive **GraphRAGZen** is organized according to the steps one takes to implement Graph RAG

- load documents
    - load text documents
    - load PDF files (to be implemented)
- preprocessing
    - clean strings
    - chunk documents
- make graph
    - optionally adjust the LLM prompts to the documents
    - extract entities (nodes and edges) using LLM
    - parse extracted entities into a graph
- post-process graph
    - Merge features of entities found multiple times in the documents
    - Cluster graph
    - Describe each cluster
- Embeddings
    - Text Embed the node and edge descriptions
    - Text Embed the cluster describtions
- Query graph
    - to be implemented
