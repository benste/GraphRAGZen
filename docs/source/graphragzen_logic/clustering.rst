Clustering
----------

There are latent structures to Graphs that can be usefull when extracting information from it.
For instance, in a graph about vehicles, part of the graph might talk about cars and another about
bicycles. When we want information about cars we can safely presume to start looking at the car part.

When **GraphRAGZen** first creates a graph these topics are not known yet.
We want to go through the graph manually and assign each node to a topic. We will use unsupervised
clustering for this, specifically the leiden algorithm leidenalg_url_.

This only assigns a number to each node, indicating to which cluster it belongs. The topic still 
needs to be extracted.

.. _leidenalg_url: https://arxiv.org/abs/1810.08473