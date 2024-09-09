.. role:: raw-html(raw)
    :format: html

Clustering
----------

There are latent structures to Graphs that can be usefull when extracting information from them.
:raw-html:`<br />`
For instance, in a graph about vehicles, part of the graph might talk about cars and another about
bicycles. When we want information about cars we can safely presume to start looking at the car part.

When **GraphRAGZen** first creates a graph these clusters are not yet known.
:raw-html:`<br />`
We don't want to go through the graph manually and assign each node to a topic. We rather use 
unsupervised clustering for this, specifically the `leiden algorithm <https://arxiv.org/abs/1810.08473>`_

This algorithm only assigns a number to each node, indicating to which cluster it belongs. 
The semantic topic of each cluster still needs to be extracted. This can be done using
:func:`graphragzen.clustering.describe.describe_clusters`
