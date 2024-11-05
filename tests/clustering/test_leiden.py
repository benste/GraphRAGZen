import unittest
from unittest.mock import patch

import networkx as nx

from graphragzen.clustering.leiden import leiden, _leiden


class TestLeiden(unittest.TestCase):

    def test_leiden_basic(self):
        """Tests basic functionality with a simple graph."""
        graph = nx.Graph()
        graph.add_edges_from([("A", "B"), ("B", "C"), ("C", "A"), ("D", "E")])

        clustered_graph, cluster_map = leiden(graph)

        # Assertions
        self.assertEqual(set(clustered_graph.nodes), set(["A", "B", "C", "D", "E"]))
        self.assertEqual(len(clustered_graph.edges), len(graph.edges))
        for cluster in cluster_map.node_name:
            self.assertTrue(all(node in clustered_graph.nodes for node in cluster))

    def test_leiden_with_min_size(self):
        """Tests Leiden with minimum community size restriction."""
        graph = nx.Graph()
        graph.add_edges_from([("A", "B"), ("B", "C"), ("D", "E")])

        clustered_graph, cluster_map = leiden(graph, min_comm_size=2)

        # Assertions
        self.assertEqual(set(clustered_graph.nodes), set(["A", "B", "C", "D", "E"]))
        self.assertEqual(set(clustered_graph.edges), set([("A", "B"), ("B", "C"), ("D", "E")]))
        for cluster in cluster_map.node_name:
            self.assertTrue(all(node in clustered_graph.nodes for node in cluster))

    def test_leiden_with_max_size(self):
        """Tests Leiden with maximum community size restriction."""
        graph = nx.Graph()
        graph.add_edges_from(
            [
                ("A", "B"),
                ("B", "C"),
                ("C", "A"),
                ("D", "E"),
                ("D", "F"),
                ("E", "F"),
            ]
        )

        clustered_graph, cluster_map = leiden(graph, max_comm_size=3)

        # Assertions
        self.assertEqual(set(clustered_graph.nodes), set(["A", "B", "C", "D", "E", "F"]))
        self.assertEqual(len(clustered_graph.edges), len(graph.edges))
        for cluster in cluster_map.node_name:
            self.assertTrue(all(node in clustered_graph.nodes for node in cluster))
        
class TestInternalLeiden(unittest.TestCase):
    
    def test_internal_leiden_subgraph(self):
        """Tests internal Leiden with subgraph creation and recursive calls."""
        graph = nx.Graph()
        graph.add_edges_from([("A", "B"), ("B", "C"), ("C", "D")])

        clusters = _leiden(graph, levels=2)

        # Assert that subgraphs are created and recursive calls are made
        self.assertIn("A", clusters)
        self.assertIn("B", clusters)
        self.assertIn("C", clusters)
        self.assertIn("D", clusters)
        # ...