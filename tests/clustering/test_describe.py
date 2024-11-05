import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import networkx as nx

from graphragzen.clustering.describe import describe_clusters
from tests.mock_llm import MockLLM

class TestDescribeClusters(unittest.TestCase):
    def test_describe_clusters_sync(self):
        # Mock LLM methods
        mock_llm = MockLLM()

        # Create sample data
        graph = nx.Graph()
        graph.add_nodes_from(["A", "B", "C"])
        cluster_entity_map = pd.DataFrame({"cluster": ["test_cluster"], "node_name": [["A", "B", "C"]]})

        # Call the function
        result = describe_clusters(mock_llm, graph, cluster_entity_map)

        # Assertions
        self.assertEqual(result.shape[0], 1)
        self.assertEqual(result["description"].iloc[0]['summary'], "This is a structured chat output")

    def test_describe_clusters_async(self):
        # Mock LLM methods
        mock_llm = MockLLM()
        
        # Create sample data
        graph = nx.Graph()
        graph.add_nodes_from(["A", "B", "C"])
        cluster_entity_map = pd.DataFrame({"cluster": ["test_cluster"], "node_name": [["A", "B", "C"]]})

        # Call the function with async flag
        result = describe_clusters(mock_llm, graph, cluster_entity_map, async_llm_calls=True)

        # Assertions
        self.assertEqual(result.shape[0], 1)
        self.assertEqual(result["description"].iloc[0]['summary'], "This is a structured chat output")
        
    def test_describe_clusters_invalid_json(self):
        # Mock LLM methods
        mock_llm = MockLLM()
        
        def run_chat(*args, **kwargs):
            return "Invalid JSON"
        mock_llm.run_chat = run_chat
        
        # Create sample data
        graph = nx.Graph()
        graph.add_nodes_from(["A", "B", "C"])
        cluster_entity_map = pd.DataFrame({"cluster": ["test_cluster"], "node_name": [["A", "B", "C"]]})

        # Call the function
        result = describe_clusters(mock_llm, graph, cluster_entity_map)

        # Assertions
        self.assertEqual(result.shape[0], 1)
        self.assertEqual(result["description"].iloc[0], "Invalid JSON")

    def test_describe_clusters_invalid_structure(self):
        # Mock LLM methods
        mock_llm = MockLLM()
        
        def run_chat(*args, **kwargs):
            return '{"invalid_key": "value"}'
        mock_llm.run_chat = run_chat

        # Create sample data
        graph = nx.Graph()
        graph.add_nodes_from(["A", "B", "C"])
        cluster_entity_map = pd.DataFrame({"cluster": ["test_cluster"], "node_name": [["A", "B", "C"]]})

        # Call the function
        result = describe_clusters(mock_llm, graph, cluster_entity_map)

        # Assertions
        self.assertEqual(result.shape[0], 1)
        self.assertEqual(result["description"].iloc[0], '{"invalid_key": "value"}')

    def test_describe_clusters_empty_cluster_entity_map(self):
        # Mock LLM methods
        mock_llm = MockLLM()
        
        def format_chat(*args, **kwargs):
            return []
        mock_llm.format_chat = format_chat
        
        def run_chat(*args, **kwargs):
            return []
        mock_llm.run_chat = run_chat
        
        # Create sample data
        graph = nx.Graph()
        graph.add_nodes_from(["A", "B", "C"])
        cluster_entity_map = pd.DataFrame({"cluster": [], "node_name": []})

        # Call the function
        result = describe_clusters(mock_llm, graph, cluster_entity_map)

        # Assertions
        self.assertEqual(result.shape[0], 0)

    def test_describe_clusters_empty_graph(self):
        # Mock LLM methods
        mock_llm = MockLLM()
        
        def format_chat(*args, **kwargs):
            return []
        mock_llm.format_chat = format_chat
        
        def run_chat(*args, **kwargs):
            return []
        mock_llm.run_chat = run_chat

        # Create sample data
        graph = nx.Graph()
        cluster_entity_map = pd.DataFrame({"cluster": ["test_cluster"], "node_name": [[]]})

        # Call the function
        result = describe_clusters(mock_llm, graph, cluster_entity_map)

        # Assertions
        self.assertEqual(result.shape[0], 1)
        self.assertEqual(result["description"].iloc[0], None)
