import unittest
from unittest.mock import MagicMock, patch
import networkx as nx

from graphragzen.merge.merge_features import merge_graph_features, _count_merge, _mean_merge, _LLM_merge

from tests.mock_llm import MockLLM


class TestMergeGraphFeatures(unittest.TestCase):
    def test_merge_graph_features_llm(self):
        # Create sample graph
        graph = nx.Graph()
        graph.add_node("A", feature="desc1\ndesc2")

        # Mock LLM
        mock_llm = MockLLM()

        # Call function
        result_graph = merge_graph_features(graph, llm=mock_llm, feature="feature")

        # Assertions
        self.assertEqual(result_graph.nodes["A"]["feature"], "This is unstructured chat output")

    def test_merge_graph_features_count(self):
        # Create sample graph
        graph = nx.Graph()
        graph.add_node("A", feature="desc1\ndesc2\ndesc1")

        # Call function
        result_graph = merge_graph_features(graph, feature="feature", how="count")

        # Assertions
        self.assertEqual(result_graph.nodes["A"]["feature"], "desc1")

    def test_merge_graph_features_mean(self):
        # Create sample graph
        graph = nx.Graph()
        graph.add_node("A", feature="1\n2\n3")

        # Call function
        result_graph = merge_graph_features(graph, feature="feature", how="mean")

        # Assertions
        self.assertEqual(result_graph.nodes["A"]["feature"], 2.0)

    def test_merge_graph_features_empty_feature(self):
        # Create sample graph
        graph = nx.Graph()
        graph.add_node("A", feature="")
        
        # Mock LLM
        mock_llm = MockLLM()

        # Call function
        result_graph = merge_graph_features(graph, llm=mock_llm, feature="feature")

        # Assertions
        self.assertEqual(result_graph.nodes["A"].get("feature"), "")

    def test_merge_graph_features_no_feature(self):
        # Create sample graph
        graph = nx.Graph()
        graph.add_node("A")
        
        # Mock LLM
        mock_llm = MockLLM()
        
        # Call function
        result_graph = merge_graph_features(graph, llm=mock_llm, feature="feature")

        # Assertions
        self.assertEqual(result_graph.nodes["A"].get("feature"), None)

class TestMergeItemFeature(unittest.TestCase):
    def test_merge_item_feature_count(self):
        feature_list = ["desc1", "desc2", "desc1"]
        result = _count_merge(feature_list)
        self.assertEqual(result, "desc1")

    def test_merge_item_feature_mean(self):
        feature_list = ["1", "2", "3"]
        result = _mean_merge(feature_list)
        self.assertEqual(result, 2.0)

    def test_merge_item_feature_llm(self):
        mock_llm = MockLLM()
        
        feature_list = ["desc1", "desc2"]
        result = _LLM_merge(
            "A", feature_list, llm=mock_llm
        )
        print(result)