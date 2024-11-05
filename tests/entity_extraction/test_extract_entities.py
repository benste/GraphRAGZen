import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import networkx as nx

from graphragzen.entity_extraction import (
    extract_raw_entities,
    raw_entities_to_graph,
    raw_entities_to_structure,
    llm_output_structures,
)

from tests.mock_llm import MockLLM

class TestExtractRawEntities(unittest.TestCase):
    @patch("graphragzen.async_tools.async_loop")
    def test_extract_raw_entities_async(self, mock_async_loop):
        # Mock LLM and other components
        mock_llm = MockLLM()

        # Create sample data
        input_data = pd.DataFrame({"chunk": ["This is a text"]})

        # Call the function
        result = extract_raw_entities(input_data, mock_llm, async_llm_calls=True)

        # Assertions
        self.assertEqual(result.shape, (1, 2))
        self.assertEqual(result["raw_entities"][0][0]['extracted_nodes'], "This is a structured chat output")
      
    def test_extract_raw_entities_sync(self):
        # Mock LLM and other components
        mock_llm = MockLLM()
        
        # Create sample data
        input_data = pd.DataFrame({"chunk": ["This is a text"]})

        # Call the function
        result = extract_raw_entities(input_data, mock_llm, async_llm_calls=False)

        # Assertions
        self.assertEqual(result.shape, (1, 2))
        self.assertEqual(result["raw_entities"][0][0]['extracted_nodes'], "This is a structured chat output")

    def test_extract_raw_entities_with_list(self):
        # Mock LLM and other components
        mock_llm = MockLLM()
        
        # Create sample data
        input_data = pd.DataFrame({"chunk": ["This is a text"]*10})

        # Call the function
        result = extract_raw_entities(input_data, mock_llm, async_llm_calls=False)

        # Assertions
        self.assertEqual(result.shape, (10, 2))
        for raw_entity in result.raw_entities:
            self.assertEqual(raw_entity[0]['extracted_nodes'], "This is a structured chat output")
            
    def test_extract_raw_entities_without_output_structure(self):
        # Mock LLM and other components
        mock_llm = MockLLM()
        
        # Create sample data
        input_data = pd.DataFrame({"chunk": ["This is a text"]})
        
        result = extract_raw_entities(input_data, mock_llm, output_structure=None, async_llm_calls=False)

        self.assertEqual(result.shape, (1, 2))
        self.assertEqual(result["raw_entities"][0][0], "This is unstructured chat output")
        
class TestRawEntitiesToGraph(unittest.TestCase):
    def test_raw_entities_to_graph(self):
        # Create sample data
        input_data = pd.DataFrame(
            {
                "raw_entities": [
                    '[{"type": "node", "name": "A"}, {"type": "edge", "source": "A", "target": "B"}]'
                ],
                "chunk_id": ["1"],
            }
        )

        # Call the function
        graph = raw_entities_to_graph(input_data)

        # Assertions
        self.assertEqual(len(graph.nodes), 2)
        self.assertEqual(len(graph.edges), 1)


class TestRawEntitiesToStructure(unittest.TestCase):
    def test_raw_entities_to_structure_valid_json(self):
        raw_string = '[{"type": "node", "name": "A"}, {"type": "edge", "source": "A", "target": "B"}]'
        structured_data = raw_entities_to_structure(raw_string)

        # Assertions
        self.assertEqual(len(structured_data), 2)
        self.assertEqual(structured_data[0]["type"], "node")
        self.assertEqual(structured_data[1]["type"], "edge")

    def test_raw_entities_to_structure_invalid_json(self):
        raw_string = "invalid json"
        structured_data = raw_entities_to_structure(raw_string)

        # Assertions
        self.assertEqual(structured_data, [])