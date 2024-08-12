import re
import numbers

import pandas as pd
import networkx as nx

from graphragzen.typing import EntityExtractionConfig, EntityExtractionPromptConfig, EntityExtractionPromptFormatting, RawEntitiesToGraphConfig
from graphragzen.entity_extraction.utils import loop_extraction
from graphragzen.preprocessing.utils import clean_str
from graphragzen.llm.base_llm import LLM


def raw_entity_extraction(dataframe: pd.DataFrame, llm: LLM, prompt_config: EntityExtractionPromptConfig, config: EntityExtractionConfig) -> tuple:
    """Let the LLM extract entities that is however just strings, output still needs to be parsed to extract structured data.

    Args:
        dataframe (pd.DataFrame)
        llm (LLM)
        config (EntityExtractionConfig)
        prompt_config (EntityExtractionPromptConfig)

    Returns:
        pd.DataFrame: Input document with new column containing the raw entities extracted
    """
    dataframe.reset_index(inplace=True, drop=True)
    
    llm_raw_output = loop_extraction(dataframe[config.column_to_extract], prompt_config.prompts, prompt_config.formatting, llm, config.max_gleans)
    
    # Map LLM output to correct df column for intermediate saving
    dataframe[config.results_column] = [llm_raw_output[id] for id in dataframe.chunk_id.tolist()]
        
    return dataframe
   

def raw_entities_to_graph(
        dataframe: pd.DataFrame,
        prompt_formatting: EntityExtractionPromptFormatting,
        config: RawEntitiesToGraphConfig,
    ) -> nx.Graph:
    """Parse the result string to create an undirected unipartite graph.

    Args:
        dataframe (pd.DataFrame): Should contain a column with raw extracted entities
        prompt_formatting (EntityExtractionPromptFormatting): formatting used for raw entity extraction.
            Should at least contain `prompt_formatting.tuple_delimiter` and `prompt_formatting.record_delimiter`
        config (RawEntitiesToGraphConfig)

    Returns:
        str:  unipartite graph in graphML format
    """
    graph = nx.Graph()
    for extracted_data, source_id in zip(*(dataframe[config.raw_entities_column], dataframe[config.reference_column])):
        
        records = extracted_data.split(prompt_formatting.record_delimiter)
        for record in records:
            # Some light cleaning
            record = re.sub(r"^\(|\)$", "", record.strip())
            record_attributes = record.split(prompt_formatting.tuple_delimiter)
            
            # Check if attribute is a node
            if record_attributes[0] == '"entity"' and len(record_attributes) >= 4:
                # Some cleaning
                entity_name = clean_str(record_attributes[1].upper())
                entity_type = clean_str(record_attributes[2].upper())
                entity_description = clean_str(record_attributes[3])

                if entity_name in graph.nodes():
                    # Merge attributes
                    node = graph.nodes[entity_name]
                    node["description"] += config.feature_delimiter + entity_description
                    node["source_id"] += config.feature_delimiter + str(source_id)
                    node["entity_type"] = (
                        entity_type if entity_type != "" else node["entity_type"]
                    )
                else:
                    graph.add_node(
                        entity_name,
                        type=entity_type,
                        description=entity_description,
                        source_id=str(source_id),
                    )

            # Check if attribute is an edge
            if (
                record_attributes[0] == '"relationship"'
                and len(record_attributes) >= 5
            ):
                # Some cleaning
                source = clean_str(record_attributes[1].upper())
                target = clean_str(record_attributes[2].upper())
                edge_description = clean_str(record_attributes[3])
                edge_source_id = clean_str(str(source_id))
                # Try to get the weight
                weight = (
                    float(record_attributes[-1])
                    if isinstance(record_attributes[-1], numbers.Number)
                    else 1.0
                )
                if source not in graph.nodes():
                    graph.add_node(
                        source,
                        type="",
                        description="",
                        source_id=edge_source_id,
                    )
                if target not in graph.nodes():
                    graph.add_node(
                        target,
                        type="",
                        description="",
                        source_id=edge_source_id,
                    )
                if graph.has_edge(source, target):
                    # Merge edge attributes
                    edge_data = graph.get_edge_data(source, target)
                    if edge_data is not None:
                        weight += edge_data["weight"]
                        edge_description = edge_data['description'] + config.feature_delimiter + edge_description
                        edge_source_id = edge_data['source_id'] + config.feature_delimiter + edge_source_id
                        
                graph.add_edge(
                    source,
                    target,
                    weight=weight,
                    description=edge_description,
                    source_id=edge_source_id,
                )
          
    return graph
