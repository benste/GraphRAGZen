import numbers
import re
from typing import Any

import networkx as nx
import pandas as pd
from graphragzen.llm.base_llm import LLM
from graphragzen.preprocessing import clean_str

from .typing import (
    EntityExtractionConfig,
    EntityExtractionPromptConfig,
    EntityExtractionPromptFormatting,
    RawEntitiesToGraphConfig,
)
from .utils import loop_extraction


def extract_raw_entities(
    dataframe: pd.DataFrame,
    llm: LLM,
    prompt_config: EntityExtractionPromptConfig,
    **kwargs: Any,
) -> tuple:
    """Let the LLM extract entities that is however just strings, output still needs to be
    parsed to extract structured data.

    Args:
        dataframe (pd.DataFrame)
        llm (LLM)
        prompt_config (EntityExtractionPromptConfig): See
            graphragzen.entity_extraction.EntityExtractionPromptConfig
        `max_gleans (int, optional): How often the LLM should be asked if all entities have been
            extracted from a single text. Defaults to 5.
        column_to_extract (str, optional): Column in a DataFrame that contains the texts to extract
            entities from. Defaults to 'chunk'.
        results_column (str, optional): Column to write the output of the LLM to.
            Defaults to 'raw_entities'.

    Returns:
        pd.DataFrame: Input document with new column containing the raw entities extracted
    """
    config = EntityExtractionConfig(**kwargs)  # type: ignore

    dataframe.reset_index(inplace=True, drop=True)

    llm_raw_output = loop_extraction(
        dataframe[config.column_to_extract],
        prompt_config.prompts,
        prompt_config.formatting,
        llm,
        config.max_gleans,
    )

    # Map LLM output to correct df column for intermediate saving
    dataframe[config.results_column] = [llm_raw_output[id] for id in dataframe.chunk_id.tolist()]

    return dataframe


def raw_entities_to_graph(
    dataframe: pd.DataFrame,
    prompt_formatting: EntityExtractionPromptFormatting,
    **kwargs: Any,
) -> nx.Graph:
    """Parse the result from raw entity extraction to create an undirected unipartite graph

    Args:
        dataframe (pd.DataFrame): Should contain a column with raw extracted entities
            prompt_formatting (EntityExtractionPromptFormatting): formatting used for raw entity
            extraction. See graphragzen.entity_extraction.EntityExtractionPromptFormatting.
        raw_entities_column (str, optional): Column in a DataFrame that contains the output of
            entity extraction. Defaults to 'raw_entities'.
        reference_column (str, optional): Value from this column in the DataFrame will be added to
            the edged and nodes. This allows to reference to the source where entities were
            extracted from when quiring the graph. Defaults to 'chunk_id'.
        feature_delimiter (str, optional): When the same node or edge is found multiple times,
            features are concatenated using this demiliter. Defaults to '\\n'.

    Returns:
        nx.Graph: unipartite graph
    """
    config = RawEntitiesToGraphConfig(**kwargs)  # type: ignore

    graph = nx.Graph()
    for extracted_data, source_id in zip(
        *(dataframe[config.raw_entities_column], dataframe[config.reference_column])
    ):

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
                    node["type"] = node["type"] if entity_type != "" else node["type"]
                else:
                    graph.add_node(
                        entity_name,
                        type=entity_type,
                        description=entity_description,
                        source_id=str(source_id),
                    )

            # Check if attribute is an edge
            if record_attributes[0] == '"relationship"' and len(record_attributes) >= 5:
                # Some cleaning
                source = clean_str(record_attributes[1].upper())
                target = clean_str(record_attributes[2].upper())
                edge_description = clean_str(record_attributes[3])
                edge_source_id = clean_str(str(source_id))
                # Try to get the weight
                weight = (
                    float(str(record_attributes[-1]))
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
                        edge_description = (
                            edge_data["description"] + config.feature_delimiter + edge_description
                        )
                        edge_source_id = (
                            edge_data["source_id"] + config.feature_delimiter + edge_source_id
                        )

                graph.add_edge(
                    source,
                    target,
                    weight=weight,
                    description=edge_description,
                    source_id=edge_source_id,
                )

    return graph
