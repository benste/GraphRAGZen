import numbers
import re
from random import sample
from typing import Any, Optional, Tuple, Union

import networkx as nx
import pandas as pd
from graphragzen.llm.base_llm import LLM
from graphragzen.preprocessing import clean_str
from tqdm import tqdm

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
    prompt_config: Optional[EntityExtractionPromptConfig] = EntityExtractionPromptConfig(),
    **kwargs: Union[dict, EntityExtractionConfig, Any],
) -> tuple:
    """Let the LLM extract entities in the form of strings, output still needs to be
    parsed to extract structured data.

    Args:
        dataframe (pd.DataFrame)
        llm (LLM)
        prompt_config (EntityExtractionPromptConfig, optional): See
            graphragzen.entity_extraction.EntityExtractionPromptConfig
        `max_gleans (int, optional): How often the LLM can be asked if all entities have been
            extracted from a single text. Defaults to 5.
        column_to_extract (str, optional): Column in a DataFrame that contains the texts to extract
            entities from. Defaults to 'chunk'.
        results_column (str, optional): Column to write the output of the LLM to.
            Defaults to 'raw_entities'.

    Returns:
        pd.DataFrame: Input document with new column containing the raw entities extracted
    """
    config = EntityExtractionConfig(**kwargs)  # type: ignore
    prompt_config = prompt_config or EntityExtractionPromptConfig()

    # Extract raw entities from each document
    dataframe.reset_index(inplace=True, drop=True)
    raw_extracted_entities = []
    for document in tqdm(
        dataframe[config.column_to_extract], total=len(dataframe), desc="extracting entities"
    ):
        # Extract entities through LLM
        raw_extracted_entities.append(
            loop_extraction(
                document,
                prompt_config.prompts,
                prompt_config.formatting,
                llm,
                config.max_gleans,
            ),
        )

    # Map LLM output to correct df column
    dataframe[config.results_column] = raw_extracted_entities

    return dataframe


def extract_more_edges(
    graph: nx.Graph,
    llm: LLM,
    prompt_config: Optional[EntityExtractionPromptConfig] = EntityExtractionPromptConfig(),
    **kwargs: Union[dict, EntityExtractionConfig, Any],
) -> str:
    """Extracts more edges from a graph. It simply takes the nodes from the graph and asks the LLM
    if there are edges between these nodes.

    Args:
        graph (nx.Graph)
        llm (LLM)
        prompt_config (Optional[EntityExtractionPromptConfig], optional):See
            graphragzen.entity_extraction.EntityExtractionPromptConfig
        extra_edges_iterations (int, optional): During extra edge extraction random nodes are
            selected to find relationships. If extra edges are extracted, how many runs are
            performed. Defaults to 100.
        extra_edges_max_nodes (int, optional): During extra edge extraction random nodes are
            selected to find relationships. How many nodes should be selected per sample?
            Defaults to 20.

    Returns:
        str: Raw LLM output
    """
    config = EntityExtractionConfig(**kwargs)  # type: ignore
    prompt_config = prompt_config or EntityExtractionPromptConfig()

    nodes = list(graph.nodes(data=True))

    raw_extracted_edges = ""
    for _ in tqdm(range(config.extra_edges_iterations), desc="finding more edges"):
        # Sample nodes
        sampled_nodes = sample(nodes, min([len(nodes), config.extra_edges_max_nodes]))

        # Create a string from the nodes
        tuple_delimiter = prompt_config.formatting.tuple_delimiter
        entities_string = prompt_config.formatting.record_delimiter.join(
            [_node_to_entity_string(node, tuple_delimiter) for node in sampled_nodes]
        )
        prompt_config.formatting.entities_string = entities_string

        # Create final prompt
        prompt = prompt_config.prompts.entity_relationship_prompt.format(
            **prompt_config.formatting.model_dump()
        )

        # Run prompt
        chat = llm.format_chat([("user", prompt)])
        llm_output = llm.run_chat(chat).removesuffix(prompt_config.formatting.completion_delimiter)
        raw_extracted_edges += prompt_config.formatting.record_delimiter + llm_output

    return raw_extracted_edges


def _node_to_entity_string(node: tuple, tuple_delimiter: str) -> str:
    node_name = node[0]
    node_type = node[1]["type"]
    node_description = node[1]["description"]

    return f"(entity{tuple_delimiter}{node_name}{tuple_delimiter}{node_type}{tuple_delimiter}{node_description})"  # noqa: E501


def raw_entities_to_graph(
    input: Union[pd.DataFrame, str],
    prompt_formatting: Optional[
        EntityExtractionPromptFormatting
    ] = EntityExtractionPromptFormatting(),
    **kwargs: Union[dict, RawEntitiesToGraphConfig, Any],
) -> Tuple[nx.Graph, pd.DataFrame]:
    """Parse the result from raw entity extraction to create an undirected unipartite graph

    Args:
        input (Union[pd.DataFrame, str]): If a raw extracted entities string is provided it will
            simply be parsed to a graph.
            When a dataframe is provided it should contain a column with raw extracted entities
            strings and a reference column whos value will be added to the nodes and edged.
        raw_entities_column (str, optional): Column in a DataFrame that contains the output of
            entity extraction. Defaults to 'raw_entities'.
        reference_column (str, optional): Value from this column in the DataFrame will be added to
            the edged and nodes. This allows to reference to the source where entities were
            extracted from when quiring the graph. Defaults to 'chunk_id'.
        prompt_formatting (EntityExtractionPromptFormatting): formatting used for raw entity
            extraction. See graphragzen.entity_extraction.EntityExtractionPromptFormatting.
        feature_delimiter (str, optional): When the same node or edge is found multiple times,
            features are concatenated using this demiliter. Defaults to '\\n'.

    Returns:
        nx.Graph: unipartite graph
    """
    config = RawEntitiesToGraphConfig(**kwargs)  # type: ignore
    prompt_formatting = prompt_formatting or EntityExtractionPromptFormatting()

    if isinstance(input, str):
        dataframe = pd.DataFrame(
            {
                config.raw_entities_column: [input],
                config.reference_column: [None],
            }
        )
    else:
        dataframe = input

    graph = nx.Graph()
    for extracted_data, source_id in zip(
        *(dataframe[config.raw_entities_column], dataframe[config.reference_column])
    ):

        source_id = str(source_id)

        records = extracted_data.split(prompt_formatting.record_delimiter)
        for record in records:
            # Some light cleaning and splitting of raw text
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
                    node["source_id"] += config.feature_delimiter + source_id
                    node["type"] = node["type"] if entity_type != "" else node["type"]
                else:
                    graph.add_node(
                        entity_name,
                        type=entity_type,
                        description=entity_description,
                        source_id=source_id,
                    )

            # Check if attribute is an edge
            if record_attributes[0] == '"relationship"' and len(record_attributes) >= 4:
                # Some cleaning
                source = clean_str(record_attributes[1].upper())
                target = clean_str(record_attributes[2].upper())
                edge_description = clean_str(record_attributes[3])

                # Try to get the weight
                if len(record_attributes) > 4:
                    weight = (
                        float(str(record_attributes[-1]))
                        if isinstance(record_attributes[-1], numbers.Number)
                        else 1.0
                    )

                # Add nodes for this edge if they do not exist yet
                if source not in graph.nodes():
                    graph.add_node(
                        source,
                        type="",
                        description="",
                        source_id=source_id,
                    )

                if target not in graph.nodes():
                    graph.add_node(
                        target,
                        type="",
                        description="",
                        source_id=source_id,
                    )

                # If edge already exist, concat or merge features
                if graph.has_edge(source, target):
                    edge = graph.edges[(source, target)]
                    edge["weight"] = edge.get("weight", 0) + weight
                    edge["description"] = (
                        edge.get("description", "") + config.feature_delimiter + edge_description
                    )
                    edge["source_id"] = (
                        edge.get("source_id", "") + config.feature_delimiter + source_id
                    )
                else:
                    graph.add_edge(
                        source,
                        target,
                        weight=weight,
                        description=edge_description,
                        source_id=source_id,
                    )

    return graph
