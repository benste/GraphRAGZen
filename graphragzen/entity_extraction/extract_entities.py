import json
import numbers
import re
from typing import Any, List, Optional, Tuple, Union

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
        dataframe (pd.DataFrame):
        llm (LLM):
        prompt_config (EntityExtractionPromptConfig, optional): See
            graphragzen.entity_extraction.EntityExtractionPromptConfig. Defaults to
            EntityExtractionPromptConfig().
        `max_gleans (int, optional): How often the LLM can be asked if all entities have been
            extracted from a single text. Defaults to 5.
        column_to_extract (str, optional): Column in a DataFrame that contains the texts to extract
            entities from. Defaults to 'chunk'.
        results_column (str, optional): Column to write the output of the LLM to.
            Defaults to 'raw_entities'.
        output_structure (BaseModel, optional): Output structure to force, using e.g. grammars from
            llama.cpp.
            Defaults to graphragzen.entity_extraction.llm_output_structures.ExtractedEntities

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
                config.output_structure,
            ),
        )

    # Map LLM output to correct df column
    dataframe[config.results_column] = raw_extracted_entities

    return dataframe


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
        prompt_formatting (EntityExtractionPromptFormatting, optional): formatting used for raw
            entity extraction. See `graphragzen.entity_extraction.EntityExtractionPromptFormatting`.
            Defaults to EntityExtractionPromptFormatting().
        feature_delimiter (str, optional): When the same node or edge is found multiple times,
            features added to the entity are concatenated using this delimiter. Defaults to '\\n'.

    Returns:
        nx.Graph: unipartite graph
    """
    config = RawEntitiesToGraphConfig(**kwargs)  # type: ignore
    prompt_formatting = prompt_formatting or EntityExtractionPromptFormatting()

    # Make sure we handle a dataframe with many raw entity strings or just a single string
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
    for raw_extraction_strings, source_id in zip(
        *(dataframe[config.raw_entities_column], dataframe[config.reference_column])
    ):
        source_id = str(source_id)

        # This should return a list of dictionaries, on dict for each entity in the string
        structured_data = raw_entities_to_structure(raw_extraction_strings, prompt_formatting)

        for entity in structured_data:
            # Get the entity properties
            type = entity.get("type", "")
            name = entity.get("name", None)
            description = entity.get("description", "")
            category = entity.get("category", "")
            source = entity.get("source", None)
            target = entity.get("target", None)
            weight = entity.get("weight", 1.0)

            # If we have a node
            if type == "node" and name:
                if name in graph.nodes():
                    # Merge attributes if node already in graph
                    node = graph.nodes[name]
                    node["description"] += config.feature_delimiter + description
                    node["type"] = node["type"] if category != "" else node["type"]
                    node["source_id"] += config.feature_delimiter + source_id

                else:
                    # Otherwise make a new node in graph
                    graph.add_node(
                        name,
                        type=category,
                        description=description,
                        source_id=source_id,
                    )

            # If we have an edge
            elif type == "edge" and source and target:
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

                if graph.has_edge(source, target):
                    # Merge attributes if edge already in graph
                    edge = graph.edges[(source, target)]
                    edge["weight"] = edge.get("weight", 0) + weight
                    edge["description"] += config.feature_delimiter + description
                    edge["source_id"] += config.feature_delimiter + source_id
                else:
                    # Otherwise add a new edge to the graph
                    graph.add_edge(
                        source,
                        target,
                        weight=weight,
                        description=description,
                        source_id=source_id,
                    )

    return graph


def raw_entities_to_structure(
    raw_strings: Union[str, List[str]],
    prompt_formatting: Optional[
        EntityExtractionPromptFormatting
    ] = EntityExtractionPromptFormatting(),
) -> List[dict]:
    """When an LLM extracts entities using `extract_raw_entities` it is returned in a string.
    This is either a valid json string or delimited values. This function first tries to load the
    strings as json, and if that fails tries to extract the values by splitting on the delimiter.

    Args:
        raw_string (Union[str, List[str]]): As returned by
            `graphragzen.entity_extraction.extract_raw_entities`
        prompt_formatting (EntityExtractionPromptFormatting, optional): formatting used for raw
            entity extraction. See `graphragzen.entity_extraction.EntityExtractionPromptFormatting`.
            Defaults to EntityExtractionPromptFormatting().

    Returns:
        List[dict]: Each parsed entity (node or edge)
    """

    prompt_formatting = prompt_formatting or EntityExtractionPromptFormatting()

    if isinstance(raw_strings, str):
        raw_strings = [raw_strings]

    structured_data = []
    for raw_string in raw_strings:
        try:
            # Try json parsing first
            structured = json.loads(raw_string)

            if isinstance(structured, list):
                structured_data += structured

            if "extracted_nodes" in structured and isinstance(structured["extracted_nodes"], list):
                extracted_nodes = structured["extracted_nodes"]
                extracted_nodes = [e | {"type": "node"} for e in extracted_nodes]
                structured_data += extracted_nodes

            if "extracted_edges" in structured and isinstance(structured["extracted_edges"], list):
                extracted_edges = structured["extracted_edges"]
                extracted_edges = [e | {"type": "edge"} for e in extracted_edges]
                structured_data += extracted_edges

        except Exception:
           Warning(f"Could not parse an extracted entity, not a valid JSON")

    return structured_data
