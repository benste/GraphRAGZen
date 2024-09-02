import json
from copy import deepcopy
from typing import List, Optional, Tuple, Union

import networkx as nx
import pandas as pd
from graphragzen.llm.base_llm import LLM
from pydantic._internal._model_construction import ModelMetaclass
from tqdm import tqdm

from .llm_output_structures import ExtractedEntities
from .typing import EntityExtractionPromptConfig
from .utils import loop_extraction


def extract_raw_entities(
    dataframe: pd.DataFrame,
    llm: LLM,
    prompt_config: Optional[EntityExtractionPromptConfig] = EntityExtractionPromptConfig(),
    max_gleans: int = 5,
    column_to_extract: str = "chunk",
    results_column: str = "raw_entities",
    output_structure: ModelMetaclass = ExtractedEntities,  # type: ignore
) -> tuple:
    """Let the LLM extract entities in the form of strings, output still needs to be
    parsed to extract structured data.

    Args:
        dataframe (pd.DataFrame):
        llm (LLM):
        prompt_config (EntityExtractionPromptConfig, optional): See
            graphragzen.entity_extraction.EntityExtractionPromptConfig. Defaults to
            EntityExtractionPromptConfig().
        max_gleans (int, optional): How often the LLM can be asked if all entities have been
            extracted from a single text. Defaults to 5.
        column_to_extract (str, optional): Column in a DataFrame that contains the texts to extract
            entities from. Defaults to 'chunk'.
        results_column (str, optional): Column to write the output of the LLM to.
            Defaults to 'raw_entities'.
        output_structure (ModelMetaclass, optional): Output structure to force, e.g. grammars
            from llama.cpp. This SHOULD NOT be an instance of the pydantic model, just the
            reference.
            Correct = BaseLlamCpp("some text", MyPydanticModel)
            Wrong = BaseLlamCpp("some text", MyPydanticModel())
            Defaults to graphragzen.entity_extraction.ExtractedEntities

    Returns:
        pd.DataFrame: Input document with new column containing the raw entities extracted
    """
    prompt_config = prompt_config or EntityExtractionPromptConfig()

    raw_entities_df = deepcopy(dataframe)

    # Extract raw entities from each document
    raw_entities_df.reset_index(inplace=True, drop=True)
    raw_extracted_entities = []
    for document in tqdm(
        raw_entities_df[column_to_extract], total=len(raw_entities_df), desc="extracting entities"
    ):
        # Extract entities through LLM
        raw_extracted_entities.append(
            loop_extraction(
                document,
                prompt_config.prompts,
                prompt_config.formatting,
                llm,
                max_gleans,
                output_structure,
            ),
        )

    # Map LLM output to correct df column
    raw_entities_df[results_column] = raw_extracted_entities

    return raw_entities_df


def raw_entities_to_graph(
    input: Union[pd.DataFrame, str],
    raw_entities_column: str = "raw_entities",
    reference_column: str = "chunk_id",
    feature_delimiter: str = "\n",
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
        feature_delimiter (str, optional): When the same node or edge is found multiple times,
            features added to the entity are concatenated using this delimiter. Defaults to '\\n'.

    Returns:
        nx.Graph: unipartite graph
    """

    # Make sure we handle a dataframe with many raw entity strings or just a single string
    if isinstance(input, str):
        dataframe = pd.DataFrame(
            {
                raw_entities_column: [input],
                reference_column: [None],
            }
        )
    else:
        dataframe = deepcopy(input)

    graph = nx.Graph()
    for raw_extraction_strings, source_id in zip(
        *(dataframe[raw_entities_column], dataframe[reference_column])
    ):
        source_id = str(source_id)

        # This should return a list of dictionaries, on dict for each entity in the string
        structured_data = raw_entities_to_structure(raw_extraction_strings)

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
                    node["description"] += feature_delimiter + description
                    node["type"] = node["type"] if category != "" else node["type"]
                    node["source_id"] += feature_delimiter + source_id

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
                    edge["description"] += feature_delimiter + description
                    edge["source_id"] += feature_delimiter + source_id
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
) -> List[dict]:
    """When an LLM extracts entities using `extract_raw_entities` it returns a string.
    The LLM attempts to make this a valid json string, but that cannot be guarenteed. Thus parsing
    of some extracted entities might fail.

    Args:
        raw_string (Union[str, List[str]]): As returned by
            `graphragzen.entity_extraction.extract_raw_entities`

    Returns:
        List[dict]: Each parsed entity (node or edge)
    """

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
            Warning("Could not parse an extracted entity, not a valid JSON")

    return structured_data
