from copy import deepcopy

import networkx as nx
import pandas as pd
from graphragzen.llm.base_llm import LLM
from graphragzen.prompts.default_prompts import cluster_description_prompts
from pydantic._internal._model_construction import ModelMetaclass

from .llm_output_structures import ClusterDescription


def describe_clusters(
    llm: LLM,
    graph: nx.Graph,
    cluster_entity_map: pd.DataFrame,
    prompt: str = cluster_description_prompts.CLUSTER_DESCRIPTION_PROMPT,
    output_structure: ModelMetaclass = ClusterDescription,  # type: ignore
) -> pd.DataFrame:
    """Describe each cluster in the graph using the node descriptions.

    Args:
        llm (LLM):
        graph (nx.Graph):
        cluster_entity_map (pd.DataFrame):
        prompt (str, optional): The prompt to use for the LLM to describe a cluster. Defaults to
            `graphragzen.prompts.default_prompts.cluster_description_prompts.CLUSTER_DESCRIPTION_PROMPT`
        output_structure (ModelMetaclass, optional): Output structure to force, e.g. grammars
                from llama.cpp. This SHOULD NOT be an instance of the pydantic model, just the
                reference.
                Correct = BaseLlamCpp("some text", MyPydanticModel)
                Wrong = BaseLlamCpp("some text", MyPydanticModel())
                Defaults to graphragzen.entity_extraction.ClusterDescription
    Returns:
        pd.DataFrame
    """  # noqa: E501

    cluster_entity_map["raw_descriptions"] = None

    cluster_entity_map_with_descriptions = deepcopy(cluster_entity_map)

    for index, cluster in cluster_entity_map.iterrows():
        cluster_graph = graph.subgraph(cluster.node_name)

        id = 0
        # First add the nodes to the prompt string
        input_text = "Entities\n\nid,entity,description\n"
        for node in cluster_graph.nodes(data=True):
            input_text += f"{id},{node[0]},{node[1].get('description', '')}\n"
            id += 1

        # Now add the edged to the prompt string
        input_text += "\nRelationships\n\nid,source,target,description\n"
        for edge in cluster_graph.edges(data=True):
            input_text += f"{id},{edge[0]},{edge[1]},{edge[2].get('description', '')}\n"
            id += 1

        prompt = prompt.format(input_text=input_text)
        chat = llm.format_chat([("user", prompt)])
        llm_output = llm.run_chat(chat, output_structure=output_structure)

        cluster_entity_map_with_descriptions.iloc[index]["raw_descriptions"] = llm_output
        # TODO: Findings return the ID that support the findings, couple this back to the
        # correct entity

    return cluster_entity_map_with_descriptions
