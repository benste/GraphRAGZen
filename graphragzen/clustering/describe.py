import networkx as nx
import pandas as pd
from graphragzen.llm.base_llm import LLM

from .typing import DescribeClustersConfig


def describe_clusters(
    llm: LLM, graph: nx.Graph, cluster_entity_map: pd.DataFrame, **kwargs
) -> pd.DataFrame:
    """_summary_

    Args:
        llm (LLM):
        graph (nx.Graph):
        cluster_entity_map (pd.DataFrame):

    Returns:
        pd.DataFrame: _description_
    """
    config = DescribeClustersConfig(**kwargs)
    cluster_entity_map["raw_descriptions"] = None

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

        prompt = config.prompt.format(input_text=input_text)
        chat = llm.format_chat([("user", prompt)])
        llm_output = llm.run_chat(chat, output_structure=config.output_structure)

        return llm_output
        # TODO: Findings return the ID that support the findings, couple this back to the
        # correct entity
