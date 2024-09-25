import asyncio
import json
from copy import deepcopy

import networkx as nx
import pandas as pd
from graphragzen.async_tools import async_loop
from graphragzen.llm.base_llm import LLM
from graphragzen.prompts.default_prompts import cluster_description_prompts
from pydantic._internal._model_construction import ModelMetaclass
from tqdm import tqdm

from .llm_output_structures import ClusterDescription


def describe_clusters(
    llm: LLM,
    graph: nx.Graph,
    cluster_entity_map: pd.DataFrame,
    prompt: str = cluster_description_prompts.CLUSTER_DESCRIPTION_PROMPT,
    output_structure: ModelMetaclass = ClusterDescription,  # type: ignore
    async_llm_calls: bool = False,
) -> pd.DataFrame:
    """Describe each cluster in the graph using the node descriptions.

    Args:
        llm (LLM):
        graph (nx.Graph):
        cluster_entity_map (pd.DataFrame): Containing the columns 'cluster' (string identifier of
            each cluster) and 'node_name' (lists of node names that belong to a cluster).
        prompt (str, optional): The prompt to use for the LLM to describe a cluster. Defaults to
            `graphragzen.prompts.default_prompts.cluster_description_prompts.CLUSTER_DESCRIPTION_PROMPT`
        output_structure (ModelMetaclass, optional): Output structure to force, e.g. grammars
            from llama.cpp. This SHOULD NOT be an instance of the pydantic model, just the
            reference.
            Correct = BaseLlamaCpp("some text", MyPydanticModel)
            Wrong = BaseLlamaCpp("some text", MyPydanticModel())
            Defaults to graphragzen.entity_extraction.ClusterDescription
        async_llm_calls: If True will call the LLM asynchronously. Only applies to communication
            with an LLM using `OpenAICompatibleClient`, in-memory LLM's loaded using
            llama-cpp-python will always be called synchronously. Defaults to False.

    Returns:
        pd.DataFrame:
    """  # noqa: E501

    cluster_entity_map_with_descriptions = deepcopy(cluster_entity_map)
    cluster_entity_map_with_descriptions["description"] = None

    # First gather the chats so they can be run against the LLM synchronously or asynchronously
    chats = []
    for _, cluster in tqdm(
        cluster_entity_map.iterrows(), desc="describing clusters", total=len(cluster_entity_map)
    ):
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

        # format prompt and send to LLM
        formatted_prompt = prompt.format(input_text=input_text)
        chats.append(llm.format_chat([("user", formatted_prompt)]))

    # call the LLM synchronously or asynchronously
    if async_llm_calls:
        loop = asyncio.get_event_loop()
        raw_descriptions = loop.run_until_complete(
            async_loop(
                llm.a_run_chat,
                chats,
                "describing clusters asynchronously",
                output_structure=output_structure,
            )
        )
    else:
        raw_descriptions = [llm.run_chat(chat, output_structure=output_structure) for chat in chats]

    # Parse the raw descriptions and write to the cluster map
    for index, raw_description in zip(cluster_entity_map.index, raw_descriptions):
        try:
            # Try json parsing first
            structured = json.loads(raw_description)

            # Verify that it adheres to the output structure
            if output_structure:
                structured = output_structure(**structured).dict()

        except Exception:
            Warning(
                f"""Could not parse a cluster description for cluster {index}
            The LLM either produced an invalid JSON, or the JSON did not adhere to
            the output structure, writing raw llm output for this cluster.\n
            Note: during querying this cluster cannot be used to generate a context (the individual
            nodes in the cluster still can).
            Clusters without structured descriptions (i.e. str in stead of dict) should be
            identified and fixed before using them with graph querying."""
            )
            structured = raw_description

        cluster_entity_map_with_descriptions.at[index, "description"] = structured
        # TODO: Findings return the ID that support the findings, couple this back to the
        # correct entity

    return cluster_entity_map_with_descriptions
