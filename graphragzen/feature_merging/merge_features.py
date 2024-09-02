from collections import Counter
from functools import partial
from typing import List, Literal, Optional, Union

import networkx as nx
from graphragzen.llm.base_llm import LLM
from numpy import mean
from tqdm import tqdm

from .typing import MergeFeaturesPromptConfig
from .utils import _num_tokens_from_string


def merge_graph_features(
    graph: nx.Graph,
    llm: Optional[LLM],
    feature: str,
    prompt: MergeFeaturesPromptConfig = MergeFeaturesPromptConfig(),
    how: Literal["LLM", "count", "mean"] = "LLM",
    feature_delimiter: str = "\n",
    max_input_tokens: int = 4000,
    max_output_tokens: int = 500,
) -> nx.Graph:
    """Summarize lists of descriptions for each node or edge

    Args:
        graph (nx.Graph): With edges and nodes expected to have the feature 'description'.
            The descriptions are expected to be delimited by Kwargs["feature_delimiter"]
        llm (LLM, optional): Only used if `how` is set to 'LLM'. Dedaults to None.
        feature (str): The feature attached to a graph entity (node or edge) to merge.
        prompt (MergeFeaturesPromptConfig, optional): Will be formatted with the feature
            to send to the LLM. Only used if `how` is set to 'LLM'.
            See `graphragzen.typing.MergeFeaturesPromptConfig`.
            Defaults to MergeFeaturesPromptConfig.
        how (Literal['LLM', 'count', 'mean'], optional): 'LLM' summarizes the features.
            'count' takes the feature that occurs most. 'mean' takes the mean of the feature.
            Defaults to 'LLM'.
        feature_delimiter (str, optional): During entity extraction the same node or edge can be
            found multiple times, and features were concatenated using this delimiter.
            We will make a list of descriptions by splitting on this delimiter. Defaults to '\\n'.
        max_input_tokens (int, optional): Only used when how=='LLM'. Maximum input tokens until a
            summary is made. Remaining descriptions will be appended to the summary until
            max_input_tokens is reached again or no descriptions are left. Defaults to 4000.
        max_output_tokens (int, optional): Only used when how=='LLM'. Maximum number of tokens a
            summary can have. Defaults to 500.

    Returns:
        nx.Graph
    """

    item_merger = partial(
        merge_item_feature,
        llm=llm,
        prompt=prompt,
        how=how,
        max_input_tokens=max_input_tokens,
        max_output_tokens=max_output_tokens,
    )

    for node in tqdm(graph.nodes(data=True), desc=f"Merging {feature} of nodes"):
        entity_name = node[0]
        # Split and sort the feature
        feature_list = sorted(set(node[1].get(feature, "").split(feature_delimiter)))
        # Merge
        if feature_list:
            graph.nodes[entity_name][feature] = item_merger(
                entity_name=entity_name, feature_list=feature_list
            )

    for edge in tqdm(graph.edges(data=True), desc=f"Merging {feature} of edges"):
        entity_name = edge[:2]
        # Split and sort the feature
        feature_list = sorted(set(edge[2].get(feature, "").split(feature_delimiter)))
        # Merge
        if feature_list:
            graph.edges[entity_name]["description"] = item_merger(
                entity_name=entity_name, feature_list=feature_list
            )

    return graph


def merge_item_feature(
    entity_name: str,
    feature_list: List[str],
    llm: Optional[LLM],
    prompt: Optional[MergeFeaturesPromptConfig] = MergeFeaturesPromptConfig(),
    how: Literal["LLM", "count", "mean"] = "LLM",
    max_input_tokens: int = 4000,
    max_output_tokens: int = 500,
) -> Union[str, float]:
    """For a single node or edge, merge one feature that is however a list.

    Args:
        entity_name (str): Name of the node or edge
        feature_list (List[str]): List of values assigned to the one feature
        llm (LLM, optional): Only used if `how` is set to 'LLM'. Dedaults to None.
        prompt (MergeFeaturesPromptConfig, optional): Will be formatted with the feature
            to send to the LLM. Only used if `how` is set to 'LLM'.
            See `graphragzen.typing.MergeFeaturesPromptConfig`.
            Defaults to MergeFeaturesPromptConfig.
        how (Literal['LLM', 'count', 'mean'], optional): 'LLM' summarizes the features.
            'count' takes the feature that occurs most. 'mean' takes the mean of the feature.
            Defaults to 'LLM'..
        max_input_tokens (int, optional): Only used when how=='LLM'. Maximum input tokens until a
            summary is made. Remaining items in the list will be appended to the summary until
            max_input_tokens is reached again or no items are left. Defaults to 4000.
        max_output_tokens (int, optional): Only used when how=='LLM'. Maximum number of tokens a
            summary can have. Defaults to 500.

    Returns:
        str: summary
    """

    match how.lower():
        case "count":
            return _count_merge(feature_list)
        case "mean":
            return _mean_merge(feature_list)
        case "llm":
            return _LLM_merge(
                entity_name, feature_list, llm, prompt, max_input_tokens, max_output_tokens
            )
        case _:
            # If an exact match is not confirmed, raise exception
            raise Exception(
                "Merging strategy not recognized, must be one of ['LLM', 'count', 'mean']"
            )


def _count_merge(feature_list: List[str]) -> str:
    """Returns the feature description occuring most. Ties are broken by alphabetical order.

    Args:
        feature_list (List[str]): List of feature descriptions

    Returns:
        str: Most occuring feature description
    """
    return Counter(sorted(feature_list)).most_common(1)[0][0]


def _mean_merge(feature_list: List[str]) -> float:
    """Returns the mean of the feature descriptions.

    Args:
        feature_list (List[str]): List of feature descriptions

    Returns:
        str: mean of the feature descriptions.
    """
    # Try and force feature descriptions to floats and average
    float_feature_list = [float(feature) for feature in feature_list]
    return float(mean(float_feature_list))


def _LLM_merge(
    entity_name: str,
    feature_list: List[str],
    llm: Optional[LLM],
    prompt: Optional[MergeFeaturesPromptConfig] = MergeFeaturesPromptConfig(),
    max_input_tokens: int = 4000,
    max_output_tokens: int = 500,
) -> str:
    """Use a LLM to summarize a list of descriptions

    Args:
        entity_name (str): Name of the node or edge
        feature_list (List[str]): feature descriptions to merge
        llm (LLM, optional): Only used if `how` is set to 'LLM'. Dedaults to None.
        prompt (MergeFeaturesPromptConfig, optional): Will be formatted with the feature
            to send to the LLM. Only used if `how` is set to 'LLM'.
            See `graphragzen.typing.MergeFeaturesPromptConfig`.
            Defaults to MergeFeaturesPromptConfig.
        max_input_tokens (int, optional): Only used when how=='LLM'. Maximum input tokens until a
            summary is made. Remaining descriptions will be appended to the summary until
            max_input_tokens is reached again or no descriptions are left. Defaults to 4000.
        max_output_tokens (int, optional): Maximum number of tokens a summary can have.
            Defaults to infinite.

    Returns:
        str: summary
    """

    if llm is None:
        raise Exception("No LLM provided; cannot merge features with strategy 'LLM'")

    if prompt is None:
        raise Exception(
            "No MergeFeaturesPromptConfig provided; cannot merge features with strategy 'LLM'"
        )

    def _summarize(llm: LLM, prompt: MergeFeaturesPromptConfig, max_output_tokens: int) -> str:
        prompt = prompt.prompt.format(**prompt.formatting.model_dump())  # type: ignore
        chat = llm.format_chat([("user", prompt)])
        return llm.run_chat(chat, max_tokens=max_output_tokens)

    usable_tokens = max_input_tokens - _num_tokens_from_string(prompt.prompt, llm.tokenizer)

    descriptions_collected = []
    for feature in feature_list:
        usable_tokens -= _num_tokens_from_string(feature, llm.tokenizer)
        descriptions_collected.append(feature)

        # If buffer is full, or all descriptions have been added, summarize
        if usable_tokens <= 0:
            # Calculate result (final or partial)
            prompt.formatting.entity_name = entity_name
            prompt.formatting.description_list = descriptions_collected
            summarized = _summarize(llm, prompt, max_output_tokens)

            # Add summarization to 'descriptions' to be part of the next possible loop
            descriptions_collected = [summarized]

            # reset values for a possible next loop
            usable_tokens = (
                max_input_tokens
                - _num_tokens_from_string(prompt.prompt, llm.tokenizer)
                - _num_tokens_from_string(summarized, llm.tokenizer)
            )

    if len(descriptions_collected) <= 1:
        return " ".join(descriptions_collected)

    # Final prompt
    prompt.formatting.entity_name = entity_name
    prompt.formatting.description_list = descriptions_collected

    return _summarize(llm, prompt, max_output_tokens)
