from typing import List
from functools import partial

import networkx as nx


from graphragzen.typing import DescriptionSummarizationPromptConfig, DescriptionSummarizationConfig
from graphragzen.llm.base_llm import LLM
from .utils import _num_tokens_from_string
   

def summarize_descriptions(graph: nx.Graph, llm: LLM, prompt: DescriptionSummarizationPromptConfig, **kwargs: DescriptionSummarizationConfig) -> nx.Graph:
    """Summarize lists of descriptions for each node or edge

    Args:
        graph (nx.Graph): With edges and nodes expected to have the feature 'description'. The descriptions are expected
            to be delimited by Kwargs["feature_delimiter"]
        llm (LLM)
        prompt (DescriptionSummarizationPromptConfig): See from graphragzen.typing.DescriptionSummarizationPromptConfig

    Kwargs:
        feature_delimiter (str, optional): When during entity extraction the same node or edge was found multiple 
            times, features were concatenated using this delimiter. We will make a list of descriptions by splitting
            on this delimiter. Defaults to '\n'.
        max_input_tokens (int, optional): Maximum input tokens until a summarization is made. Remaining descriptions
            will be appended to the summarization until max_input_tokens is reached again or no descriptions are left.
            Defaults to 4000.
        max_output_tokens (int, optional): Maximum number of tokens a summary can have. Defaults to 500.
        
    Returns:
        nx.Graph
    """
    config = DescriptionSummarizationConfig(**kwargs)
    
    item_summarizer = partial(_summarize_item, llm=llm, prompt=prompt, max_input_tokens=config.max_input_tokens, max_output_tokens=config.max_output_tokens)
    
    i = 1
    for node in graph.nodes(data=True):
        print(f"summarizing descriptions of node {i} of {len(graph.nodes())}")
        entity_name = node[0]
        descriptions = sorted(set(node[1].get("description", "").split(config.feature_delimiter)))
        graph.nodes[entity_name]["description"] = item_summarizer(entity_name=entity_name, descriptions=descriptions)
        i+=1
        
    i = 1
    for edge in graph.edges(data=True):
        print(f"summarizing descriptions of edge {i} of {len(graph.edges())}")
        entity_name = edge[:2]
        descriptions = sorted(set(edge[2].get("description", "").split(config.feature_delimiter)))
        graph.edges[entity_name]["description"] = item_summarizer(entity_name=entity_name, descriptions=descriptions)
        i+=1
        
    return graph

def _summarize_item(entity_name: str, descriptions: List[str], llm: LLM, prompt: DescriptionSummarizationPromptConfig, max_input_tokens: int = 4000, max_output_tokens: int = 500) -> str:
    """Summarize a list of descriptions for a single node or edge

    Args:
        entity_name (str): Name of the node or edge
        descriptions (List[str]): descriptions to summarize
        llm (LLM)
        prompt (DescriptionSummarizationPromptConfig): See from graphragzen.typing.DescriptionSummarizationPromptConfig
        max_input_tokens (int, optional): Maximum input tokens until a summarization is made. Remaining descriptions
            will be appended to the summarization until max_input_tokens is reached again or no descriptions are left.
            Defaults to 4000.
        max_output_tokens (int, optional): Maximum number of tokens a summary can have. Defaults to 500.    
        
    Returns:
        str: summary
    """  
    usable_tokens = max_input_tokens - _num_tokens_from_string(
            prompt.prompt, llm.tokenizer
        )

    descriptions_collected = []
    for description in descriptions:
        usable_tokens -= _num_tokens_from_string(description, llm.tokenizer)
        descriptions_collected.append(description)
        
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
                - _num_tokens_from_string(prompt, llm.tokenizer)
                - _num_tokens_from_string(summarized, llm.tokenizer)
            )

    if len(descriptions_collected) <= 1:
        return " ".join(descriptions_collected)
    
    # Final prompt
    prompt.formatting.entity_name = entity_name
    prompt.formatting.description_list = descriptions_collected
    
    return _summarize(llm, prompt, max_output_tokens)
        

def _summarize(llm: LLM, prompt: DescriptionSummarizationPromptConfig, max_output_tokens: int = -1) -> str:
    """Use a LLM to summarize a list of descriptions

    Args:
        llm (LLM)
        prompt (DescriptionSummarizationPromptConfig): Already formatted with the entity name and descriptions
        max_output_tokens (int, optional): Maximum number of tokens a summary can have. Defaults to infinite.

    Returns:
        str: summary
    """
    prompt = prompt.prompt.format(**prompt.formatting.model_dump())
    chat = llm.format_chat([("user", prompt)])
    return llm.run_chat(chat, max_tokens=max_output_tokens)
