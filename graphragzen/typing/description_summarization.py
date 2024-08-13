from typing import List, Optional
from .MappedBaseModel import MappedBaseModel

from graphragzen.prompts.default_prompts import summarization_prompts
  
  
class DescriptionSummarizationPromptFormatting(MappedBaseModel):
    """Values used to format the description summarization prompt

    Args:
        entity_name (str, optional): Name of the node or edge. Defaults to None.
        entity_name (List[str], optional): List of descriptions for the node or edge. Defaults to None.
    """
    
    entity_name: Optional[str] = None
    description_list: Optional[List[str]] = None
    
class DescriptionSummarizationPromptConfig(MappedBaseModel):
    """Config for the prompt used to summarize descriptions

    Args:
        prompt (str, optional): Base prompt. Defaults to `graphragzen.prompts.default_prompts.summarization_prompts.SUMMARIZE_PROMPT`
        formatting (DescriptionSummarizationPromptFormatting, optional): Values used to format the prompt
    """
    prompt: str = summarization_prompts.SUMMARIZE_PROMPT
    formatting: DescriptionSummarizationPromptFormatting


class DescriptionSummarizationConfig(MappedBaseModel):
    """Config for the summarization of node and edge descriptions

    Args:
        feature_delimiter (str, optional): When during entity extraction the same node or edge was found multiple 
            times, features were concatenated using this delimiter. We will make a list of descriptions by splitting
            on this delimiter. Defaults to '\n'.
        max_input_tokens (int, optional): Maximum input tokens until a summarization is made. Remaining descriptions
            will be appended to the summarization until max_input_tokens is reached again or no descriptions are left.
            Defaults to 4000.
        max_output_tokens (int, optional): Maximum number of tokens a summary can have. Defaults to 500.
    """
    feature_delimiter: str = "\n"  # 
    max_input_tokens: int = 4000
    max_output_tokens: int = 500