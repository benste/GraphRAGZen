from typing import List, Optional

from graphragzen.prompts.default_prompts import summarization_prompts

from ..typing.MappedBaseModel import MappedBaseModel


class MergeFeaturesPromptFormatting(MappedBaseModel):
    """Values used to format the description summarization prompt

    Args:
        entity_name (str, optional): Name of the node or edge. Defaults to None.
        description_list (List[str], optional): List of descriptions for the node or edge.
            Defaults to None.
    """

    entity_name: Optional[str] = None
    description_list: Optional[List[str]] = None


class MergeFeaturesPromptConfig(MappedBaseModel):
    """Config for the prompt used to summarize descriptions

    Args:
        prompt (str, optional): Base prompt.
            Defaults to `graphragzen.prompts.default_prompts.summarization_prompts.SUMMARIZE_PROMPT`
        formatting (DescriptionSummarizationPromptFormatting, optional): Values used to format
            the prompt
    """

    prompt: str = summarization_prompts.SUMMARIZE_PROMPT
    formatting: MergeFeaturesPromptFormatting = MergeFeaturesPromptFormatting()
