from typing import List, Literal, Optional

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


class MergeFeaturesConfig(MappedBaseModel):
    """Config for the merging a feature of node and edge descriptions

    During entity extraction the same node or edge can be found multiple times, at which point
    features are concatenated using a delimiter. This can be made into a list again and merged into
    a single desrciption.

    Args:
        feature (str): The feature attached to a graph entity (node or edge) to merge.
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
    """

    feature: str
    how: Literal["LLM", "count", "mean"] = "LLM"
    feature_delimiter: str = "\n"  #
    max_input_tokens: int = 4000
    max_output_tokens: int = 500
