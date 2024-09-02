from typing import List, Optional

from graphragzen.prompts.default_prompts import entity_extraction_prompts

from ..typing.MappedBaseModel import MappedBaseModel


class EntityExtractionPromptFormatting(MappedBaseModel):
    """Values used to format the entity extraction prompt

    Args:
        entity_categories (List[str], optional):  The categories that can be assigned to entities.
            Defaults to ['organization', 'person', 'geo', 'event'].
        input_text (str, optional): The text to extract entities from. Defaults to None.
    """  # noqa: W605

    entity_categories: List[str] = ["organization", "person", "geo", "event"]
    input_text: Optional[str] = None


class EntityExtractionPrompts(MappedBaseModel):
    """Base prompts for entity extraction

    Args:
        entity_extraction_prompt (str, optional): Main extraction prompt.
            Defaults to `graphragzen.prompts.default_prompts.entity_extraction_prompts.ENTITY_EXTRACTION_PROMPT`
        continue_prompt (str, optional): Prompt that asks the LLM to continue extracting entities.
            Defaults to `graphragzen.prompts.default_prompts.entity_extraction_prompts.CONTINUE_PROMPT`
        loop_prompt (str, optional): Prompt that asks the LLM if there are more entities to extract.
            Defaults to `graphragzen.prompts.default_prompts.entity_extraction_prompts.LOOP_PROMPT`
    """  # noqa: E501

    entity_extraction_prompt: str = entity_extraction_prompts.ENTITY_EXTRACTION_PROMPT
    continue_prompt: str = entity_extraction_prompts.CONTINUE_PROMPT
    loop_prompt: str = entity_extraction_prompts.LOOP_PROMPT


class EntityExtractionPromptConfig(MappedBaseModel):
    """Config for the prompt used to extract entities

    Args:
        prompt (EntityExtractionPrompts, optional): Base prompts
        formatting (EntityExtractionPromptFormatting, optional): Values used to format the base
            prompts.
    """

    prompts: EntityExtractionPrompts = EntityExtractionPrompts()
    formatting: EntityExtractionPromptFormatting = EntityExtractionPromptFormatting()
