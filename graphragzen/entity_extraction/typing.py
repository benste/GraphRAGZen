from typing import List, Optional

from graphragzen.prompts.default_prompts import entity_extraction_prompts

from ..typing.MappedBaseModel import MappedBaseModel


class EntityExtractionPromptFormatting(MappedBaseModel):
    """Values used to format the entity extraction prompt

    Args:
        tuple_delimiter (str, optional): Delimiter between tuples in an output record.
            Defaults to '\<|>'.
        record_delimiter (str, optional):  Delimiter between records. Defaults to '##'.
        completion_delimiter (str, optional): Delimiter when no more entities can be extracted.
            Defaults to '\<\|COMPLETE|>'.
        entity_types (List[str], optional):  The types that can be assigned to entities.
            Defaults to ['organization', 'person', 'geo', 'event'].
        input_text (str, optional): The text to extract entities from. Defaults to None.
    """

    tuple_delimiter: str = "<|>"
    record_delimiter: str = "##"
    completion_delimiter: str = "<|COMPLETE|>"
    entity_types: List[str] = ["organization", "person", "geo", "event"]
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


class EntityExtractionConfig(MappedBaseModel):
    """Config for the extraction of entities from text

    Args:
        max_gleans (int, optional): How often the LLM should be asked if all entities have been
            extracted from a single text. Defaults to 5.
        column_to_extract (str, optional): Column in a DataFrame that contains the texts to extract
            entities from. Defaults to 'chunk'.
        results_column (str, optional): Column to write the output of the LLM to.
            Defaults to 'raw_entities'.
    """

    max_gleans: int = 5
    column_to_extract: str = "chunk"
    results_column: str = "raw_entities"


class RawEntitiesToGraphConfig(MappedBaseModel):
    """Config for generating a graph from the raw extracted entities

    Args:
        raw_entities_column (str, optional): Column in a DataFrame that contains the output of
            entity extraction. Defaults to 'raw_entities'.
        reference_column (str, optional): Value from this column in the DataFrame will be added to
            the edged and nodes. This allows to reference to the source where entities were
            extracted from when quiring the graph. Defaults to 'chunk_id'.
        feature_delimiter (str, optional): When the same node or edge is found multiple times,
            features are concatenated using this demiliter. Defaults to '\\n'.
    """

    raw_entities_column: str = "raw_entities"
    reference_column: str = "chunk_id"
    feature_delimiter: str = "\n"
