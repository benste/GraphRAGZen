from typing import Optional, List
from pydantic import BaseModel, field_validator

from graphragzen.prompts.prompt_tuning import domain, entity_types, persona, entity_relationship, entity_extraction, entity_summarization

class GenerateDomainConfig(BaseModel):
    """Config for generating a domain relevant to a set of documents

    Args:
        prompt (str, optional): Prompt to use for generating a domain. Defaults to `domain.GENERATE_DOMAIN_PROMPT`.
            If `domain` is not specified this will be used to infer the domain.
        domain (str, optional): The domain relevant to a set of documents. 
            If not specified, the `prompt` will be used to infer the domain. Defaults to None.
    """
    prompt: Optional[str] = domain.GENERATE_DOMAIN_PROMPT
    domain: Optional[str] = None
    
    @field_validator('prompt', mode="before")
    def set_prompt(cls, prompt):
        return prompt or domain.GENERATE_DOMAIN_PROMPT
    
class GeneratePersonaConfig(BaseModel):
    """Config for generating a persona relevant to the domain of a set of documents

    Args:
        prompt (str, optional): Prompt to use for generating a persona. Defaults to `persona.GENERATE_PERSONA_PROMPT`.
    """
    prompt: Optional[str] = persona.GENERATE_PERSONA_PROMPT
    
    @field_validator('prompt', mode="before")
    def set_prompt(cls, prompt):
        return prompt or  persona.GENERATE_PERSONA_PROMPT
    
class GenerateEntityTypesConfig(BaseModel):
    """Config for generating a entity types relevant to a set of documents

    Args:
        prompt (str, optional): Prompt to use for generating entity types.
            Defaults to `entity_types.GENERATE_ENTITY_TYPE_PROMPT`.
            If `entity_types` is not specified this will be used to infer the entity types.
        entity_types (List[str], optional): The entity types relevant to a set of documents. 
            If not specified, the `prompt` will be used to infer the entity types. Defaults to None.
    """
    prompt: str = entity_types.GENERATE_ENTITY_TYPE_PROMPT
    entity_types: Optional[List[str]]
    
    @field_validator('prompt', mode="before")
    def set_prompt(cls, prompt):
        return prompt or entity_types.GENERATE_ENTITY_TYPE_PROMPT
    
class EntityRelationshipExamplesConfig(BaseModel):
    """Config for generating a list of entity/relationships examples

    Args:
        prompt (str, optional): Prompt to use for generating entity/relationships examples.
            Defaults to `entity_relationship.ENTITY_RELATIONSHIPS_GENERATION_PROMPT`
        example_template (str, optional): The template of example extracted entities that will
            be formatted using, among others, the entity relationships extracted using the
            prompt. Defaults to `entity_relationship.EXAMPLE_EXTRACTION_TEMPLATE`
        max_examples (int, optional): Number of examples to create.
    """
    prompt: Optional[str] = entity_relationship.ENTITY_RELATIONSHIPS_GENERATION_PROMPT
    example_template: Optional[str] = entity_relationship.EXAMPLE_EXTRACTION_TEMPLATE
    max_examples: Optional[int] = 3
    
    @field_validator('prompt', mode="before")
    def set_prompt(cls, prompt):
        return prompt or entity_relationship.ENTITY_RELATIONSHIPS_GENERATION_PROMPT
    
    @field_validator('example_template', mode="before")
    def set_example_template(cls, example_template):
        return example_template or entity_relationship.EXAMPLE_EXTRACTION_TEMPLATE
    
class CreateEntityExtractionPromptConfig(BaseModel):
    """Config for creating an entity extraction prompt.

    Args:
        prompt (str, optional): Base prompt that will be formatted to the final prompt.
            Defaults to `entity_extraction.ENTITY_EXTRACTION_PROMPT`
        prompt_max_tokens (int, optional): Maximum number of tokens the final prompt is allowed to be. Defaults to 3000
    """
    prompt: Optional[str] = entity_extraction.ENTITY_EXTRACTION_TEMPLATE
    prompt_max_tokens: Optional[int] = 3000
    
    @field_validator('prompt', mode="before")
    def set_prompt(cls, prompt):
        return prompt or entity_extraction.ENTITY_EXTRACTION_TEMPLATE
    
class CreateEntitySummarizationPromptConfig(BaseModel):
    """Config for generating a prompt that can be used to summarize entities

    Args:
        prompt_template (str, optional): The template that will be formatted using a persona.
            Defaults to `entity_summarization.ENTITY_SUMMARIZATION_TEMPLATE`
    """
    prompt_template: Optional[str] = entity_summarization.ENTITY_SUMMARIZATION_TEMPLATE
    
    @field_validator('prompt_template', mode="before")
    def set_prompt(cls, prompt_template):
        return prompt_template or  entity_summarization.ENTITY_SUMMARIZATION_TEMPLATE

