from typing import List

from tqdm import tqdm

from graphragzen.typing import (
    GenerateEntityRelationshipExamplesConfig,
    GenerateDomainConfig,
    GeneratePersonaConfig,
    GenerateEntityTypesConfig,
    CreateEntityExtractionPromptConfig,
    CreateEntitySummarizationPromptConfig,
)
from graphragzen.llm.base_llm import LLM
from graphragzen.description_summarization import _num_tokens_from_string


def generate_domain(llm: LLM, documents: List[str], **kwargs: GenerateDomainConfig) -> str:
    """Generate a domain to use for GraphRAG prompts.

    Args:
        llm (LLM)
        documents (List[str]): Sample of documents that later will be used to create a graph.
            You likely want this to be chunks of the whole documents.

    Kwargs:
        prompt (str, optional): Prompt to use for generating a domain.
            If `domain` is not specified this will be used to infer the domain.
            Defaults to `graphragzen.prompts.prompt_tuning.domain.GENERATE_DOMAIN_PROMPT`.
        domain (str, optional): The domain relevant to a set of documents.
            If not specified, the `prompt` will be used to infer the domain. Defaults to None.

    Returns:
        str: domain
    """
    config = GenerateDomainConfig(**kwargs)

    if config.domain:
        # User provided a domain, no need to generate one
        return config.domain

    docs_str = "\n".join(documents)
    domain_prompt = config.prompt.format(input_text=docs_str)
    chat = llm.format_chat([("user", domain_prompt)])
    return llm.run_chat(chat)


def generate_persona(llm: LLM, domain: str, **kwargs: GeneratePersonaConfig) -> str:
    """Generate a persona relevant to a domain to use for GraphRAG prompts.

    Args:
        llm (LLM)
        domain (str): To base the persona on

    Kwargs:
        prompt (str, optional): Prompt to use for generating a persona.
            Defaults to `graphragzen.prompts.prompt_tuning.persona.GENERATE_PERSONA_PROMPT`.

    Returns:
        str: persona
    """
    config = GeneratePersonaConfig(**kwargs)

    persona_prompt = config.prompt.format(domain=domain)

    chat = llm.format_chat([("user", persona_prompt)])
    return llm.run_chat(chat)


def generate_entity_types(
    llm: LLM,
    documents: List[str],
    domain: str,
    persona: str,
    **kwargs: GenerateEntityTypesConfig,
) -> str | list[str]:
    """Generate entity type categories from a given set of (small) documents.

        Example Output:
        ['military unit', 'organization', 'person', 'location', 'event', 'date', 'equipment']

    Args:
        llm (LLM)
        documents (List[str]): Sample of documents that later will be used to create a graph.
            You likely want this to be chunks of the whole documents.
        domain (str): Relevant to the documents
        persona (str): Relevant to the domain

    Kwargs:
        prompt (str, optional): Prompt to use for generating entity types.
            Defaults to `graphragzen.prompts.prompt_tuning.entity_types.GENERATE_ENTITY_TYPE_PROMPT`
            If `entity_types` is not specified this will be used to infer the entity types.
        entity_types (List[str], optional): The entity types relevant to a set of documents.
            If not specified, the `prompt` will be used to infer the entity types. Defaults to None.

    Returns:
        str | list[str]: entity types
    """
    config = GenerateEntityTypesConfig(**kwargs)

    if config.entity_types:
        # User provided entity types, no need to generate them
        return config.entity_types

    docs_str = "\n".join(documents)
    entity_types_prompt = config.prompt.format(domain=domain, input_text=docs_str)
    chat = llm.format_chat([("model", persona), ("user", entity_types_prompt)])
    return llm.run_chat(chat)


def generate_entity_relationship_examples(
    llm: LLM,
    documents: List[str],
    persona: str,
    entity_types: list[str],
    **kwargs: GenerateEntityRelationshipExamplesConfig,
) -> list[str]:
    """Generate a list of entity/relationships examples for use in generating an entity
        extraction prompt.

    Will return in tuple_delimiter format depending

    Args:
        llm (LLM)
        documents (List[str]): Sample of documents that later will be used to create a graph.
            You likely want this to be chunks of the whole documents.
        persona (str): Relevant to the domain
        entity_types (list[str]): Generated from the documents by `generate_entity_types`

    Kwargs:
        prompt (str, optional): Prompt to use for generating entity/relationships examples.
            Defaults to `graphragzen.prompts.prompt_tuning.entity_relationship.ENTITY_RELATIONSHIPS_GENERATION_PROMPT`
        example_template (str, optional): The template of example extracted entities that will
            be formatted using, among others, the entity relationships extracted using the
            prompt. Defaults to graphragzen.prompts.prompt_tuning.entity_relationship.EXAMPLE_EXTRACTION_TEMPLATE`
        max_examples (int, optional): Number of examples to create.

    Returns:
        list[str]: examples
    """  # noqa: E501
    config = GenerateEntityRelationshipExamplesConfig(**kwargs)

    entity_types_str = ", ".join(entity_types)
    sampled_documents = documents[: config.max_examples]

    history = llm.format_chat([("model", persona)])
    messages = [
        config.prompt.format(entity_types=entity_types_str, input_text=doc)
        for doc in sampled_documents
    ]

    chats = [llm.format_chat([("user", message)], history) for message in messages]
    entity_relations = [
        llm.run_chat(chat) for chat in tqdm(chats, desc="generating example entity relations")
    ]

    # Format the examples and return
    return [
        config.example_template.format(
            n=i + 1, input_text=doc, entity_types=entity_types_str, output=entity_relation
        )
        for i, (doc, entity_relation) in enumerate(zip(sampled_documents, entity_relations))
    ]


def create_entity_extraction_prompt(
    llm: LLM,
    entity_types: List[str],
    entity_relationship_examples: List[str],
    **kwargs: CreateEntityExtractionPromptConfig,
) -> str:
    """
    Create a prompt for entity extraction.

    This does not use LLM but simply formats a string using the inputs and makes sure the resulting
        prompt is not too large.

    Args:
        llm (LLM)
        Entity types (List[str]): The types of entities to extract
            (e.g. ['person', 'profession', 'location'])
        entity_relationship_examples (List[str]): Generated by
            `generate_entity_relationship_examples`

    Kwargs:
        prompt_template (str, optional): The template that will be formatted to the final prompt.
            Defaults to `graphragzen.prompts.prompt_tuning.entity_extraction.ENTITY_EXTRACTION_PROMPT`
        prompt_max_tokens (int, optional): Maximum number of tokens the final prompt is allowed to be.
            Defaults to 3000

    Returns:
        str: Prompt to use for entity extraction
    """  # noqa: E501
    config = CreateEntityExtractionPromptConfig(**kwargs)

    prompt = config.prompt
    entity_types = ", ".join(entity_types)

    tokens_left = (
        config.prompt_max_tokens
        - _num_tokens_from_string(prompt, llm.tokenizer)
        - _num_tokens_from_string(entity_types, llm.tokenizer)
    )

    examples_prompt = ""

    # Iterate over examples, while we have tokens left or examples left
    for i, example in enumerate(entity_relationship_examples):
        example_tokens = _num_tokens_from_string(example, llm.tokenizer)

        # Squeeze in at least one example
        if i > 0 and example_tokens > tokens_left:
            break

        examples_prompt += example
        tokens_left -= example_tokens

    # Format prompt and return
    return prompt.format(entity_types=entity_types, examples=examples_prompt)


def create_entity_summarization_prompt(
    persona: str, **kwargs: CreateEntitySummarizationPromptConfig
) -> str:
    """Create a prompt for entity summarization.

     Args:
        persona (str): Relevant to the domain

    Kwargs:
        prompt_template (str, optional): The template that will be formatted using a persona.
            Defaults to `graphragzen.prompts.prompt_tuning.entity_summarization.ENTITY_SUMMARIZATION_TEMPLATE`

    Returns:
        str: Prompt to use for entity summarization
    """  # noqa: E501
    config = CreateEntitySummarizationPromptConfig(**kwargs)

    return config.prompt_template.format(persona=persona)
