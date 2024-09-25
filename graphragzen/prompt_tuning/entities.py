import asyncio
import json
from typing import List, Optional

from graphragzen.async_tools import async_loop
from graphragzen.feature_merging import _num_tokens_from_string
from graphragzen.llm.base_llm import LLM
from graphragzen.prompts.prompt_tuning import (
    entity_categories,
    entity_extraction,
    entity_relationship,
)
from pydantic._internal._model_construction import ModelMetaclass
from tqdm import tqdm

from .llm_output_structures import ExtractedCategories


def generate_entity_categories(
    llm: LLM,
    documents: List[str],
    domain: str,
    persona: str,
    prompt: str = entity_categories.GENERATE_ENTITY_CATEGORIES_PROMPT,
    output_structure: ModelMetaclass = ExtractedCategories,
    entity_categories: Optional[List[str]] = None,
) -> list[str]:
    """Generate entity categories from a given set of (small) documents.

    Example Output
    ['military unit', 'organization', 'person', 'location', 'event', 'date', 'equipment']

    Args:
        llm (LLM):
        documents (List[str]): Sample of documents that later will be used to create a graph.
            You likely want this to be chunks of the whole documents.
        domain (str): Relevant to the documents
        persona (str): Relevant to the domain
        prompt (str, optional): Prompt to use for generating entity categories.
            Defaults to `graphragzen.prompts.prompt_tuning.entity_categories.GENERATE_ENTITY_CATEGORIES_PROMPT`
            If `entity_categories` is not specified this will be used to infer the entity categories.
        output_structure (ModelMetaclass, optional): Output structure to force, e.g. grammars
            from llama.cpp. This SHOULD NOT be an instance of the pydantic model, just the
            reference.
            Correct = BaseLlamaCpp("some text", MyPydanticModel)
            Wrong = BaseLlamaCpp("some text", MyPydanticModel())
            Defaults to graphragzen.prompt_tuning.llm_output_structures.ExtractedCategories
        entity_categories (List[str], optional): The entity categories relevant to a set of documents.
            If not specified, the `prompt` will be used to infer the entity categories. Defaults to None.

    Returns:
        list[str]: entity categories
    """  # noqa: E501

    if entity_categories:
        # User provided entity categories, no need to generate them
        return entity_categories

    docs_str = "\n".join(documents)
    entity_categories_prompt = prompt.format(domain=domain, input_text=docs_str)

    chat = llm.format_chat([("model", persona), ("user", entity_categories_prompt)])
    response = llm.run_chat(chat, output_structure=output_structure)

    categories = json.loads(response).get("categories", [])
    return categories


def generate_entity_relationship_examples(
    llm: LLM,
    documents: List[str],
    persona: str,
    entity_categories: list[str],
    prompt: str = entity_relationship.ENTITY_RELATIONSHIPS_GENERATION_PROMPT,
    example_template: str = entity_relationship.EXAMPLE_RELATION_TEMPLATE,
    max_examples: int = 5,
    async_llm_calls: bool = False,
) -> list[str]:
    """Generate a list of entity/relationships examples for use in generating an entity
    extraction prompt.

    Will return in tuple_delimiter format depending

    Args:
        llm (LLM):
        documents (List[str]): Sample of documents that later will be used to create a graph.
            You likely want this to be chunks of the whole documents.
        persona (str): Relevant to the domain
        entity_categories (list[str]): Generated from the documents by `generate_entity_categories`
        prompt (str, optional): Prompt to use for generating entity/relationships examples.
            Defaults to `graphragzen.prompts.prompt_tuning.entity_relationship.ENTITY_RELATIONSHIPS_GENERATION_PROMPT`
        example_template (str, optional): The template of example extracted entities that will
            be formatted using, among others, the entity relationships extracted using the
            prompt. Defaults to `graphragzen.prompts.prompt_tuning.entity_relationship.EXAMPLE_EXTRACTION_TEMPLATE`
        max_examples (int, optional): Number of examples to create. Defaults to 5.
        async_llm_calls: If True will call the LLM asynchronously. Only applies to communication
            with an LLM using `OpenAICompatibleClient`, in-memory LLM's loaded using
            llama-cpp-python will always be called synchronously. Defaults to False.

    Returns:
        list[str]: Entity relationship examples
    """  # noqa: E501

    entity_categories_str = ", ".join(entity_categories)
    sampled_documents = documents[:max_examples]

    history = llm.format_chat([("model", persona)])
    messages = [
        prompt.format(entity_categories=entity_categories_str, input_text=doc)
        for doc in sampled_documents
    ]

    chats = [llm.format_chat([("user", message)], history) for message in messages]

    if async_llm_calls:
        loop = asyncio.get_event_loop()
        entity_relations = loop.run_until_complete(
            async_loop(llm.a_run_chat, chats, "generating example entity relations asynchronously")
        )
    else:
        entity_relations = [
            llm.run_chat(chat) for chat in tqdm(chats, desc="generating example entity relations")
        ]

    # Format the examples and return
    return [
        example_template.format(
            n=i + 1, input_text=doc, entity_categories=entity_categories_str, output=entity_relation
        )
        for i, (doc, entity_relation) in enumerate(zip(sampled_documents, entity_relations))
    ]


def create_entity_extraction_prompt(
    llm: LLM,
    entity_categories: List[str],
    entity_relationship_examples: List[str],
    prompt_template: str = entity_extraction.ENTITY_EXTRACTION_TEMPLATE,
    prompt_max_tokens: int = 3000,
) -> str:
    """Create a prompt for entity extraction.

    This does not use an LLM but simply formats a string using the inputs and makes sure the resulting
    prompt is not too large.

    Args:
        llm (LLM):
        entity_categories (List[str]): The categories of entities to extract
            (e.g. ['person', 'profession', 'location'])
        entity_relationship_examples (List[str]): Generated by
            `generate_entity_relationship_examples`
        prompt_template (str, optional): The template that will be formatted to the final prompt.
            Defaults to `graphragzen.prompts.prompt_tuning.entity_extraction.ENTITY_EXTRACTION_PROMPT`
        prompt_max_tokens (int, optional): Maximum number of tokens the final prompt is allowed to be.
            Defaults to 3000

    Returns:
        str: Prompt to use for entity extraction
    """  # noqa: E501

    prompt = prompt_template
    entity_categories_string = ", ".join(entity_categories)

    tokens_left = (
        prompt_max_tokens
        - _num_tokens_from_string(prompt, llm.tokenizer)
        - _num_tokens_from_string(entity_categories_string, llm.tokenizer)
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
    return prompt.format(entity_categories=entity_categories_string, examples=examples_prompt)
