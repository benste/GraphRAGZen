from copy import deepcopy

from graphragzen.llm.base_llm import LLM

from .typing import EntityExtractionPromptFormatting, EntityExtractionPrompts


def loop_extraction(
    document: str,
    prompts: EntityExtractionPrompts,
    prompts_formatting: EntityExtractionPromptFormatting,
    llm: LLM,
    max_gleans: int = 5,
) -> str:
    """Extract entities in a loop, asking a few times if all entities are extracted using the
        correct prompts.

    Args:
        document (str): Document to extract entities from
        prompts (EntityExtractionPrompts): Base prompts.
            See `graphragzen.typing.EntityExtractionPrompts`
        prompts_formatting (EntityExtractionPromptFormatting): Values used to format the entity
            extraction prompt. See `graphragzen.typing.EntityExtractionPromptFormatting`.
        llm (LLM)
        max_gleans (int, optional): How often the LLM should be asked if all entities have been
            extracted. Defaults to 5.

    Returns:
        str: Raw description of extracted entities.
    """

    prompts_formatting.input_text = document

    # First entity extraction
    prompt = prompts.entity_extraction_prompt.format(**prompts_formatting.model_dump())
    chat = llm.format_chat([("user", prompt)])
    llm_output = llm.run_chat(chat).removesuffix(prompts_formatting.completion_delimiter)
    chat = llm.format_chat([("model", llm_output)], chat)
    extracted_entities = deepcopy(llm_output)

    # Extract more entities LLM might have missed first time around
    for _ in range(max_gleans):
        chat = llm.format_chat([("user", prompts.continue_prompt)], chat)
        if llm.num_chat_tokens(chat) >= llm.config.context_size:
            # Context limit reached, can't extract more
            break

        llm_output = llm.run_chat(chat).removesuffix(prompts_formatting.completion_delimiter)
        extracted_entities += prompts_formatting.record_delimiter + llm_output or ""
        chat = llm.format_chat([("model", llm_output)], chat)

        # Check if the LLM thinks there are still entities missing
        loop_chat = llm.format_chat([("user", prompts.loop_prompt)], chat)
        continuation = llm.run_chat(loop_chat).removesuffix(prompts_formatting.completion_delimiter)
        if "yes" in continuation.lower():
            break

    return extracted_entities
