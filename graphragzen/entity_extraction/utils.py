from typing import List, Optional

from graphragzen.llm.base_llm import LLM
from pydantic._internal._model_construction import ModelMetaclass

from .typing import EntityExtractionPromptFormatting, EntityExtractionPrompts


def loop_extraction(
    document: str,
    prompts: EntityExtractionPrompts,
    prompts_formatting: EntityExtractionPromptFormatting,
    llm: LLM,
    max_gleans: int = 5,
    output_structure: Optional[ModelMetaclass] = None,
) -> List[str]:
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
        output_structure (ModelMetaclass, optional): Output structure to force, using e.g. grammars
            from llama.cpp.

    Returns:
        List[str]: Raw json string of extracted entities.
    """

    prompts_formatting.input_text = document

    # First entity extraction
    prompt = prompts.entity_extraction_prompt.format(**prompts_formatting.model_dump())
    chat = llm.format_chat([("user", prompt)])
    llm_output = llm.run_chat(chat, output_structure=output_structure)
    chat = llm.format_chat([("model", llm_output)], chat)
    extracted_entities = [llm_output]

    # Extract more entities LLM might have missed first time around
    for i in range(max_gleans):
        continue_prompt = prompts.continue_prompt.format(**prompts_formatting.model_dump())
        chat = llm.format_chat([("user", continue_prompt)], chat)
        if llm.num_chat_tokens(chat) >= llm.context_size:
            # Context limit reached, can't extract more
            break

        llm_output = llm.run_chat(chat, output_structure=output_structure)

        extracted_entities.append(llm_output or "")
        chat = llm.format_chat([("model", llm_output)], chat)

        # Check if the LLM thinks there are still entities missing
        if i < max_gleans - 1:
            loop_chat = llm.format_chat([("user", prompts.loop_prompt)], chat)
            continuation = llm.run_chat(loop_chat)
            if "yes" in continuation.lower():
                break

    return extracted_entities


async def a_loop_extraction(
    document: str,
    prompts: EntityExtractionPrompts,
    prompts_formatting: EntityExtractionPromptFormatting,
    llm: LLM,
    max_gleans: int = 5,
    output_structure: Optional[ModelMetaclass] = None,
) -> List[str]:
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
        output_structure (ModelMetaclass, optional): Output structure to force, using e.g. grammars
            from llama.cpp.

    Returns:
        List[str]: Raw json string of extracted entities.
    """

    prompts_formatting.input_text = document

    # First entity extraction
    prompt = prompts.entity_extraction_prompt.format(**prompts_formatting.model_dump())
    chat = llm.format_chat([("user", prompt)])
    llm_output = await llm.a_run_chat(chat, output_structure=output_structure)
    chat = llm.format_chat([("model", llm_output)], chat)
    extracted_entities = [llm_output]

    # Extract more entities LLM might have missed first time around
    for i in range(max_gleans):
        continue_prompt = prompts.continue_prompt.format(**prompts_formatting.model_dump())
        chat = llm.format_chat([("user", continue_prompt)], chat)
        if llm.num_chat_tokens(chat) >= llm.context_size:
            # Context limit reached, can't extract more
            break

        llm_output = await llm.a_run_chat(chat, output_structure=output_structure)

        extracted_entities.append(llm_output or "")
        chat = llm.format_chat([("model", llm_output)], chat)

        # Check if the LLM thinks there are still entities missing
        if i < max_gleans - 1:
            loop_chat = llm.format_chat([("user", prompts.loop_prompt)], chat)
            continuation = await llm.a_run_chat(loop_chat)
            if "yes" in continuation.lower():
                break

    return extracted_entities
