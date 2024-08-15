from typing import List

from graphragzen.llm.base_llm import LLM
from tqdm import tqdm

from .typing import EntityExtractionPromptFormatting, EntityExtractionPrompts


def loop_extraction(
    documents: List[str],
    prompts: EntityExtractionPrompts,
    prompts_formatting: EntityExtractionPromptFormatting,
    llm: LLM,
    max_gleans: int = 5,
) -> List[str]:
    """Extract entities in a loop, asking a few times if all entities are extracted using the
        correct prompts.

    Args:
        documents (List[str]): Documents to extract entities from
        prompts (EntityExtractionPrompts): Base prompts.
            See `graphragzen.typing.EntityExtractionPrompts`
        prompts_formatting (EntityExtractionPromptFormatting): Values used to format the entity
            extraction prompt. See `graphragzen.typing.EntityExtractionPromptFormatting`.
        llm (LLM)
        max_gleans (int, optional): How often the LLM should be asked if all entities have been
            extracted. Defaults to 5.

    Returns:
        List[str]: Per document raw description of extracted entities.
    """
    llm_raw_output = []
    for doc in tqdm(documents, position=1):
        prompts_formatting.input_text = doc

        # First entity extraction
        prompt = prompts.entity_extraction_prompt.format(**prompts_formatting.model_dump())
        chat = llm.format_chat([("user", prompt)])
        raw_output = llm.run_chat(chat).removesuffix(prompts_formatting.completion_delimiter)
        llm_raw_output.append(raw_output)
        chat = llm.format_chat([("model", raw_output)], chat)

        # Extract more entities LLM might have missed first time around
        for _ in range(max_gleans):
            chat = llm.format_chat([("user", prompts.continue_prompt)], chat)
            raw_output = llm.run_chat(chat).removesuffix(prompts_formatting.completion_delimiter)
            llm_raw_output[-1] += prompts_formatting.record_delimiter + raw_output or ""
            chat = llm.format_chat([("model", raw_output)], chat)

            # Check if the LLM thinks there are still entities missing
            loop_chat = llm.format_chat([("user", prompts.loop_prompt)], chat)
            continuation = llm.run_chat(loop_chat).removesuffix(
                prompts_formatting.completion_delimiter
            )
            if "yes" in continuation.lower():
                break

    return llm_raw_output
