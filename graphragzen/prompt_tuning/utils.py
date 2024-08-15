from typing import Any, List

from graphragzen.llm.base_llm import LLM

from .typing import GenerateDomainConfig, GeneratePersonaConfig


def generate_domain(llm: LLM, documents: List[str], **kwargs: Any) -> str:
    """Generate a domain to use for GraphRAG prompts.

    Args:
        llm (LLM)
        documents (List[str]): Sample of documents that later will be used to create a graph.
            You likely want this to be chunks of the whole documents.
        prompt (str, optional): Prompt to use for generating a domain.
            If `domain` is not specified this will be used to infer the domain.
            Defaults to `graphragzen.prompts.prompt_tuning.domain.GENERATE_DOMAIN_PROMPT`.
        domain (str, optional): The domain relevant to a set of documents.
            If not specified, the `prompt` will be used to infer the domain. Defaults to None.

    Returns:
        str: domain
    """
    config = GenerateDomainConfig(**kwargs)  # type: ignore

    if config.domain:
        # User provided a domain, no need to generate one
        return config.domain

    docs_str = "\n".join(documents)
    domain_prompt = config.prompt.format(input_text=docs_str)
    chat = llm.format_chat([("user", domain_prompt)])
    return llm.run_chat(chat)


def generate_persona(llm: LLM, domain: str, **kwargs: Any) -> str:
    """Generate a persona relevant to a domain to use for GraphRAG prompts.

    Args:
        llm (LLM)
        domain (str): To base the persona on
        prompt (str, optional): Prompt to use for generating a persona.
            Defaults to `graphragzen.prompts.prompt_tuning.persona.GENERATE_PERSONA_PROMPT`.

    Returns:
        str: persona
    """
    config = GeneratePersonaConfig(**kwargs)  # type: ignore

    persona_prompt = config.prompt.format(domain=domain)

    chat = llm.format_chat([("user", persona_prompt)])
    return llm.run_chat(chat)
