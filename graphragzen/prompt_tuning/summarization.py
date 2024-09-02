from graphragzen.prompts.prompt_tuning import entity_summarization


def create_description_summarization_prompt(
    persona: str,
    prompt_template: str = entity_summarization.ENTITY_SUMMARIZATION_TEMPLATE,
) -> str:
    """Create a prompt for entity summarization.

    Args:
        persona (str): Relevant to the domain
        prompt_template (str, optional): The template that will be formatted using a persona.
            Defaults to `graphragzen.prompts.prompt_tuning.entity_summarization.ENTITY_SUMMARIZATION_TEMPLATE`

    Returns:
        str: Prompt to use for entity summarization
    """  # noqa: E501

    return prompt_template.format(persona=persona)
