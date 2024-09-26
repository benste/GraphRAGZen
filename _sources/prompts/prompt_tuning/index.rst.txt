.. _prompt_tuning_prompts_label:

Prompt tuning Prompts
======================

These are the prompts used to create the prompts that can be used to create a graph using an LLM.

The prompts created by these prompts are tuned to the documents, making them more likely to create
a relevant graph.

The prompt for entity extraction is created like such:
    1. Domain: We fist ask the LLM to create the domains that the documents span
    2. Persona: with the domains the LLM can create a persona (e.g. You are an expert {{role}}.
        You are skilled at {{relevant skills}})
    3. Entity categories: using the domain and persona we ask the LLM to extract from the documents
        the categories a node could get (e.g. person, school of thought, ML)
    4. Examples: Using all of the above we ask the LLM to create some example document->entities
        extracted
    5. Entity extraction prompt: We merge all of the above information in a prompt that can be
        used to extract entities

.. toctree::
    :maxdepth: 2

    domain.rst
    persona.rst
    entity_categories.rst
    entity_relationship.rst
    entity_extraction.rst
    summarization_prompt.rst
