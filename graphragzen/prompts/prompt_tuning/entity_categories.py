# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Fine-tuning prompts for entity categories generation."""

GENERATE_ENTITY_CATEGORIES_PROMPT = """
The goal is to study the connections and relations between the entity categories and their features in order to understand all available information from the text.
The user's task is to Identify the relations and structure of the community of interest, specifically within the {domain} domain.
As part of the analysis, you want to identify the entity categories present in the following text.
The entity categories must be relevant to the user task.
Avoid general entity categories such as "other" or "unknown".
This is VERY IMPORTANT: Do not generate redundant or overlapping entity categories. For example, if the text contains "company" and "organization" entity categories, you should return only one of them.
Don't worry about quantity, always choose quality over quantity. And make sure EVERYTHING in your answer is relevant to the context of entity extraction.
And remember, it is ENTITY CATEGORIES what we need.
Return the entity categories in as a list of comma separated strings.
=====================================================================
EXAMPLE SECTION: The following section includes example output. These examples **must be excluded from your answer**.

EXAMPLE 1
Task: Determine the connections and organizational hierarchy within the specified community.
Text: Example_Org_A is a company in Sweden. Example_Org_A's director is Example_Individual_B.
RESPONSE:
organization, person
END OF EXAMPLE 1

EXAMPLE 2
Task: Identify the key concepts, principles, and arguments shared among different philosophical schools of thought, and trace the historical or ideological influences they have on each other.
Text: Rationalism, epitomized by thinkers such as René Descartes, holds that reason is the primary source of knowledge. Key concepts within this school include the emphasis on the deductive method of reasoning.
RESPONSE:
concept, person, school of thought
END OF EXAMPLE 2

EXAMPLE 3
Task: Identify the full range of basic forces, factors, and trends that would indirectly shape an issue.
Text: Industry leaders such as Panasonic are vying for supremacy in the battery production sector. They are investing heavily in research and development and are exploring new technologies to gain a competitive edge.
RESPONSE:
organization, technology, sectors, investment strategies
END OF EXAMPLE 3
======================================================================

======================================================================
REAL DATA: The following section is the real data. You should use only this real data to prepare your answer. Generate Entity Categories only.
Task: Identify the relations and structure of the community of interest, specifically within the {domain} domain.
Text: {input_text}
RESPONSE:
{{<entity_categories>}}
"""  # noqa: E501
