# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Fine-tuning prompts for entity relationship generation."""

ENTITY_RELATIONSHIPS_GENERATION_PROMPT = """
You are an intelligent assistant that helps a human to analyze the information in a text document and extract a knowledg graph from it. A knowledge graph consists of nodes and their edges (relationships between the nodes in the graph).

-Goal-
Given a text document that is potentially relevant to this activity and a list of categories, identify all nodes of those categories from the text and all edges among the identified nodes.

-Steps-
1. Identify all nodes. For each identified node, extract the following information:
- name: Name of the node, capitalized
- category: One of the following categories: [{entity_categories}]
- description: Comprehensive description of the node's attributes and activities
Format each node as a JSON with the following format:
{{"type": "node", "name": <name>, "category": <category>, "description": <description>}}
for example: {{"type": "node", "name": "Microsoft", "category": "organization", "description": "Microsoft is a technology company"}}

2. From the nodes identified in step 1, identify all pairs of (source_node, target_node) that are *clearly related* to each other.
For each edge, extract the following information:
- source: name of the source node, as identified in step 1
- target: name of the target node, as identified in step 1
- description: explanation as to why you think the source node and the target node are related to each other
- weight: a numeric score indicating strength of the edge between the source node and target node
Format each edge as a JSON with the following format:
{{"type": "edge", "source": <source>, "target": <target>, "description": <description>, "weight": <weight>}}
for eaxmple: {{"type": "edge", "source": "company A", "target": "person A", "description": "company A is currently owned by person A", "weight": 8}}

3. Return output in English as a single list of all JSON entities and relationships identified in steps 1 and 2.

-Real Data-
######################
entity_categories: {entity_categories}
text: {input_text}
######################
output:
"""  # noqa: E501

EXAMPLE_RELATION_TEMPLATE = """
Example {n}:

entity_categories: [{entity_categories}]
text:
{input_text}
------------------------
output:
{output}
#############################

"""  # noqa: E501
