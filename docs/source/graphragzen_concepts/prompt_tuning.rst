.. role:: raw-html(raw)
    :format: html

.. _prompt_tuning_explanation_label:

Prompt tuning
-------------

**GraphRAGZen** comes with default prompts to extract entities from documents, summarize features, etc.

Although these can be used out-of-the-box to extract entities from documents and create a graph, 
they are not tuned to the documents. 
:raw-html:`<br />`
Higher quality graphs could be obtained by making
these prompts more relevant to the domain of the documents. 

Your can provided you own custom prompts, but **GraphRAGZen** does come with functions to create these prompts from a sample of your documents.

These functions rely on their own prompts that ask the LLM to look at the documents and:

- Create a domain for the documents
- Create a persona that is an expert in the create domain
- Define which entity types are present in the documents (e.g. person, location, school of thought)
- Create some `document->entities extracted` examples

All of this information is then merged into a prompt that can be used to extract entities.

A similar method is used to create a description summarization prompt.

The default prompts: :ref:`default_prompts_label`

The prompts used to create new prompts: :ref:`prompt_tuning_prompts_label`

See this example on how to create prompts specific to the domain of your documents (click on source)
:func:`graphragzen.examples.autotune_custom_prompts.create_custom_prompts`