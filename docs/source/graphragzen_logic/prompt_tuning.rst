Prompt tuning
-------------

**GraphRAGZen** comes with default prompts to extract entities from documents, summarize features, etc.

Although these can be used out-of-the-box to extract entities from documents and create a graph, 
they are not tuned to the documents. Higher quality graphs could be obtained by making
these prompts more relevant to the domain of the documents. 

Your own prompts can be provided, but GraphRAGZen comes with functions to create these prompts.

These functions rely on their own prompts, that ask the LLM to look at some documents and

- Create a domain for the document
- Create a persona that is an expert in the create domain
- Define which entity types are present in the documents (e.g. person, location, school of thought)
- Create some `document->entities extracted` examples

All of this information is then merged into a prompt that can be used to extract entities.

A similar method is used to create a description summarization prompt.

The default prompts: :ref:`default_prompts_label`

The prompts used to create new prompts: :ref:`prompt_tuning_prompts_label`