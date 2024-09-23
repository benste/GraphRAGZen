# mypy: ignore-errors
# flake8: noqa
import os
from random import sample

from graphragzen import load_documents, preprocessing, prompt_tuning
from graphragzen.llm import OpenAICompatibleClient, Phi35MiniGGUF


def create_custom_prompts() -> str:
    """
    Use an LLM to generate a prompt for entity extraction comprises the following steps.
    1. Domain: We fist ask the LLM to create the domains that the documents span
    2. Persona: with the domains the LLM can create a persona (e.g. You are an expert {{role}}.
        You are skilled at {{relevant skills}})
    3. Entity categories: using the domain and persona we ask the LLM to extract from the documents
        the categories a node could get (e.g. person, school of thought, ML)
    4. Examples: Using all of the above we ask the LLM to create some example document->entities
        extracted
    5. Entity extraction prompt: We merge all of the above information in a prompt that can be
        used to extract entities

    Note: Each function's optional parameters have sane defaults. Check out their
    docstrings for their desrciptions and see if you want to overwrite any
    """
    # Load an LLM locally
    print("Loading LLM")
    llm = Phi35MiniGGUF(
        model_storage_path="/home/bens/projects/GraphRAGZen/models/Phi-3.5-mini-instruct-Q4_K_M.gguf",
        tokenizer_URI="microsoft/Phi-3.5-mini-instruct",
        context_size=32786,
        persistent_cache_file="./phi35_mini_instruct_persistent_cache.yaml",
    )

    # # Communicate with an LLM running on a server
    # llm = OpenAICompatibleClient(
    #     base_url = "http://localhost:8081",
    #     context_size = 32768,
    #     persistent_cache_file="./phi35_mini_instruct_persistent_cache.yaml"
    # )

    # Load raw documents
    print("Loading raw documents")
    raw_documents = load_documents.load_text_documents(
        raw_documents_folder="/home/bens/projects/GraphRAGZen/documents/sample-python-3.10.13-documentation"  # noqa: E501
    )

    # Split documents into chunks based on tokens
    print("Chunking documents")
    chunked_documents = preprocessing.chunk_documents(raw_documents, llm)

    # Let's not use all documents, that's not neccessary and too slow
    print("Sampling documents")
    chunks = chunked_documents.chunk.tolist()
    sampled_documents = sample(chunks, min([len(chunks), 15]))

    # Get the domain representing the documents
    print("Generating domain")
    domain = prompt_tuning.generate_domain(llm, sampled_documents)

    # Get the persona representing the documents
    print("Generating persona")
    persona = prompt_tuning.generate_persona(llm, domain)

    # Get the entity categories present the documents
    print("Generating entity categories")
    entity_categories = prompt_tuning.generate_entity_categories(
        llm, sampled_documents, domain, persona
    )

    # Generate some entity relationship examples
    print("Generating entity relationship examples")
    entity_relationship_examples = prompt_tuning.generate_entity_relationship_examples(
        llm, sampled_documents, persona, entity_categories, max_examples=3
    )

    # Create the actual entity extraction prompt
    print("Generating entity extraction prompt")
    entity_extraction_prompt = prompt_tuning.create_entity_extraction_prompt(
        llm, entity_categories, entity_relationship_examples
    )

    # Also create a prompt to summarize the descriptions of the entities
    print("Generating description summarization prompt")
    summarization_prompt = prompt_tuning.create_description_summarization_prompt(persona)

    return entity_extraction_prompt, summarization_prompt


## Uncomment and run the following to create custom prompts

# entity_extraction_prompt, summarization_prompt = create_custom_prompts()

# # Save the prompts
# outfol = "graphtest"

# if not os.path.isdir(outfol):
#     os.makedirs(outfol)

# with open(os.path.join(outfol, "Custom_Entity_Extraction_Prompt.txt"), "w") as text_file:
#     text_file.write(entity_extraction_prompt)

# with open(os.path.join(outfol, "Custom_Summarization_Prompt.txt"), "w") as text_file:
#     text_file.write(summarization_prompt)
