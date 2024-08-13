# mypy: ignore-errors

import networkx as nx

from graphragzen.llm import load_llm
from graphragzen import preprocessing
from graphragzen import entity_extraction
from graphragzen import description_summarization
from graphragzen import typing
from graphragzen import clustering


def entity_extraction_pipeline() -> nx.Graph:
    # Note: Each config loaded from `typing` has sane defaults but can be overwritten as seen fit

    # Load an LLM locally
    print("Loading LLM")
    llm = load_llm.load_gemma2_gguf(
        model_storage_path="/home/bens/projects/DemystifyGraphRAG/models/gemma-2-2b-it-Q4_K_M.gguf",
        tokenizer_URI="google/gemma-2-2b-it",
    )

    # Load raw documents
    print("Loading raw documents")
    raw_documents = preprocessing.load_text_documents(
        raw_documents_folder="/home/bens/projects/DemystifyGraphRAG/data/01_raw/machine_learning_intro"  # noqa: E501
    )

    # Split documents into chunks based on tokens
    print("Chunking documents")
    chunk_config = typing.ChunkConfig()  # let's use default parameters
    chunked_documents = preprocessing.chunk_documents(raw_documents, llm, config=chunk_config)

    # Extract entities from the chunks
    print("Extracting raw entities")
    prompt_config = typing.EntityExtractionPromptConfig()
    entity_extraction_config = typing.EntityExtractionConfig()  # let's use default parameters
    raw_entities = entity_extraction.raw_entity_extraction(
        chunked_documents, llm, prompt_config, config=entity_extraction_config
    )

    # Create a graph from the raw extracted entities
    print("Creating graph from raw entities")
    entity_to_graph_config = typing.RawEntitiesToGraphConfig()  # let's use default parameters
    entity_graph = entity_extraction.raw_entities_to_graph(
        raw_entities, prompt_config.formatting, config=entity_to_graph_config
    )

    # Each node could be found multiple times in the documents and thus have multiple descriptions.
    # We'll summarize these into one description per node and edge
    print("Summarizing entity descriptions")
    prompt_config = typing.DescriptionSummarizationPromptConfig()
    summarization_config = typing.DescriptionSummarizationConfig()  # let's use default parameters
    entity_graph = description_summarization.summarize_descriptions(
        entity_graph, llm, prompt_config, config=summarization_config
    )

    # Let's clusted the nodes and assign the cluster ID as a property to each node
    print("Clustering graph")
    entity_graph = clustering.leiden(entity_graph, max_comm_size=10)

    print("Pipeline finished successful \n\n")
    return entity_graph


entity_graph = entity_extraction_pipeline()
1 + 1
# def create_entity_extraction_prompt():
#     """
#     Create a prompt for entity extraction.
#     1. Domain: We fist ask the LLM to create the domains that the documents span
#     2. Persona: with the domains the LLM can create a persona (e.g. You are an expert {{role}}.
#           You are skilled at {{relevant skills}})
#     3. Entity types: using the domain and persona we ask the LLM to extract from the documents
#           the types of entities a node could get (e.g. person, school of thought, ML)
#     4. Examples: Using all of the above we ask the LLM to create some example document->entities
#           extracted
#     5. Entity extraction prompt: We merge all of the above information in a prompt that can be
#           used to extract entities
#     """
#     # TODO
#     pass
