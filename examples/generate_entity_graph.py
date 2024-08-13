# mypy: ignore-errors

import networkx as nx

from graphragzen.llm import load_gemma2_gguf
from graphragzen import preprocessing
from graphragzen import entity_extraction
from graphragzen import description_summarization
from graphragzen import clustering


def entity_graph_pipeline() -> nx.Graph:
    # Note: Each config has sane defaults but can be overwritten as seen fit at instantiation of
    # the config

    # Load an LLM locally
    print("Loading LLM")
    llm = load_gemma2_gguf(
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
    chunk_config = preprocessing.ChunkConfig(
        window_size=400,
        overlap=100,
    )
    chunked_documents = preprocessing.chunk_documents(raw_documents, llm, config=chunk_config)

    # Extract entities from the chunks
    print("Extracting raw entities")
    prompt_config = entity_extraction.EntityExtractionPromptConfig()
    entity_extraction_config = entity_extraction.EntityExtractionConfig(max_gleans=3)
    raw_entities = entity_extraction.extract_raw_entities(
        chunked_documents, llm, prompt_config, config=entity_extraction_config
    )

    # Create a graph from the raw extracted entities
    print("Creating graph from raw entities")
    entity_to_graph_config = (
        entity_extraction.RawEntitiesToGraphConfig()
    )  # let's use default parameters
    entity_graph = entity_extraction.raw_entities_to_graph(
        raw_entities, prompt_config.formatting, config=entity_to_graph_config
    )

    # Each node could be found multiple times in the documents and thus have multiple descriptions.
    # We'll summarize these into one description per node and edge
    print("Summarizing entity descriptions")
    prompt_config = description_summarization.DescriptionSummarizationPromptConfig()
    summarization_config = (
        description_summarization.DescriptionSummarizationConfig()
    )  # let's use default parameters
    entity_graph = description_summarization.summarize_graph_descriptions(
        entity_graph, llm, prompt_config, config=summarization_config
    )

    # Let's clusted the nodes and assign the cluster ID as a property to each node
    print("Clustering graph")
    entity_graph = clustering.leiden(entity_graph, max_comm_size=10)

    print("Pipeline finished successful \n\n")
    return entity_graph


entity_graph = entity_graph_pipeline()
