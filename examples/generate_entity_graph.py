# mypy: ignore-errors

import networkx as nx

from graphragzen.llm import load_gemma2_gguf
from graphragzen import preprocessing
from graphragzen import entity_extraction
from graphragzen import feature_merging
from graphragzen import clustering


def entity_graph_pipeline() -> nx.Graph:
    # Note: Each function's optional parameters have sane defaults. Check out their
    # docstrings for their desrciptions and see if you want to overwrite any

    # Load an LLM locally
    print("Loading LLM")
    llm = load_gemma2_gguf(
        model_storage_path="/home/bens/projects/DemystifyGraphRAG/models/gemma-2-2b-it-Q4_K_M.gguf",
        tokenizer_URI="google/gemma-2-2b-it",
    )

    # Load raw documents
    print("Loading raw documents")
    raw_documents = preprocessing.load_text_documents(
        raw_documents_folder="/home/bens/projects/DemystifyGraphRAG/data/01_raw/machine_learning_intro"
    )

    # Split documents into chunks based on tokens
    print("Chunking documents")
    chunked_documents = preprocessing.chunk_documents(
        raw_documents,
        llm,
        window_size=400,
    )

    # Extract entities from the chunks
    print("Extracting raw entities")
    prompt_config = entity_extraction.EntityExtractionPromptConfig() # default prompt
    raw_entities = entity_extraction.extract_raw_entities(
        chunked_documents, llm, prompt_config, max_gleans=3
    )

    # Create a graph from the raw extracted entities
    print("Creating graph from raw entities")
    entity_graph = entity_extraction.raw_entities_to_graph(raw_entities, prompt_config.formatting)

    # Each node and edge could be found multiple times in the documents and thus have
    # multiple descriptions. We'll summarize these into one description per node and edge
    print("Summarizing entity descriptions")
    prompt_config = feature_merging.MergeFeaturesPromptConfig() # default prompt
    entity_graph = feature_merging.merge_graph_features(
        entity_graph, llm, prompt_config, feature="description", how="LLM"
    )

    # Let's clusted the nodes and assign the cluster ID as a property to each node
    print("Clustering graph")
    entity_graph = clustering.leiden(entity_graph, max_comm_size=10)

    print("Pipeline finished successful \n\n")
    return entity_graph


entity_graph = entity_graph_pipeline()
