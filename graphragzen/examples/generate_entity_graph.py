# mypy: ignore-errors

import networkx as nx
from graphragzen import (
    clustering,
    entity_extraction,
    feature_merging,
    load_documents,
    preprocessing,
    text_embedding,
)
from graphragzen.llm import load_openAI_API_client, load_phi35_mini_gguf  # noqa: F401


def entity_graph_pipeline() -> nx.Graph:
    # Note: Each function's optional kwargs have sane defaults. Check out their
    # docstrings for their descriptions and see if you want to overwrite any

    # Load an LLM locally
    print("Loading LLM")
    llm = load_phi35_mini_gguf(
        model_storage_path="/home/bens/projects/GraphRAGZen/models/Phi-3.5-mini-instruct-Q4_K_M.gguf",  # noqa: E501
        tokenizer_URI="microsoft/Phi-3.5-mini-instruct",
        persistent_cache_file="./phi35_mini_persistent_cache.yaml",
        context_size=32786,
    )

    # # Communicate with an LLM running on a server
    # llm = load_openAI_API_client(
    #     base_url = "http://localhost:8081",
    #     context_size = 32768,
    #     use_cache=True,
    #     persistent_cache_file="./phi35_mini_persistent_cache.yaml"
    # )

    # Load text embedder
    embedder = text_embedding.load_nomic_embed_text(
        huggingface_URI="nomic-ai/nomic-embed-text-v1.5"
    )

    # Create vector DB (nomic-embed-text-v1.5 creates vectors of size 768)
    vector_db_client = text_embedding.create_vector_db(vector_size=768)

    # Load raw documents
    print("Loading raw documents")
    raw_documents = load_documents.load_text_documents(
        raw_documents_folder="/home/bens/projects/GraphRAGZen/documents/sample-python-3.10.13-documentation"  # noqa: E501
    )

    # Split documents into chunks based on tokens
    print("Chunking documents")
    chunked_documents = preprocessing.chunk_documents(
        raw_documents,
        llm,
    )

    # Extract entities from the chunks
    print("Extracting raw entities")
    raw_entities = entity_extraction.extract_raw_entities(chunked_documents, llm, max_gleans=3)

    # Create a graph from the raw extracted entities
    print("Creating graph from raw entities")
    entity_graph = entity_extraction.raw_entities_to_graph(raw_entities)

    # Each node and edge could be found multiple times in the documents and thus have
    # multiple descriptions. We'll summarize these into one description per node and edge
    print("Summarizing entity descriptions")
    prompt_config = feature_merging.MergeFeaturesPromptConfig()  # default prompt
    entity_graph = feature_merging.merge_graph_features(
        entity_graph, llm, prompt_config, feature="description", how="LLM"
    )

    # Let's clusted the nodes and assign the cluster ID as a property to each node
    print("Clustering graph")
    entity_graph, cluster_entity_map = clustering.leiden(entity_graph, max_comm_size=10)

    # Describe each cluster, creating a so-called cluster report
    cluster_report = clustering.describe_clusters(llm, entity_graph, cluster_entity_map)

    # Embed the descriptions of each node and edge
    embedded_features = text_embedding.embed_graph_features(
        entity_graph, embedder, vector_db_client=vector_db_client, features_to_embed=["description"]
    )

    print("Pipeline finished successful \n\n")
    return entity_graph, cluster_entity_map, cluster_report, embedded_features
