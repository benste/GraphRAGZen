# mypy: ignore-errors
# flake8: noqa
import os

import networkx as nx
from graphragzen import (
    clustering,
    entity_extraction,
    load_documents,
    merge,
    preprocessing,
    text_embedding,
)
from graphragzen.llm import BaseLlamaCpp, OpenAICompatibleClient, Phi35MiniGGUF


def entity_graph_pipeline(
    custom_entity_extraction_prompt: str = None,
    custom_summarization_prompt: str = None,
) -> nx.Graph:
    # Note: Each function's optional args have sane defaults. Check out their
    # docstrings for their descriptions and see if you want to overwrite any

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
    #     base_url="http://localhost:8081",
    #     context_size=32768,
    #     persistent_cache_file="./v6-Finch_7B_persistent_cache.yaml",
    # )

    # Load text embedder
    embedder = text_embedding.NomicTextEmbedder(huggingface_URI="nomic-ai/nomic-embed-text-v1.5")

    # Create vector DB (nomic-embed-text-v1.5 creates vectors of size 768)
    print("Loading vector database")
    vector_db = text_embedding.QdrantLocalVectorDatabase(vector_size=embedder.vector_size)

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
    if custom_entity_extraction_prompt:
        # Custom prompt if available
        custom_prompts = entity_extraction.EntityExtractionPrompts(
            entity_extraction_prompt=custom_entity_extraction_prompt
        )
        prompt_config = entity_extraction.EntityExtractionPromptConfig(prompt=custom_prompts)
    else:
        # Default prompt
        prompt_config = entity_extraction.EntityExtractionPromptConfig()

    raw_entities = entity_extraction.extract_raw_entities(
        chunked_documents, llm, max_gleans=1, prompt_config=prompt_config
    )

    # Create a graph from the raw extracted entities
    print("Creating graph from raw entities")
    entity_graph = entity_extraction.raw_entities_to_graph(raw_entities)

    # Merge nodes that are very similar to eachother (e.g. 'Pierce Brosnan' and 'pierce_brosnan')
    entity_graph, _ = merge.merge_similar_graph_nodes(entity_graph, embedder)

    # Each node and edge could be found multiple times in the documents and thus have
    # multiple descriptions. We'll summarize these into one description per node and edge
    print("Summarizing entity descriptions")
    if custom_summarization_prompt:
        # Custom prompt if available
        prompt_config = merge.MergeFeaturesPromptConfig(prompt=custom_summarization_prompt)
    else:
        # default prompt
        prompt_config = merge.MergeFeaturesPromptConfig()

    entity_graph = merge.merge_graph_features(
        entity_graph, llm, prompt=prompt_config, feature="description", how="LLM"
    )

    # Let's cluster the nodes and assign the cluster ID as a property to each node
    print("Clustering graph")
    entity_graph, cluster_entity_map = clustering.leiden(
        entity_graph,
        max_comm_size=20,
        min_comm_size=5,
        levels=1,
    )

    # Describe each cluster, creating a so-called cluster report
    print("Describing clusters")
    cluster_report = clustering.describe_clusters(llm, entity_graph, cluster_entity_map)

    # Embed the descriptions of each node and edge
    print("Embedding entity descriptions")
    _ = text_embedding.embed_graph_features(
        entity_graph, embedder, vector_db=vector_db, features_to_embed=["description"]
    )

    print("Pipeline finished successful \n\n")
    return (
        chunked_documents,
        entity_graph,
        cluster_entity_map,
        cluster_report,
        embedder,
        vector_db,
    )


# # Uncomment and run the following to create a knowledge graph
# outfol = "graphtest2"

# Load custom prompts if available
# with open(os.path.join(outfol, "Custom_Entity_Extraction_Prompt.txt"), "r") as text_file:
#     entity_extraction_prompt = text_file.read()

# with open(os.path.join(outfol, "Custom_Summarization_Prompt.txt"), "r") as text_file:
#     summarization_prompt = text_file.read()

# # Extract entities
# chunked_documents, entity_graph, cluster_report, vector_db = entity_graph_pipeline(
#     entity_extraction_prompt, summarization_prompt
# )

# # Or use with default prompts
# chunked_documents, entity_graph, cluster_report, vector_db = entity_graph_pipeline()

# # Save everything we need for querying
# if not os.path.isdir(outfol):
#     os.makedirs(outfol)

# chunked_documents.to_pickle(os.path.join(outfol, "source_documents.pkl"))
# nx.write_graphml(entity_graph, os.path.join(outfol, "entity_graph.graphml"))
# cluster_report.to_pickle(os.path.join(outfol, "cluster_report.pkl"))
# vector_db.save(os.path.join(outfol, "vector_db"))
