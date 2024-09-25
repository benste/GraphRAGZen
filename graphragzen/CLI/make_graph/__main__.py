from random import sample
import argparse
import os
import networkx as nx
from graphragzen import (
    clustering,
    entity_extraction,
    feature_merging,
    load_documents,
    preprocessing,
    text_embedding,
    prompt_tuning,
)
from graphragzen.llm import OpenAICompatibleClient, Phi35MiniGGUF, Gemma2GGUF, BaseLlamaCpp

from huggingface_hub import hf_hub_download

if __name__=="__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Create a knowledge graph from documents using the GraphRAGZen library",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
    parser.add_argument("--documents_folder", type=str, required=True, help="REQUIRED - Folder containing raw text documents")
    parser.add_argument("--project_folder", type=str, required=True, help="REQUIRED - Folder to save the output data")
    parser.add_argument("--llm_context_size", type=int, required=True, help="REQUIRED - Context size of the LLM", default=32768)
    parser.add_argument("--tokenizer_uri", type=str, required=True, help="REQUIRED - HuggingFace URI to the LLM tokenizer", default="microsoft/Phi-3.5-mini-instruct")
    parser.add_argument("--llm_model_name", type=str, help="Name of the LLM from OpenAI or on the custom server to call")
    parser.add_argument("--api_uri", type=str, help="URI to an OpenAI compatible API endpoint of a server running an LLM")
    parser.add_argument("--openai_key_env_variable", type=str, help="ENV variable that stores the OpenAI API key. Only used of `api_uri` is not provided.")
    parser.add_argument("--gguf_model_path", type=str, help="Path to local GGUF model. Only used of `api_uri` and `openai_key_env_variable` are not provided.")
    parser.add_argument("--gguf_model_repo", type=str, help="HuggingFace repo containing GGUF models. Only used of `api_uri`, `openai_key_env_variable` and `gguf_model_path` are not provided.", default="bartowski/Phi-3.5-mini-instruct-GGUF")
    parser.add_argument("--gguf_model_filename", type=str, help="Filename of the exact GGUF file in the HuggingFace repo. Only used of `api_uri`, `openai_key_env_variable` and `gguf_model_path` are not provided", default="Phi-3.5-mini-instruct-Q4_K_M.gguf")
    
    args = parser.parse_args()

    # Load LLM client for custom OpenAI compatible API endpoint
    print("Initializing LLM")
    if args.api_uri:
        if not args.llm_model_name:
            raise Exception("API_URI provided but not llm_model_name. I need to know which LLM model on the server to call")
        
        llm = OpenAICompatibleClient(
            base_url=args.api_uri,required=True,
            model_name=args.llm_model_name,
            context_size=args.llm_context_size,
            hf_tokenizer_URI=args.tokenizer_uri,
            persistent_cache_file=f"./{args.api_uri}:{args.llm_model_name}_persistent_cache.yaml"
        )
     
    # Load LLM client to talk to OpenAI   
    elif args.openai_key_env_variable:
        if not args.model_name:
            raise Exception("openai_key_env_variable provided but not model_name. I need to know which OpenAI model to call")
        
        llm = OpenAICompatibleClient(
            api_key_env_variable = args.openai_key_env_variable,
            model_name=args.model_name,
            context_size=args.llm_context_size,
            use_cache=True,
            cache_persistent=True,
            persistent_cache_file=f"./OpenAI:{args.model_name}_persistent_cache.yaml"
        )
        
    
    else:
        # Load GGUF model locally
        if args.gguf_model_repo and args.gguf_model_filename and not args.gguf_model_path:
            # Download model
            args.gguf_model_path = hf_hub_download(repo_id=args.gguf_model_repo, filename=args.gguf_model_filename)
        
        if args.gguf_model_path:
            if "phi-3.5" in args.gguf_model_path.lower():
                loader_class = Phi35MiniGGUF
            elif "gemma-2" in args.gguf_model_path.lower():
                loader_class = Gemma2GGUF
            else:
                loader_class = BaseLlamaCpp
        
            llm = loader_class(
                model_storage_path=args.gguf_model_path,
                tokenizer_URI=args.tokenizer_uri,
                context_size=args.llm_context_size,
                persistent_cache_file=f"./{os.path.split(args.gguf_model_path)[-1]}_persistent_cache.yaml",
            )

    # Load text embedder
    embedder = text_embedding.NomicTextEmbedder(huggingface_URI="nomic-ai/nomic-embed-text-v1.5")

    # Create vector DB
    print("Loading vector database")
    vector_db = text_embedding.QdrantLocalVectorDatabase(vector_size=768)

    # Load raw documents
    print("Loading raw documents")
    raw_documents = load_documents.load_text_documents(
        raw_documents_folder=args.documents_folder
    )
    
    # Split documents into chunks based on tokens
    print("Chunking documents")
    chunked_documents = preprocessing.chunk_documents(
        raw_documents,
        llm,
    )
    
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

    # Extract entities from the chunks
    print("Extracting raw entities")
    custom_prompts = entity_extraction.EntityExtractionPrompts(
            entity_extraction_prompt=entity_extraction_prompt
        )
    prompt_config = entity_extraction.EntityExtractionPromptConfig(prompt=custom_prompts)
    raw_entities = entity_extraction.extract_raw_entities(
        chunked_documents, llm, max_gleans=1, prompt_config=prompt_config,
    )

    # Create a graph from the raw extracted entities
    print("Creating graph from raw entities")
    entity_graph = entity_extraction.raw_entities_to_graph(raw_entities)

    # Each node and edge could be found multiple times in the documents and thus have
    # multiple descriptions. We'll summarize these into one description per node and edge
    print("Summarizing entity descriptions")
    prompt_config = feature_merging.MergeFeaturesPromptConfig(
            prompt=summarization_prompt
        )
    entity_graph = feature_merging.merge_graph_features(
        entity_graph, llm, feature="description", how="LLM", prompt=prompt_config
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

    # Save outputs
    if not os.path.isdir(args.project_folder):
        os.makedirs(args.project_folder)

    chunked_documents.to_pickle(os.path.join(args.project_folder, "source_documents.pkl"))
    nx.write_graphml(entity_graph, os.path.join(args.project_folder, "entity_graph.graphml"))
    cluster_report.to_pickle(os.path.join(args.project_folder, "cluster_report.pkl"))
    vector_db.save(os.path.join(args.project_folder, "vector_db"))

    print("Knowledge graph and related data have been saved successfully.")

