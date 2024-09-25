import os
import argparse

import networkx as nx
import pandas as pd
from graphragzen import text_embedding
from graphragzen.llm import OpenAICompatibleClient, Phi35MiniGGUF
from graphragzen.prompts.default_prompts.local_search_prompts import LOCAL_SEARCH_PROMPT
from graphragzen.query.query import PromptBuilder

from graphragzen.llm import OpenAICompatibleClient, Phi35MiniGGUF, Gemma2GGUF, BaseLlamaCpp

from huggingface_hub import hf_hub_download

if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description="Query a knowledge graph create by the GraphRAGZen library",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
    parser.add_argument("--project_folder", type=str, required=True, help="REQUIRED - Folder where the results from `make_graph` are saved")
    parser.add_argument("--llm_context_size", type=int, required=True, help="REQUIRED - Context size of the LLM", default=32768)
    parser.add_argument("--tokenizer_uri", type=str, required=True, help="REQUIRED - HuggingFace URI to the LLM tokenizer", default="microsoft/Phi-3.5-mini-instruct")
    parser.add_argument("--llm_model_name", type=str, help="Name of the LLM from OpenAI or the LLM running on a custom server to call")
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
            base_url=args.api_uri,
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

    # Load vector DB already populated with vector embedding of the graph entity descriptions
    print("Loading Vector DB")
    vector_db = text_embedding.vector_databases.QdrantLocalVectorDatabase(
        database_location=os.path.join(args.project_folder, "vector_db")
    )
    
    # Load graph
    print("Loading Knowdlege Graph")
    graph = nx.read_graphml(os.path.join(args.project_folder, "entity_graph.graphml"))

    # Source documents can be added as additional context during qierying
    print("Loading source documents")
    source_doc_file = os.path.join(args.project_folder, "source_documents.pkl")
    if os.path.isfile(source_doc_file):
        source_documents = pd.read_pickle(source_doc_file)
    else:
        source_documents = None

    # Load cluster report
    print("Loading Cluster Report")
    cluster_report_file = os.path.join(args.project_folder, "cluster_report.pkl")
    if os.path.isfile(cluster_report_file):
        cluster_report = pd.read_pickle(cluster_report_file)
    else:
        cluster_report = None
        
    # Prompt builder initialized once and used in subsequent queries
    prompt_builder = PromptBuilder(
        embedding_model=embedder,
        vector_db=vector_db,
        graph=graph,
        source_documents=source_documents,
        cluster_report=cluster_report,
    )

    # Ask for user query
    while True:
        user_input = input("Input your query (type 'stop' to exit)\n")
        
        if user_input.strip("'").strip('"').lower() == 'stop':
            break
        
        prompt = prompt_builder.build_prompt(user_input)

        chat = llm.format_chat([("user", prompt)])
        response = llm.run_chat(chat=chat)
        print(f"{response}\n\n")
