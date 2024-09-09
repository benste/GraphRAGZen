# mypy: ignore-errors
# flake8: noqa
import os

import networkx as nx
import pandas as pd
from graphragzen import text_embedding
from graphragzen.llm import load_openAI_API_client, load_phi35_mini_gguf  # noqa: F401
from graphragzen.prompts.default_prompts.local_search_prompts import LOCAL_SEARCH_PROMPT
from graphragzen.query.query import PromptBuilder


def question() -> None:
    # This is just here for sphinx to find the file when making documentation
    print("Do you know the difference between objects and values in python?")


# # Uncomment and run the following to run a query against your knowledge graph

# graph_output_folder = "graphtest"

# # Load an LLM locally
# print("Loading LLM")
# llm = load_phi35_mini_gguf(
#     model_storage_path="/home/bens/projects/GraphRAGZen/models/Phi-3.5-mini-instruct-Q4_K_M.gguf",  # noqa: E501
#     tokenizer_URI="microsoft/Phi-3.5-mini-instruct",
#     persistent_cache_file="./phi35_mini_instruct_persistent_cache.yaml",
#     context_size=32786,
# )

# # # Communicate with an LLM running on a server
# # llm = load_openAI_API_client(
# #     base_url = "http://localhost:8081",
# #     context_size = 32768,
# #     use_cache=True,
# #     persistent_cache_file="./phi35_mini_instruct_persistent_cache.yaml"
# # )


# # Load text embedder
# embedder = text_embedding.load_nomic_embed_text(huggingface_URI="nomic-ai/nomic-embed-text-v1.5")

# # Load vector DB already populated with vector embedding of the graph entity descriptions
# print("Loading Vector DB")
# vector_db_client = text_embedding.load_vector_db(os.path.join(graph_output_folder, "vector_db"))

# # Source documents can be added as additional context during qierying
# print("Loading source documents")
# source_documents = pd.read_pickle(os.path.join(graph_output_folder, "source_documents.pkl"))

# # Load graph
# print("Loading Knowdlege Graph")
# graph = nx.read_graphml(os.path.join(graph_output_folder, "entity_graph.graphml"))

# # Load cluster report
# print("Loading Cluster Report")
# cluster_report = pd.read_pickle(os.path.join(graph_output_folder, "cluster_report.pkl"))

# # Prompt builder initialized once and used in subsequent queries
# prompt_builder = PromptBuilder(
#     embedding_model=embedder,
#     vector_db_client_or_location=vector_db_client,
#     graph=graph,
#     source_documents=source_documents,
#     cluster_report=cluster_report,
# )

# # Build prompt and query
# print("Building prompt")
# prompt = prompt_builder.build_prompt(
#     query="What is the difference between objects and values in python?",
# )

# chat = llm.format_chat([("user", prompt)])
# response = llm.run_chat(chat=chat)
# print(response)
