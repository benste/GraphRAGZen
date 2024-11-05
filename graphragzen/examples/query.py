# mypy: ignore-errors
# flake8: noqa
import os
from uuid import uuid4

import networkx as nx
import pandas as pd
from graphragzen import text_embedding
from graphragzen.llm import OpenAICompatibleClient, Phi35MiniGGUF
from graphragzen.query.query import GraphRAGPromptBuilder, RAGPromptBuilder


def question() -> None:
    # This is just here for sphinx to find the file when making documentation
    print("Do you know the difference between objects and values in python?")


# # Uncomment and run the following to run a query using regular RAG and against your knowledge graph
# graph_output_folder = "graphtest"

# # Load an LLM locally
# print("Loading LLM")
# llm = Phi35MiniGGUF(
#     model_storage_path="/home/bens/projects/GraphRAGZen/models/Phi-3.5-mini-instruct-Q4_K_M.gguf",
#     tokenizer_URI="microsoft/Phi-3.5-mini-instruct",
#     context_size=32786,
#     persistent_cache_file="./phi35_mini_instruct_persistent_cache.yaml",
# )

# # # Communicate with an LLM running on a server
# # llm = OpenAICompatibleClient(
# #     base_url = "http://localhost:8081",
# #     context_size = 32768,
# #     persistent_cache_file="./phi35_mini_instruct_persistent_cache.yaml"
# # )


# # Load text embedder
# embedder = text_embedding.NomicTextEmbedder(huggingface_URI="nomic-ai/nomic-embed-text-v1.5")

# # Load vector DB already populated with vector embedding of the graph entity descriptions
# print("Loading Vector DB")
# vector_db = text_embedding.vector_databases.QdrantLocalVectorDatabase(
#     database_location=os.path.join(graph_output_folder, "vector_db")
# )

# # Source documents can be added as additional context during querying
# print("Loading source documents")
# source_documents = pd.read_pickle(os.path.join(graph_output_folder, "source_documents.pkl"))

# # Load graph
# print("Loading Knowdlege Graph")
# graph = nx.read_graphml(os.path.join(graph_output_folder, "entity_graph.graphml"))

# # Load cluster report
# print("Loading Cluster Report")
# cluster_report = pd.read_pickle(os.path.join(graph_output_folder, "cluster_report.pkl"))

# # GraphRAG Prompt builder initialized once and used in subsequent queries
# grahrag_prompt_builder = GraphRAGPromptBuilder(
#     embedding_model=embedder,
#     vector_db=vector_db,
#     graph=graph,
#     source_documents=source_documents,
#     cluster_report=cluster_report,
# )

# # Normal RAG Prompt builder initialized once and used in subsequent queries
# rag_prompt_builder = RAGPromptBuilder(
#     embedding_model=embedder,
#     vector_db=vector_db,
# )

# # Build prompt and query the LLM
# query = "What is the difference between objects and values in python?"

# print("Building GraphRAG prompt")
# graphrag_prompt = grahrag_prompt_builder.build_prompt(query)
# print("Querying LLM using GraphRAG prompt")
# chat = llm.format_chat([("user", graphrag_prompt)])
# graphrag_response = llm.run_chat(chat=chat)

# print("Building RAG prompt")
# rag_prompt = rag_prompt_builder.build_prompt(query)
# print("Querying LLM using RAG prompt")
# chat = llm.format_chat([("user", rag_prompt)])
# rag_response = llm.run_chat(chat=chat)

# print(f"GRAPHRAG RESPONSE:\n{graphrag_response}")
# print(f"\n\n\nRAG RESPONSE:\n{rag_response}")
