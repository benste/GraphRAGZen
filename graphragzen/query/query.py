from typing import Optional

import networkx as nx
import pandas as pd
from graphragzen.prompts.default_prompts.local_search_prompts import LOCAL_SEARCH_PROMPT
from graphragzen.query import get_context
from graphragzen.text_embedding.embedding_models import BaseEmbedder
from graphragzen.text_embedding.vector_databases import VectorDatabase


class PromptBuilder:
    """
    A class used to build a prompt based on graph data, vector searches, and additional documents or
    cluster reports.

    Attributes:
        embedding_model (BaseEmbedder): An embedding model used to generate embeddings for text and
            queries.
        vector_db (VectorDatabase): A client for interacting with the vector database.
        graph (nx.Graph): A graph containing nodes and edges that represent entities and their
            relationships.
        source_documents (Optional[pd.DataFrame]): A DataFrame containing source documents for
            additional context.
        cluster_report (Optional[pd.DataFrame]): A DataFrame containing cluster reports for
            additional context.
    """

    def __init__(
        self,
        embedding_model: BaseEmbedder,
        vector_db: VectorDatabase,
        graph: nx.Graph,
        source_documents: Optional[pd.DataFrame] = None,
        cluster_report: Optional[pd.DataFrame] = None,
    ) -> None:
        """Initializes the PromptBuilder instance with the necessary components.

        Args:
            embedding_model (BaseEmbedder): An embedding model to generate embeddings for text and
                queries.
            vector_db (VectorDatabase):
            graph (nx.Graph): A graph containing nodes and edges that represent entities and their
                relationships.
            source_documents (pd.DataFrame, optional): A DataFrame containing source
                documents for additional context. Defaults to None.
            cluster_report (pd.DataFrame, optional): A DataFrame containing cluster reports
                for additional context. Should contain the columns 'cluster' (string identifier of
                each cluster), 'node_name' (lists of node names that belong to a cluster) and
                'description' (a string describing the cluster). Defaults to None.
        """

        self.embedding_model = embedding_model
        self.vector_db = vector_db
        self.graph = graph
        self.source_documents = source_documents
        self.cluster_report = cluster_report

    def build_prompt(
        self,
        query: str,
        prompt: str = LOCAL_SEARCH_PROMPT,
        score_threshold: float = 0.0,
        top_k_similar_entities: int = 10,
        top_k_inside_edges: int = 3,
        top_k_outside_edges: int = 3,
        top_k_cluster_descriptions: int = 3,
        top_k_source_documents: int = 3,
        source_documents_id_key: str = "chunk_id",
        source_documents_source_key: str = "chunk",
        source_documents_feature_delimiter: str = "\n",
    ) -> str:
        """
        Builds a prompt based on the query, graph data, vector searches, and additional documents or
        cluster reports.

        Args:
            query (str): The query for which a prompt is being constructed.
            prompt (str, optional): The base prompt template to be filled with the context data and
                query. Defaults to LOCAL_SEARCH_PROMPT.
            score_threshold (float, optional): The minimum score threshold for including vector
                search results. Entities with scores below this threshold are excluded. Defaults to
                0.0.
            top_k_similar_entities (int, optional): The number of top similar entities
                from the graph (retreived through vector search) to include in the prompt.
                Defaults to 10.
            top_k_inside_edges (int, optional): The number of inside edges by weight related to the
                similar entities to include in the prompt.
                An iside edge is defined as an edge whom's both nodes are already in the entities
                retrieved through vector search. Defaults to 3.
            top_k_outside_edges (int, optional): The top number of outside edges by weight related
                to the similar entities to include in the prompt.
                An outside edge is defined as an edge of whom's nodes extactly 1 is already in the
                entities retrieved through vector search. Defaults to 3.
            top_k_cluster_descriptions (int, optional): The number of top cluster descriptions by
                occurence, followed by rank, to include in the prompt. Defaults to 3.
            top_k_source_documents (int, optional): The number of top source documents by occurence
                to include in the prompt. Defaults to 3.
            source_documents_id_key (str): The column name in `source_documents` that represents
                unique document IDs. Defaults to "chunk_id".
            source_documents_source_key (str): The column name in `source_documents` containing the
                source text. Defaults to "chunk".
            source_documents_feature_delimiter (str): The delimiter used to split the "source_id"
                values in the Graph metadata. Default is newline.

        Returns:
            str: The final constructed prompt with context data and the original query formatted
            within the prompt template.
        """
        similar_entities = get_context.semantic_similar_entities(
            embedding_model=self.embedding_model,
            vector_db=self.vector_db,
            query=query,
            k=top_k_similar_entities,
            score_threshold=score_threshold,
        )

        similar_entities += get_context.extra_inside_edges(
            self.graph,
            similar_entities,
            k=top_k_inside_edges,
        )

        similar_entities += get_context.extra_outside_edges(
            self.graph, similar_entities, k=top_k_outside_edges
        )

        if self.source_documents is not None:
            source_texts = get_context.source_texts(
                self.source_documents,
                self.graph,
                similar_entities,
                k=top_k_source_documents,
                id_key=source_documents_id_key,
                source_key=source_documents_source_key,
                feature_delimiter=source_documents_feature_delimiter,
            )
        else:
            source_texts = None

        if self.cluster_report is not None:
            cluster_summaries = get_context.cluster_summaries(
                self.graph,
                self.cluster_report,
                similar_entities,
                k=top_k_cluster_descriptions,
            )
        else:
            cluster_summaries = None

        # BUILD PROMPT
        context_data = ""
        if source_texts is not None:
            source_table = "\n".join(
                [f"SOURCE {i+1}:\n{source}\n" for i, source in enumerate(source_texts)]
            )
            context_data += f"-----------SPECIFIC SOURCES-----------\n{source_table}"

        if cluster_summaries is not None:
            cluster_table = "\n".join(
                [
                    f"SOURCE {i+1}:\n{cs['title']}:\n{cs['summary']}\n"
                    for i, cs in enumerate(cluster_summaries)
                ]
            )
            context_data += f"-----------GLOBAL SOURCES-----------\n{cluster_table}"

        node_table = ""
        edge_table = ""
        for entity in similar_entities:
            if entity["entity_type"] == "node":
                node = self.graph.nodes[entity["entity_name"]]
                node_table += f"\n{entity['entity_name']} ({node['type']}): {node['description']}"
            elif entity["entity_type"] == "edge":
                edge = self.graph.edges[entity["entity_name"]]
                edge_table += f"\n{entity['entity_name'][0]} <-> {entity['entity_name'][1]}: {edge['description']}"  # noqa: E501

        context_data += f"-----------ENTITIES-----------\n{node_table}"
        context_data += f"-----------RELATIONSHIPS-----------\n{edge_table}"

        return prompt.format(context_data=context_data, query=query)
