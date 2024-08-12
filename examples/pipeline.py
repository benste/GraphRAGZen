from graphragzen.llm import load_llm
from graphragzen import preprocessing 
from graphragzen import entity_extraction
from graphragzen import description_summarization
from graphragzen import typing
from graphragzen import clustering

def entity_extraction_pipeline():
    # Note: Each config loaded from `typing` has sane defaults but can be overwritten as seen fit
    
    # Load an LLM locally
    llm_load_config = typing.LlmLoadingConfig(
        model_storage_path = '/home/bens/projects/DemystifyGraphRAG/models/gemma-2-2b-it-Q4_K_M.gguf',
        tokenizer_URI = 'google/gemma-2-2b-it',
    )
    llm = load_llm.load_gemma2_gguf(llm_load_config)
    
    # Load raw documents
    raw_documents_config = typing.RawDocumentsConfig(
        raw_documents_folder = '/home/bens/projects/DemystifyGraphRAG/data/01_raw/machine_learning_intro'
    )
    raw_documents = preprocessing.raw_documents(raw_documents_config)
    
    # Split documents into chunks based on tokens
    chunk_config = typing.ChunkConfig()
    chunked_documents = preprocessing.chunk_documents(raw_documents, llm, chunk_config)
    
    # Let the LLM extract entities (let's use default values)
    prompt_config = typing.EntityExtractionPromptConfig()
    entity_extraction_config = typing.EntityExtractionConfig()
    raw_entities = entity_extraction.raw_entity_extraction(chunked_documents, llm, prompt_config, entity_extraction_config)
    
    # Create a graph from the raw extracted entities
    entity_to_graph_config = typing.RawEntitiesToGraphConfig()
    entity_graph = entity_extraction.raw_entities_to_graph(raw_entities, prompt_config.formatting, entity_to_graph_config)
    
    # Each node could be found multiple times in the documents and thus have multiple descriptions. Let's make this the one description
    # by creating a summary of all descriptions per node and edge
    prompt_config = typing.DescriptionSummarizationPromptConfig()
    summarization_config = typing.DescriptionSummarizationConfig()
    entity_graph = description_summarization.summarize_descriptions(entity_graph, llm, prompt_config, summarization_config)
    
    # Let's clusted the nodes and assign the cluster ID as a property to each node
    cluster_config = typing.ClusterConfig()
    entity_graph = clustering.leiden(entity_graph, cluster_config)
    
    return entity_graph


# def create_entity_extraction_prompt():
#     """
#     Create a prompt for entity extraction.
#     1. Domain: We fist ask the LLM to create the domains that the documents span
#     2. Persona: with the domains the LLM can create a persona (e.g. You are an expert {{role}}. You are skilled at {{relevant skills}})
#     3. Entity types: using the domain and persona we ask the LLM to extract from the documents the types of entities a node could get (e.g. person, school of thought, supervised learning)
#     4. Examples: Using all of the above we ask the LLM to create some example document->entities extracted
#     5. Entity extraction prompt: We merge all of the above information in a prompt that can be used to extract entities
#     """
#     # TODO
#     pass


    