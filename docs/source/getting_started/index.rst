Getting Started
===================================

**Installation**

.. code-block:: python

    pip install graphragzen

**Examples**
These examples are rather intuitive and should get you started fast

.. collapse:: Generating a graph

    .. code-block:: python

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

.. collapse:: Auto-tune an entity extraction prompt

    .. code-block:: python

        from random import sample

        from graphragzen.llm import load_gemma2_gguf
        from graphragzen import preprocessing
        from graphragzen import prompt_tuning


        def create_entity_extraction_prompt() -> str:
            """
            Use an LLM to generate a prompt for entity extraction.
            1. Domain: We fist ask the LLM to create the domains that the documents span
            2. Persona: with the domains the LLM can create a persona (e.g. You are an expert {{role}}.
                You are skilled at {{relevant skills}})
            3. Entity types: using the domain and persona we ask the LLM to extract from the documents
                the types of entities a node could get (e.g. person, school of thought, ML)
            4. Examples: Using all of the above we ask the LLM to create some example document->entities
                extracted
            5. Entity extraction prompt: We merge all of the above information in a prompt that can be
                used to extract entities
            Note: Each function's optional parameters have sane defaults. Check out their
            docstrings for their desrciptions and see if you want to overwrite any
            """
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
            chunked_documents = preprocessing.chunk_documents(raw_documents, llm)

            # Let's not use all documents, that's not neccessary and too slow
            print("Sampling documents")
            chunks = chunked_documents.chunk.tolist()
            sampled_documents = sample(chunks, min([len(chunks), 15]))

            # Get the domain representing the documents
            print("Generating domain")
            domain = prompt_tuning.generate_domain(llm, sampled_documents)

            # Get the persona representing the documents
            print("Generating persona")
            persona = prompt_tuning.generate_persona(llm, domain)

            # Get the entity types present the documents
            print("Generating entity types")
            entity_types = prompt_tuning.generate_entity_types(llm, sampled_documents, domain, persona)

            # Generate some entity relationship examples
            print("Generating entity relationship examples")
            entity_relationship_examples = prompt_tuning.generate_entity_relationship_examples(
                llm, sampled_documents, persona, entity_types, max_examples=3
            )

            # Create the actual entity extraction prompt
            print("Generating entity extraction prompt")
            entity_extraction_prompt = prompt_tuning.create_entity_extraction_prompt(
                llm, entity_types, entity_relationship_examples
            )

            # Also create a prompt to summarize the descriptions of the entities
            print("Generating description summarization prompt")
            description_summarization_prompt = prompt_tuning.create_description_summarization_prompt(
                persona
            )

            return entity_extraction_prompt, description_summarization_prompt


‎ 
