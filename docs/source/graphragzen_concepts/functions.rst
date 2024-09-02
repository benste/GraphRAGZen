Functions
----------

**GraphRAGZen** utilizes semi-pure python functions to maintain a modular and intuitive library.

This means that:

- It does not modifying global variables.
- It does not mutate input.
- If no LLM output is required, the same output is guaranteed for the same input.
    - If an LLM output is requisted this no longer holds (hence semi-pure)


All function inputs are organized according to:

.. code-block:: python

    def somefunction(
            data_from_pipeline: type-hint,
            other_data_from_pipeline: type-hint,
            parameter_1: str = "some_string",
            parameter_2: bool = True,
        )

1. The first *n* inputs are always data as expected from a data-pipeline (loaded documents, LLM
instance, extracted graph, etc.)

2. The later inputs are always parameters. These are the parameters that determing how the function
operates.

3. The parameters all sane default values (if possible, e.g. raw_documents_path cannot have a default)
