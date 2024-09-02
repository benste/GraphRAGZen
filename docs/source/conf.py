# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os
import sys
from docutils import nodes
from docutils.parsers.rst import Directive
import importlib

# Custom print logic 
class ShowVariableWithNewlines(Directive):
    # Define the options for the directive
    required_arguments = 2  # Require two arguments: module name and variable name
    has_content = False

    def run(self):
        module_name = self.arguments[0]
        variable_name = self.arguments[1]

        # Import the module dynamically
        try:
            module = importlib.import_module(module_name)
            variable_value = getattr(module, variable_name)

            # Replace \n with actual newlines for proper display
            formatted_value = variable_value.replace('\\n', '\n')

            # Create a literal block node to display the formatted value
            literal_node = nodes.literal_block(formatted_value, formatted_value)
            return [literal_node]

        except (ImportError, AttributeError) as e:
            error_msg = f"Error: {e}"
            error_node = nodes.error(None, nodes.paragraph(text=error_msg))
            return [error_node]


sys.path.insert(0, os.path.abspath("../.."))
sys.path.insert(0, os.path.abspath('../../graphragzen/prompts/default_prompts'))
sys.path.insert(0, os.path.abspath('../../graphragzen/prompts/prompt_tuning'))

import graphragzen


# Register the custom directive
def setup(app):
    app.add_directive('show_variable_with_newlines', ShowVariableWithNewlines)


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'GraphRAGZen'
copyright = '2024, Ben Steemers'
author = 'Ben Steemers'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'alabaster'
html_theme = 'sphinx_rtd_theme'
# html_theme = 'cloud'

html_static_path = ['_static']
	
# activating extensions
extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    'sphinx_rtd_size',
    'sphinx_collapse'
    ]
    
sphinx_rtd_size_width = "75%"
autosummary_generate = True
autosummary_generate_overwrite = False
napoleon_include_init_with_doc = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']
 
# The master toctree document.
master_doc = "index"

# supported source file types
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
 
 
# -- Options for autodoc ----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#configuration

# Automatically extract typehints when specified and place them in
# descriptions of the relevant function/method.
autodoc_typehints = "description"

# Don't show class signature with the class' name.
autodoc_class_signature = "separated"