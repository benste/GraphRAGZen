# custom_directives.py
from docutils import nodes
from docutils.parsers.rst import Directive, directives
import importlib

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