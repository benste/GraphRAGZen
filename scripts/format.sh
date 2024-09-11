#!/bin/bash
Help()
{
   # Display Help
   echo "Formats all python files found using black and isort"
   echo
   echo "Syntax: format [-h]"
   echo "options:"
   echo "h      Print this Help."
   echo "p      setup dependencies through poetry before running the tests"
}

setup_poetry=false
while getopts ":hp" option; do
    case $option in
        h) # display Help
            Help
            exit;;
        p) # setup poetry
            setup_poetry=true
    esac
done

# Make sure we exit on any errors
set -e

# Set-up poetry
if [ $setup_poetry = true ]; then
    echo "setting up Poetry"
    sh ./scripts/setup_poetry.sh -d
    echo $'\n\n\n'
fi

# Apply formatting
echo "Running Black formatting..."
poetry run black --config scripts/test_config/black.toml ./graphragzen/**/*.py
echo "Running isort..."
poetry run isort graphragzen/**/*.py --settings-file scripts/test_config/.isort.cfg
echo "Done!"