#!/bin/bash
Help()
{
   # Display Help
   echo "Build documentation using Sphinx"
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

# Build documenation using sphinx
poetry run sphinx-build -ETa -j auto -D language=en -b html -d docs/build/doctrees docs/source _build
