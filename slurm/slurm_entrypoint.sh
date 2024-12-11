#!/bin/bash
set -e

# Clone the main project repo and install the python package
cd pituitary_nfpa/

# Check out a specific commit according to the environment variable ENTRYPOINT_COMMIT
if [[ -v ENTRYPOINT_COMMIT ]]; then
	git checkout $ENTRYPOINT_COMMIT
fi

# Add the user console script install location to the path
export PATH=${HOME}/.local/bin/:${PATH}

# Install the project's python package
pip install --no-dependencies --user -e .
cd ..

# Kick off the main python entrypoint
pituitary-nfpa "$@"
