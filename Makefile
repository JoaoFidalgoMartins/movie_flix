# Define the name of the virtual environment
VENV_DIR = .venv

# Define the requirements file
REQUIREMENTS_FILE = requirements.txt

# Define the Python interpreter
PYTHON = python3

# Define the source directory
SRC_DIR = src

# Create a virtual environment
$(VENV_DIR)/bin/activate:
	$(PYTHON) -m venv $(VENV_DIR)

# Install requirements
install: $(VENV_DIR)/bin/activate
	$(VENV_DIR)/bin/pip install -r $(REQUIREMENTS_FILE)

# Set up the environment with the source directory as the root
setup: install
	export PYTHONPATH=$(SRC_DIR):$$PYTHONPATH

# Clean up the virtual environment
clean:
	rm -rf $(VENV_DIR)

.PHONY: install setup clean