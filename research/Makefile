# Makefile for research notebook tests using nbmake

# Use the absolute path to pytest within the sage_env environment
PYTEST = /opt/homebrew/anaconda3/envs/sage_env/bin/pytest

.PHONY: test_all test clear_notebooks check_env prep_commit

# Check if the correct conda environment is active
check_env:
	@if [ "$$CONDA_DEFAULT_ENV" != "sage_env" ]; then \
		echo "Error: 'sage_env' is not active."; \
		echo "Please run: conda activate sage_env"; \
		exit 1; \
	fi

# Run all Jupyter notebooks in the directory
test_all: check_env
	$(PYTEST) --nbmake *.ipynb

# Run specific notebooks provided via the FILES variable
test: check_env
	@if [ -z "$(FILES)" ]; then \
		echo "Error: Please specify files using the FILES variable."; \
		echo "Example: make test FILES=\"HKL-knots.ipynb\""; \
		exit 1; \
	fi
	$(PYTEST) --nbmake $(FILES)

# Clear cell outputs in all Jupyter notebooks in research/
clear_notebooks: check_env
	jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace *.ipynb

# Perform pre-commit cleanup specifically for research
prep_commit: check_env clear_notebooks
	@echo "Research cleanup complete."
