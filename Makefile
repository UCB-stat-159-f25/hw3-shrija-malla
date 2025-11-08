# Makefile for MyST site

# Name of the conda environment
ENV_NAME = myst-env

# Default target (optional)
.PHONY: all
all: html

.PHONY: env
env:
	@echo ">>> Creating or updating conda environment: $(ENV_NAME)"
	@if conda env list | grep -q "$(ENV_NAME)"; then \
		echo "Environment exists, updating..."; \
		conda env update -n $(ENV_NAME) -f environment.yml --prune; \
	else \
		echo "Creating new environment..."; \
		conda env create -n $(ENV_NAME) -f environment.yml; \
	fi
	@echo "Environment $(ENV_NAME) ready (not activated)."


.PHONY: html
html:
	@echo ">>> Building HTML site..."
	myst build --html
	@echo "Build complete! You can view it locally in the _build/html directory."

.PHONY: clean
clean:
	@echo ">>> Cleaning up build artifacts..."
	rm -rf _build figures audio
	@echo "Cleanup complete."
