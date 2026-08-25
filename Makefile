.PHONY: all format lint test tests integration_tests docker_tests help extended_tests worker_tests worker_sync dev_server notebook_sync notebook_check notebook_lab

# Default target executed when no arguments are given to make.
all: help

# Define a variable for the test file path.
TEST_FILE ?= tests/unit_tests/
integration_test integration_tests: TEST_FILE = tests/integration_tests/

# Auto-detect the package name based on current directory
PACKAGE_NAME := $(shell find . -maxdepth 1 -type d -name "*langchain*" -o -name "*langgraph*" | head -1 | sed 's|^\./||')

# Keep interactive notebook dependencies isolated from package test environments.
NOTEBOOK_ENV ?= .venv-notebooks
NOTEBOOK_PYTHON_VERSION ?= 3.12
NOTEBOOK_PYTHON = $(NOTEBOOK_ENV)/bin/python

######################
# NOTEBOOKS
######################

# Create/update the locked notebook environment with this checkout installed.
notebook_sync:
	UV_PROJECT_ENVIRONMENT=$(NOTEBOOK_ENV) uv sync --python $(NOTEBOOK_PYTHON_VERSION) --group notebook

# Verify Jupyter dependencies resolve and langchain-cloudflare is loaded locally.
notebook_check: notebook_sync
	$(NOTEBOOK_PYTHON) -c "from pathlib import Path; import cloudflare, jupyterlab, langchain_cloudflare; package_path = Path(langchain_cloudflare.__file__).resolve(); expected_path = Path('libs/langchain-cloudflare/langchain_cloudflare').resolve(); assert expected_path in package_path.parents or package_path.parent == expected_path, f'Expected local package under {expected_path}, got {package_path}'; print(f'Notebook environment OK: {package_path}')"
	$(NOTEBOOK_PYTHON) -c "from cloudflare import Cloudflare; from langchain_cloudflare import CloudflareAISearchRetriever, CloudflareBrowserRunLoader, CloudflareBrowserRunTool; client = Cloudflare(api_token='not-a-real-token'); assert callable(client.aisearch.namespaces.instances.create); assert callable(client.aisearch.namespaces.instances.search); print('Notebook API shapes OK')"
	$(NOTEBOOK_PYTHON) -m jupyterlab --version

# Launch JupyterLab from the repo root with credentials inherited from .env.
notebook_lab: notebook_sync
	@test -f .env || (echo 'Missing repo-root .env' && exit 1)
	@set -a; . ./.env; set +a; \
	$(NOTEBOOK_PYTHON) -m jupyterlab

# unit tests are run with the --disable-socket flag to prevent network calls
test tests:
	uv run pytest --disable-socket --allow-unix-socket $(TEST_FILE)

test_watch:
	uv run ptw --snapshot-update --now . -- -vv $(TEST_FILE)

# integration tests are run without the --disable-socket flag to allow network calls
# Loads .env from repo root automatically
integration_test integration_tests:
	@if [ -f ../../.env ]; then set -a && . ../../.env && set +a; fi && \
	export TEST_CF_API_TOKEN="$${TEST_CF_API_TOKEN:-$$CF_API_TOKEN}" && \
	unset VIRTUAL_ENV && \
	uv run pytest $(TEST_FILE) -v

# Worker integration tests live outside tests/integration_tests so CI's default
# integration target does not collect Wrangler-dependent tests.
# Requires wrangler OAuth login.
# These test the Python Workers bindings with pywrangler dev server
worker_tests:
	@echo "Running Worker integration tests..."
	@echo "Note: Requires 'npx wrangler login' first"
	@if [ -f ../../.env ]; then set -a && . ../../.env && set +a; fi && \
	unset VIRTUAL_ENV && \
	uv run pytest tests/worker_tests/ -v

# Sync Worker dependencies (run before worker_tests or dev_server)
worker_sync:
	cd examples/workers && uv run pywrangler sync

# Start the dev server manually for debugging
dev_server:
	@echo "Starting pywrangler dev server on port 8799..."
	@echo "Press Ctrl+C to stop"
	cd examples/workers && uv run pywrangler dev --port 8799

######################
# LINTING AND FORMATTING
######################

# Define a variable for Python and notebook files.
PYTHON_FILES=.
MYPY_CACHE=.mypy_cache
lint format: PYTHON_FILES=.
lint_diff format_diff: PYTHON_FILES=$(shell git diff --relative=libs/partners/cloudflare --name-only --diff-filter=d master | grep -E '\.py$$|\.ipynb$$')
lint_package: PYTHON_FILES=$(PACKAGE_NAME)
lint_tests: PYTHON_FILES=tests
lint_tests: MYPY_CACHE=.mypy_cache_test

lint lint_diff lint_package lint_tests:
	[ "$(PYTHON_FILES)" = "" ] || uv run ruff check $(PYTHON_FILES)
	[ "$(PYTHON_FILES)" = "" ] || uv run ruff format $(PYTHON_FILES) --diff
	[ "$(PYTHON_FILES)" = "" ] || mkdir -p $(MYPY_CACHE) && uv run mypy $(PACKAGE_NAME) --cache-dir $(MYPY_CACHE)

format format_diff:
	[ "$(PYTHON_FILES)" = "" ] || uv run ruff format $(PYTHON_FILES)
	[ "$(PYTHON_FILES)" = "" ] || uv run ruff check --select I --fix $(PYTHON_FILES)

spell_check:
	uv run codespell --toml pyproject.toml

spell_fix:
	uv run codespell --toml pyproject.toml -w

check_imports: $(shell find $(PACKAGE_NAME) -name '*.py' 2>/dev/null || echo "")
	[ "$(PACKAGE_NAME)" = "" ] || uv run python ./scripts/check_imports.py $^

######################
# HELP
######################

help:
	@echo '----'
	@echo 'check_imports                - check imports'
	@echo 'format                       - run code formatters'
	@echo 'lint                         - run linters'
	@echo 'test                         - run unit tests'
	@echo 'tests                        - run unit tests'
	@echo 'test TEST_FILE=<test_file>   - run all tests in file'
	@echo 'integration_tests            - run non-Worker integration tests (loads .env automatically)'
	@echo 'worker_sync                  - sync Worker dependencies (pywrangler sync)'
	@echo 'worker_tests                 - run Worker integration tests (requires wrangler login)'
	@echo 'dev_server                   - start pywrangler dev server for debugging'
	@echo 'notebook_sync                - create/update the locked notebook environment'
	@echo 'notebook_check               - verify notebook imports use this checkout'
	@echo 'notebook_lab                 - load .env and launch JupyterLab from the repo root'
