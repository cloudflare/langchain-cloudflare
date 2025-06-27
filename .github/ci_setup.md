# CI/CD Setup for Langchain Cloudflare

This directory contains the GitHub Actions workflows for the Langchain Cloudflare project.

## Workflows

### Main CI (`ci.yml`)
- Triggers on pushes to `main` and pull requests
- Determines which packages need linting/testing based on changed files
- Runs pre-commit hooks, linting, and testing in parallel
- Uses matrix strategy to test multiple Python versions (3.9, 3.12)

### Lint Workflow (`_lint.yml`)
- Reusable workflow for linting individual packages
- Runs `make lint` which includes:
  - Ruff linting
  - Ruff format checking (with `--diff`)
  - MyPy type checking
- Caches MyPy cache for faster runs

### Test Workflow (`_test.yml`)
- Reusable workflow for testing individual packages
- Runs `make test` for unit tests
- Runs `make integration_tests` for integration tests (continues on error if tests don't exist)
- Ensures no additional files are created during testing

### Release Workflow (`_release.yml`)
- Manual workflow for releasing packages to PyPI
- Supports both packages:
  - `libs/langchain-cloudflare`
  - `libs/langgraph-checkpoint-cloudflare-d1`
- Runs full CI (lint + test) before building
- Uses PyPI trusted publishing for secure releases
- Creates GitHub releases with auto-generated notes

## Scripts

### `check_diff.py`
- Determines which packages need CI based on changed files
- Outputs JSON arrays for the matrix strategy
- Runs CI on all packages if infrastructure files change

## Package Structure

The project contains two packages in the `libs/` directory:
- `libs/langchain-cloudflare` - Main Langchain Cloudflare integration
- `libs/langgraph-checkpoint-cloudflare-d1` - LangGraph checkpoint store for Cloudflare D1

Each package has its own:
- `pyproject.toml` with version and dependencies
- `Makefile` with `lint`, `format`, `test`, and `integration_tests` targets
- Poetry-based dependency management

## Pre-commit

The project uses pre-commit hooks for:
- Basic file checks (YAML, trailing whitespace, etc.)
- Ruff linting and formatting
- MyPy type checking

## Release Process

See [RELEASE.md](../RELEASE.md) for detailed release instructions.
