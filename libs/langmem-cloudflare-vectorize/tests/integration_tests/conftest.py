"""Pytest configuration and fixtures for REST API integration tests.

These tests exercise CloudflareVectorizeBaseStore against the live Cloudflare
Vectorize/Workers AI REST APIs. See tests/worker_tests/ for the equivalent
suite against the Vectorize/AI Worker bindings.
"""

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

# libs/langmem-cloudflare-vectorize/tests/integration_tests -> repo root
_repo_root = Path(__file__).parent.parent.parent.parent.parent
_env_file = _repo_root / ".env"
if _env_file.exists():
    load_dotenv(_env_file)


# MARK: - Credential Helpers


def get_vectorize_credentials() -> tuple[str, str, str]:
    """Get Cloudflare Vectorize/Workers AI REST API credentials."""
    account_id = os.environ.get("CF_ACCOUNT_ID")
    ai_api_token = (
        os.environ.get("CF_AI_API_TOKEN")
        or os.environ.get("TEST_CF_API_TOKEN")
        or os.environ.get("CF_API_TOKEN")
    )
    vectorize_api_token = (
        os.environ.get("CF_VECTORIZE_API_TOKEN")
        or os.environ.get("TEST_CF_API_TOKEN")
        or os.environ.get("CF_API_TOKEN")
    )

    if not account_id:
        pytest.skip("CF_ACCOUNT_ID environment variable not set")
    if not ai_api_token:
        pytest.skip("CF_AI_API_TOKEN environment variable not set")
    if not vectorize_api_token:
        pytest.skip("CF_VECTORIZE_API_TOKEN environment variable not set")

    return account_id, ai_api_token, vectorize_api_token


# MARK: - Fixtures


@pytest.fixture(scope="session")
def vectorize_credentials() -> tuple[str, str, str]:
    return get_vectorize_credentials()
