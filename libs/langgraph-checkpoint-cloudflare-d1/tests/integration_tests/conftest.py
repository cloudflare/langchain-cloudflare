"""Pytest configuration and fixtures for REST API integration tests.

These tests exercise CloudflareD1Saver / AsyncCloudflareD1Saver against the
live Cloudflare D1 REST API. See tests/worker_tests/ for the equivalent
suite against WorkerCloudflareD1Saver (the D1 Worker-binding saver).
"""

import os
import uuid
from pathlib import Path

import pytest
from dotenv import load_dotenv

# libs/langgraph-checkpoint-cloudflare-d1/tests/integration_tests -> repo root
_repo_root = Path(__file__).parent.parent.parent.parent.parent
_env_file = _repo_root / ".env"
if _env_file.exists():
    load_dotenv(_env_file)


# MARK: - Credential Helpers


def get_d1_credentials() -> tuple[str, str, str]:
    """Get Cloudflare D1 REST API credentials from environment variables."""
    account_id = os.environ.get("CF_ACCOUNT_ID")
    database_id = os.environ.get("CF_D1_DATABASE_ID")
    api_token = (
        os.environ.get("CF_D1_API_TOKEN")
        or os.environ.get("TEST_CF_API_TOKEN")
        or os.environ.get("CF_API_TOKEN")
    )

    if not account_id:
        pytest.skip("CF_ACCOUNT_ID environment variable not set")
    if not database_id:
        pytest.skip("CF_D1_DATABASE_ID environment variable not set")
    if not api_token:
        pytest.skip("CF_D1_API_TOKEN environment variable not set")

    return account_id, database_id, api_token


# MARK: - Fixtures


@pytest.fixture
def d1_credentials() -> tuple[str, str, str]:
    return get_d1_credentials()


@pytest.fixture
def thread_id() -> str:
    """A fresh thread_id per test, so tests don't collide in the shared table."""
    return f"lgcp-d1-integration-{uuid.uuid4().hex}"
