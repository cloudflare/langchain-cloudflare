"""Pytest configuration and fixtures for langchain-cloudflare tests.

This module provides fixtures for both REST API and Worker binding integration tests.
"""

import os
import shutil
import socket
import subprocess
import time
from contextlib import closing
from pathlib import Path

import pytest
from dotenv import load_dotenv

# Load environment variables from the repo root .env file
# This ensures all tests have access to Cloudflare credentials
_repo_root = Path(
    __file__
).parent.parent.parent.parent  # libs/langchain-cloudflare -> root
_env_file = _repo_root / ".env"
if _env_file.exists():
    load_dotenv(_env_file)
else:
    # Fallback: try the integration_tests directory
    _integration_env = Path(__file__).parent / "integration_tests" / ".env"
    if _integration_env.exists():
        load_dotenv(_integration_env)

# MARK: - Helper Functions


def find_free_port() -> int:
    """Find an available port on localhost."""
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("localhost", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def get_worker_project_dir() -> Path:
    """Get the path to the examples/workers directory."""
    return Path(__file__).parent.parent / "examples" / "workers"


def sync_package_to_python_modules(project_dir: Path) -> None:
    """Copy the latest package source to python_modules for Workers.

    pywrangler has a bug where it doesn't update the bundled packages,
    so we need to manually copy the source files.

    Args:
        project_dir: Path to the examples/workers directory
    """
    # Sync langchain_cloudflare
    src_dir = project_dir.parent.parent / "langchain_cloudflare"
    dest_dir = project_dir / "python_modules" / "langchain_cloudflare"

    if dest_dir.exists():
        # Copy all .py files from source to destination
        for src_file in src_dir.glob("*.py"):
            shutil.copy2(src_file, dest_dir / src_file.name)

    # Also sync sqlalchemy_cloudflare_d1 if configured as local dependency
    sqlalchemy_src = os.environ.get("SQLALCHEMY_D1_LOCAL_PATH")
    if sqlalchemy_src:
        sqlalchemy_src_dir = Path(sqlalchemy_src) / "src" / "sqlalchemy_cloudflare_d1"
        sqlalchemy_dest_dir = (
            project_dir / "python_modules" / "sqlalchemy_cloudflare_d1"
        )
        if sqlalchemy_src_dir.exists() and sqlalchemy_dest_dir.exists():
            for src_file in sqlalchemy_src_dir.glob("*.py"):
                shutil.copy2(src_file, sqlalchemy_dest_dir / src_file.name)


def pywrangler_dev_server(
    project_dir: Path, timeout: int = 300
) -> tuple[subprocess.Popen, int]:
    """Start a pywrangler dev server and return the process and port.

    Args:
        project_dir: Path to the project directory containing wrangler.jsonc
        timeout: Maximum time to wait for server startup (default 300s for CI)

    Returns:
        Tuple of (process, port)
    """
    port = find_free_port()

    # Prepare environment - clear VIRTUAL_ENV to avoid uv conflicts
    # Remove API token env vars to let wrangler use OAuth instead
    # (API tokens may not have edge-preview permissions needed for remote bindings)
    env = os.environ.copy()
    env.pop("VIRTUAL_ENV", None)  # Remove VIRTUAL_ENV to avoid uv warnings/conflicts
    env.pop("CF_API_TOKEN", None)  # Let wrangler use OAuth token
    env.pop("CLOUDFLARE_API_TOKEN", None)
    env.pop("TEST_CF_API_TOKEN", None)

    # Collect output lines for better error reporting
    output_lines = []

    # Start the dev server
    # AI and Vectorize bindings require remote: true in wrangler.jsonc
    # (they don't support local simulation, only remote binding connections)
    # D1 supports both local and remote
    process = subprocess.Popen(
        ["uv", "run", "pywrangler", "dev", "--port", str(port)],
        cwd=project_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )

    # Wait for server to be ready
    start_time = time.time()
    ready_message = "[wrangler:info] Ready on"

    while time.time() - start_time < timeout:
        if process.poll() is not None:
            # Process exited - read any remaining output
            remaining = process.stdout.read() if process.stdout else ""
            output_lines.append(remaining)
            full_output = "\n".join(output_lines)
            raise RuntimeError(f"pywrangler dev exited unexpectedly:\n{full_output}")

        line = process.stdout.readline() if process.stdout else ""
        if line:
            output_lines.append(line.rstrip())

        if ready_message in line:
            return process, port

        # Also check for alternative ready messages
        if f"localhost:{port}" in line.lower() or "ready" in line.lower():
            # Give it a moment to fully initialize
            time.sleep(0.5)
            return process, port

    # Timeout reached
    process.terminate()
    full_output = "\n".join(output_lines)
    raise TimeoutError(
        f"pywrangler dev did not start within {timeout} seconds.\n"
        f"Output:\n{full_output}"
    )


# MARK: - Worker Fixtures


@pytest.fixture(scope="session")
def initialized_worker():
    """Session-scoped fixture that sets up the Worker environment once.

    This runs once per test session to sync the package source to python_modules
    (workaround for pywrangler bug).
    """
    project_dir = get_worker_project_dir()

    # Only sync if examples/workers exists
    if project_dir.exists():
        sync_package_to_python_modules(project_dir)

    return True


@pytest.fixture
def worker_project_dir():
    """Return the examples/workers directory."""
    return get_worker_project_dir()


# Store the session-scoped server state
_session_server: dict = {"process": None, "port": None}


@pytest.fixture(scope="session")
def dev_server(initialized_worker):
    """Session-scoped fixture that starts ONE pywrangler dev server for all tests.

    The server is started once at the beginning of the test session and
    stopped when all tests complete. This avoids the overhead and flakiness
    of starting/stopping the server for each test.

    Yields:
        int: The port number the server is running on
    """
    project_dir = get_worker_project_dir()

    if not project_dir.exists():
        pytest.skip("examples/workers directory not found")

    process = None
    try:
        process, port = pywrangler_dev_server(project_dir)
        _session_server["process"] = process
        _session_server["port"] = port
        yield port
    finally:
        if process is not None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            _session_server["process"] = None
            _session_server["port"] = None


# MARK: - Credential Helpers


def get_cf_credentials():
    """Get Cloudflare credentials from environment variables.

    Uses TEST_CF_API_TOKEN to avoid conflicts with wrangler OAuth auth.
    """
    account_id = os.environ.get("CF_ACCOUNT_ID") or os.environ.get(
        "CLOUDFLARE_ACCOUNT_ID"
    )
    api_token = (
        os.environ.get("TEST_CF_API_TOKEN")
        or os.environ.get("CF_API_TOKEN")
        or os.environ.get("CLOUDFLARE_API_TOKEN")
    )

    if not account_id:
        pytest.skip("CF_ACCOUNT_ID environment variable not set")
    if not api_token:
        pytest.skip("TEST_CF_API_TOKEN environment variable not set")

    return account_id, api_token


# MARK: - Vectorize Index Fixtures


@pytest.fixture(scope="session")
def vectorize_index():
    """Session-scoped fixture that uses the persistent Vectorize index.

    The index 'langchain-test-persistent' is already configured in wrangler.jsonc.
    This fixture just returns the index name.

    Yields:
        str: The name of the persistent index
    """
    index_name = "langchain-test-persistent"
    yield index_name


@pytest.fixture(scope="session")
def dev_server_with_vectorize(dev_server, vectorize_index):
    """Session-scoped fixture that provides the dev server with Vectorize index name.

    Reuses the same dev_server fixture to avoid starting multiple servers.

    Yields:
        tuple: (port, index_name)
    """
    yield dev_server, vectorize_index
