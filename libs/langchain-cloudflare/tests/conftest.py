"""Pytest configuration and fixtures for langchain-cloudflare tests.

This module provides fixtures for both REST API and Worker binding integration tests.
"""

import os
import shutil
import socket
import subprocess
import time
import uuid
from contextlib import closing
from pathlib import Path

import pytest
import requests
from dotenv import load_dotenv

# MARK: - Collection Hooks

_STRUCTURED_OUTPUT_STREAMING_TESTS = {
    "test_structured_output",
    "test_structured_output_async",
    "test_structured_output_pydantic_2_v1",
}


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Mark unsupported structured-output streaming contracts as xfail.

    Workers AI returns no chunks when LangChain streams a structured-output
    wrapper, even though equivalent non-streaming calls succeed. Applying the
    marker during collection preserves the upstream LangChain test methods;
    overriding them solely to add a marker violates the inherited integration
    test contract.
    """
    class_node = (
        "tests/integration_tests/test_chat_models.py::TestChatCloudflareWorkersAI::"
    )
    xfail_marker = pytest.mark.xfail(
        reason="Workers AI structured-output wrappers return no streamed chunks",
        strict=False,
    )

    for item in items:
        test_name = item.name.partition("[")[0]
        if item.nodeid.startswith(class_node) and (
            test_name in _STRUCTURED_OUTPUT_STREAMING_TESTS
        ):
            item.add_marker(xfail_marker)


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


def _get_nvm_node_bin() -> str | None:
    """Return the newest installed nvm Node bin dir, if present."""
    nvm_versions = Path.home() / ".nvm" / "versions" / "node"
    if not nvm_versions.exists():
        return None

    bins = sorted(
        (
            path / "bin"
            for path in nvm_versions.iterdir()
            if (path / "bin" / "node").exists()
        ),
        reverse=True,
    )
    return str(bins[0]) if bins else None


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
    """Start a Worker dev server and return the process and port.

    Follows the same sequence as the package.json "dev" script:
    1. ``uv run pywrangler sync`` - install Pyodide-compatible deps
    2. ``./scripts/setup_pyodide_deps.sh`` - install wheels/stubs that
       pywrangler can't handle (langchain>=1.0.0, langgraph, xxhash, etc.)
    3. ``npx wrangler dev`` - start the dev server with the prepared modules

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

    # Prefer a modern nvm-managed Node when the shell PATH points at an older system
    # install. Wrangler 4 requires Node >= 20.
    nvm_node_bin = _get_nvm_node_bin()
    if nvm_node_bin:
        env["PATH"] = f"{nvm_node_bin}:{env['PATH']}"

    # Step 1: Run pywrangler sync to install Pyodide-compatible deps
    sync_result = subprocess.run(
        ["uv", "run", "pywrangler", "sync"],
        cwd=project_dir,
        capture_output=True,
        text=True,
        env=env,
    )
    if sync_result.returncode != 0:
        raise RuntimeError(
            f"pywrangler sync failed:\n{sync_result.stderr}\n{sync_result.stdout}"
        )

    # Step 2: Run setup_pyodide_deps.sh for wheels/stubs pywrangler can't handle
    setup_script = project_dir / "scripts" / "setup_pyodide_deps.sh"
    if setup_script.exists():
        setup_result = subprocess.run(
            [str(setup_script)],
            cwd=project_dir,
            capture_output=True,
            text=True,
            env=env,
        )
        if setup_result.returncode != 0:
            raise RuntimeError(
                f"setup_pyodide_deps.sh failed:\n"
                f"{setup_result.stderr}\n{setup_result.stdout}"
            )

    # pywrangler sync may install the latest published package. Copy the local
    # source after sync/setup so Worker tests run against the checkout under test.
    sync_package_to_python_modules(project_dir)

    # Collect output lines for better error reporting
    output_lines = []

    # Step 3: Start the dev server via npx wrangler dev
    # AI and Vectorize bindings require remote: true in wrangler.jsonc
    # (they don't support local simulation, only remote binding connections)
    # D1 supports both local and remote
    process = subprocess.Popen(
        ["npx", "--yes", "wrangler", "dev", "--port", str(port)],
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
    (workaround for pywrangler bug that doesn't update bundled packages).

    Note: The full dependency setup (pywrangler sync + setup_pyodide_deps.sh +
    wrangler dev) is handled by pywrangler_dev_server().
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


# MARK: - AI Search Fixtures


AI_SEARCH_FIXTURE_QUERY = "langchaincloudflarefixture"
AI_SEARCH_FIXTURE_KEY_PREFIX = "langchain-cloudflare-fixture-"
AI_SEARCH_FIXTURE_DOCUMENTS = [
    (
        # Not "overview": a filename whose terminal slug is exactly
        # "overview" (e.g. "...-overview.md") reproducibly hangs at
        # status="running" forever in Cloudflare AI Search's indexing
        # pipeline, confirmed by isolating it from the other fixture docs --
        # "...-overview-again.md" and other slugs process normally in under
        # a minute, only the bare "overview" terminal slug hangs. Looks like
        # an internal reserved-name collision on Cloudflare's side, not
        # something on our end to fix.
        "intro",
        "\n".join(
            [
                "# LangChain Cloudflare fixture introduction",
                "",
                "langchaincloudflarefixture validates AI Search retrieval.",
                "Cloudflare AI Search returns this introduction document for tests.",
            ]
        ),
    ),
    (
        "workers",
        "\n".join(
            [
                "# LangChain Cloudflare fixture workers",
                "",
                "langchaincloudflarefixture validates Worker binding retrieval.",
                "The Worker test searches this document through an ai_search binding.",
            ]
        ),
    ),
    (
        "rest",
        "\n".join(
            [
                "# LangChain Cloudflare fixture rest",
                "",
                "langchaincloudflarefixture validates REST API retrieval.",
                "The REST test searches this document through the AI Search API.",
            ]
        ),
    ),
]


def get_ai_search_credentials() -> tuple[str, str, str, str]:
    """Get Cloudflare AI Search credentials from environment variables."""
    account_id = os.environ.get("CF_ACCOUNT_ID") or os.environ.get(
        "CLOUDFLARE_ACCOUNT_ID"
    )
    api_token = (
        os.environ.get("CF_AI_SEARCH_API_TOKEN")
        or os.environ.get("TEST_CF_API_TOKEN")
        or os.environ.get("CF_API_TOKEN")
        or os.environ.get("CLOUDFLARE_API_TOKEN")
    )
    instance_name = os.environ.get("CF_AI_SEARCH_INSTANCE_NAME")
    namespace = os.environ.get("CF_AI_SEARCH_NAMESPACE", "default")

    if not account_id:
        pytest.skip("CF_ACCOUNT_ID environment variable not set")
    if not api_token:
        pytest.skip("CF_AI_SEARCH_API_TOKEN environment variable not set")
    if not instance_name:
        pytest.skip("CF_AI_SEARCH_INSTANCE_NAME environment variable not set")

    # The retriever reads CF_AI_SEARCH_API_TOKEN, while older test environments may
    # only provide TEST_CF_API_TOKEN or CF_API_TOKEN.
    os.environ.setdefault("CF_AI_SEARCH_API_TOKEN", api_token)
    os.environ.setdefault("CF_AI_SEARCH_QUERY", AI_SEARCH_FIXTURE_QUERY)

    return account_id, api_token, instance_name, namespace


def _ai_search_instance_url(
    account_id: str,
    instance_name: str,
    namespace: str,
) -> str:
    """Build the AI Search instance REST API base URL."""
    base_url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai-search"
    if namespace and namespace != "default":
        return f"{base_url}/namespaces/{namespace}/instances/{instance_name}"
    return f"{base_url}/instances/{instance_name}"


def _ai_search_request_json(
    session: requests.Session,
    method: str,
    url: str,
    **kwargs,
) -> dict:
    """Issue an AI Search API request and return the JSON body."""
    response = session.request(method, url, timeout=60, **kwargs)
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise RuntimeError(
            f"AI Search API request failed: {method} {url} "
            f"status={response.status_code} body={response.text}"
        ) from exc

    data = response.json()
    if not data.get("success", True):
        raise RuntimeError(f"AI Search API request failed: {data}")
    return data


def _list_ai_search_items(session: requests.Session, instance_url: str) -> list[dict]:
    """List AI Search items for the test instance."""
    data = _ai_search_request_json(
        session,
        "GET",
        f"{instance_url}/items",
        params={"per_page": 50},
    )
    result = data.get("result") or []
    return result if isinstance(result, list) else []


def _delete_ai_search_item(
    session: requests.Session,
    instance_url: str,
    item_id: str,
) -> None:
    """Delete one AI Search item, ignoring already-deleted items."""
    response = session.delete(f"{instance_url}/items/{item_id}", timeout=60)
    if response.status_code == 404:
        return
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise RuntimeError(
            f"AI Search item delete failed: status={response.status_code} "
            f"body={response.text}"
        ) from exc


def _delete_ai_search_fixture_items(
    session: requests.Session,
    instance_url: str,
    item_ids: list[str] | None = None,
) -> None:
    """Delete known AI Search fixture items by ID or by fixture key prefix."""
    if item_ids is None:
        items = _list_ai_search_items(session, instance_url)
        item_ids = [
            item["id"]
            for item in items
            if str(item.get("key", "")).startswith(AI_SEARCH_FIXTURE_KEY_PREFIX)
        ]

    for item_id in item_ids:
        _delete_ai_search_item(session, instance_url, item_id)


def _upload_ai_search_fixture_documents(
    session: requests.Session,
    instance_url: str,
) -> list[str]:
    """Upload deterministic AI Search fixture documents."""
    run_id = uuid.uuid4().hex[:8]
    item_ids = []
    for slug, content in AI_SEARCH_FIXTURE_DOCUMENTS:
        filename = f"{AI_SEARCH_FIXTURE_KEY_PREFIX}{run_id}-{slug}.md"
        files = {
            "file": (
                filename,
                content.encode("utf-8"),
                "text/markdown",
            )
        }
        data = _ai_search_request_json(
            session,
            "POST",
            f"{instance_url}/items",
            files=files,
        )
        result = data.get("result") or {}
        item_id = result.get("id")
        if item_id:
            item_ids.append(item_id)
    return item_ids


def _search_ai_search_fixture(
    session: requests.Session,
    instance_url: str,
) -> list[dict]:
    """Search for the seeded fixture query and return matching chunks."""
    data = _ai_search_request_json(
        session,
        "POST",
        f"{instance_url}/search",
        json={
            "query": AI_SEARCH_FIXTURE_QUERY,
            "ai_search_options": {
                "retrieval": {
                    "max_num_results": 3,
                    "retrieval_type": "hybrid",
                },
                "query_rewrite": {"enabled": False},
                "reranking": {"enabled": False},
            },
        },
    )
    result = data.get("result") if isinstance(data.get("result"), dict) else data
    chunks = result.get("chunks") if isinstance(result, dict) else []
    return chunks or []


def _wait_for_ai_search_fixture(
    session: requests.Session,
    instance_url: str,
    timeout_seconds: int = 420,
) -> None:
    """Wait until the fixture documents are indexed and searchable.

    This is genuine AI Search platform latency, not a code or naming issue
    (confirmed by direct measurement, isolated from any test-suite
    concurrency): each item goes queued -> running -> completed, and
    running alone commonly takes 60-100+ seconds per document regardless of
    document size (a single 195-byte, 1-chunk document took 62s). Uploading
    all 3 fixture documents together and polling to completion, with no
    other load on the account, took 109s end-to-end. 420s leaves comfortable
    headroom above that measured baseline for real-world variance.
    """
    deadline = time.time() + timeout_seconds
    last_status = ""

    while time.time() < deadline:
        items = [
            item
            for item in _list_ai_search_items(session, instance_url)
            if str(item.get("key", "")).startswith(AI_SEARCH_FIXTURE_KEY_PREFIX)
        ]
        statuses = {item.get("key"): item.get("status") for item in items}
        last_status = str(statuses)

        if len(items) >= len(AI_SEARCH_FIXTURE_DOCUMENTS) and all(
            item.get("status") == "completed" for item in items
        ):
            chunks = _search_ai_search_fixture(session, instance_url)
            if len(chunks) >= 3:
                return

        if any(item.get("status") == "error" for item in items):
            raise RuntimeError(f"AI Search fixture indexing failed: {statuses}")

        time.sleep(3)

    raise TimeoutError(
        "AI Search fixture documents were not searchable within "
        f"{timeout_seconds} seconds. Last item status: {last_status}"
    )


@pytest.fixture(scope="session")
def ai_search_test_data():
    """Seed deterministic AI Search data for REST and Worker integration tests."""
    account_id, api_token, instance_name, namespace = get_ai_search_credentials()
    instance_url = _ai_search_instance_url(account_id, instance_name, namespace)

    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {api_token}"})

    uploaded_item_ids: list[str] = []
    try:
        _delete_ai_search_fixture_items(session, instance_url)
        uploaded_item_ids = _upload_ai_search_fixture_documents(session, instance_url)
        _wait_for_ai_search_fixture(session, instance_url)
        yield {
            "instance_name": instance_name,
            "namespace": namespace,
            "query": AI_SEARCH_FIXTURE_QUERY,
            "item_ids": uploaded_item_ids,
        }
    finally:
        _delete_ai_search_fixture_items(
            session,
            instance_url,
            item_ids=uploaded_item_ids,
        )


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
