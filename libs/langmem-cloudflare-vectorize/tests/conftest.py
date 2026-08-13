"""Pytest configuration and fixtures for Worker binding integration tests.

Starts a `pywrangler dev` server against examples/workers and exposes its
port to tests/worker_tests/. See tests/integration_tests/conftest.py for the
REST API equivalent (no Worker/wrangler involved there).
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

# libs/langmem-cloudflare-vectorize/tests -> repo root
_repo_root = Path(__file__).parent.parent.parent.parent
_env_file = _repo_root / ".env"
if _env_file.exists():
    load_dotenv(_env_file)

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

    pywrangler ignores the [tool.uv.sources] local-path overrides in
    examples/workers/pyproject.toml and resolves langchain-cloudflare from
    PyPI instead (a stale release, predating the binding-related fixes this
    example needs), so we manually copy both packages' source files after
    every sync -- same workaround the checkpointer and langchain-cloudflare
    Worker examples use. langmem-cloudflare-vectorize itself is not a
    pywrangler dependency at all (see examples/workers/pyproject.toml's
    comment on why), so this is the only way its source ever lands there.
    """
    repo_libs_dir = project_dir.parent.parent.parent

    for package_dir_name, module_name in [
        ("langchain-cloudflare", "langchain_cloudflare"),
        ("langmem-cloudflare-vectorize", "langmem_cloudflare_vectorize"),
    ]:
        src_dir = repo_libs_dir / package_dir_name / module_name
        dest_dir = project_dir / "python_modules" / module_name
        if not src_dir.exists():
            continue
        dest_dir.mkdir(parents=True, exist_ok=True)
        for src_file in src_dir.glob("*.py"):
            shutil.copy2(src_file, dest_dir / src_file.name)


def pywrangler_dev_server(
    project_dir: Path, timeout: int = 300
) -> tuple[subprocess.Popen, int]:
    """Start a Worker dev server and return the process and port.

    1. ``uv run pywrangler sync`` - install Pyodide-compatible deps
    2. ``bash scripts/setup_pyodide_deps.sh`` - swap in the langgraph/langchain-core
       wheels and xxhash/ormsgpack/uuid_utils/websockets stubs pywrangler can't handle
    3. ``npx wrangler dev`` - start the dev server with the prepared modules
    """
    port = find_free_port()

    env = os.environ.copy()
    env.pop("VIRTUAL_ENV", None)  # Avoid uv warnings/conflicts
    env.pop("CF_API_TOKEN", None)  # Let wrangler use OAuth instead of an API token
    env.pop("CLOUDFLARE_API_TOKEN", None)
    env.pop("TEST_CF_API_TOKEN", None)

    # Wrangler 4 requires Node >= 20; prefer a modern nvm-managed Node if present.
    nvm_node_bin = _get_nvm_node_bin()
    if nvm_node_bin:
        env["PATH"] = f"{nvm_node_bin}:{env['PATH']}"

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

    setup_script = project_dir / "scripts" / "setup_pyodide_deps.sh"
    if setup_script.exists():
        setup_result = subprocess.run(
            ["bash", str(setup_script)],
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

    # pywrangler sync resolves the published PyPI package. Copy the local
    # source after sync/setup so Worker tests run against the checkout under test.
    sync_package_to_python_modules(project_dir)

    output_lines = []

    process = subprocess.Popen(
        ["npx", "--yes", "wrangler", "dev", "--port", str(port)],
        cwd=project_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )

    start_time = time.time()
    ready_message = "[wrangler:info] Ready on"

    while time.time() - start_time < timeout:
        if process.poll() is not None:
            remaining = process.stdout.read() if process.stdout else ""
            output_lines.append(remaining)
            full_output = "\n".join(output_lines)
            raise RuntimeError(f"pywrangler dev exited unexpectedly:\n{full_output}")

        line = process.stdout.readline() if process.stdout else ""
        if line:
            output_lines.append(line.rstrip())

        if ready_message in line:
            return process, port

        if f"localhost:{port}" in line.lower() or "ready" in line.lower():
            time.sleep(0.5)
            return process, port

    process.terminate()
    full_output = "\n".join(output_lines)
    raise TimeoutError(
        f"pywrangler dev did not start within {timeout} seconds.\nOutput:\n{full_output}"
    )


# MARK: - Worker Fixtures


@pytest.fixture(scope="session")
def initialized_worker():
    """Session-scoped fixture that syncs the package source once up front."""
    project_dir = get_worker_project_dir()
    if project_dir.exists():
        sync_package_to_python_modules(project_dir)
    return True


_session_server: dict = {"process": None, "port": None}


@pytest.fixture(scope="session")
def dev_server(initialized_worker):
    """Session-scoped fixture that starts ONE pywrangler dev server for all tests.

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
