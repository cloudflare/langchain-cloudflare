#!/bin/bash
# setup_pyodide_deps.sh - Setup dependencies that can't be installed via pywrangler
#
# This handles packages that have no Pyodide wheels:
# - langgraph, langchain-core, langgraph-sdk, langgraph-checkpoint,
#   langchain-protocol (manually extracted wheels, matching the versions this
#   repo's uv.lock resolves)
# - xxhash, ormsgpack, uuid_utils, websockets (pure Python stubs)
#
# Why this is needed:
# - langmem-cloudflare-vectorize depends on langgraph.store.base and
#   langchain-cloudflare, which pulls in the same langgraph/langchain-core
#   chain the langgraph-checkpoint-cloudflare-d1 Worker example needs
# - langgraph depends on langgraph-checkpoint, langgraph-sdk, and xxhash
# - langgraph-checkpoint depends on ormsgpack
# - langchain-core's utils.uuid depends on uuid_utils.compat.uuid7
# - langchain-core's language_models._compat_bridge depends on
#   langchain_protocol.protocol (split out of langchain-core as of the 1.5.x
#   line -- pure Python, just needs extracting like the others, no stub)
# - langgraph.runtime imports langgraph_sdk.auth.types.BaseUser, which drags
#   in the rest of langgraph_sdk's package __init__ -- including, as of
#   langgraph-sdk 0.4.x, a websocket streaming transport this Worker never
#   uses. websockets itself has a C accelerator with no Pyodide wheel.
# - xxhash, ormsgpack, uuid_utils, and websockets are C/Rust extensions (or,
#   for websockets, pull in more than we need) with no usable Pyodide path
# - pywrangler sync fails when it encounters these missing wheels
#
# Solution:
# - Install a Pyodide-compatible langchain-core 0.3.x via pywrangler
# - Replace it by manually extracting the langchain-core/langgraph/langgraph-sdk/
#   langgraph-checkpoint/langchain-protocol wheels actually used by this repo
# - Use pure Python stubs for xxhash, ormsgpack, uuid_utils, and websockets
#
# Adapted from libs/langgraph-checkpoint-cloudflare-d1/examples/workers/scripts/setup_pyodide_deps.sh,
# which was itself adapted from libs/langchain-cloudflare/examples/workers/scripts/setup_pyodide_deps.sh.
# This example only needs langgraph.store / langgraph.graph, not
# langgraph.prebuilt, so unlike langchain-cloudflare's copy this one does not
# extract the separate langgraph-prebuilt wheel.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PYTHON_MODULES="$PROJECT_DIR/python_modules"
STUBS_DIR="$PROJECT_DIR/stubs"
TEMP_DIR="$PROJECT_DIR/.wheels_temp"

# MARK: - Configuration

LANGCHAIN_CORE_VERSION="1.5.4"
LANGGRAPH_VERSION="1.2.11"
LANGGRAPH_SDK_VERSION="0.4.2"
LANGGRAPH_CHECKPOINT_VERSION="4.2.0"
# Pure-Python, split out of langchain-core as of the 1.5.x line; no C
# extensions, so it just needs extracting like the others, no stub.
LANGCHAIN_PROTOCOL_VERSION="0.0.18"

echo "=== Setting up Pyodide-incompatible dependencies ==="
echo "Project: $PROJECT_DIR"
echo "Python modules: $PYTHON_MODULES"

# MARK: - Download and Extract Wheels

download_and_extract_wheel() {
    local package=$1
    local version=$2
    local wheel_name="${package//-/_}-${version}-py3-none-any.whl"
    local url="https://files.pythonhosted.org/packages/py3/${package:0:1}/${package}/${wheel_name}"

    echo "Downloading $package $version..."

    mkdir -p "$TEMP_DIR"
    cd "$TEMP_DIR"

    # Try direct URL first, fall back to pip download
    if ! curl -sL -o "$wheel_name" "$url" 2>/dev/null || [ ! -s "$wheel_name" ]; then
        echo "  Direct download failed, using pip..."
        pip download --no-deps --python-version 3.12 --platform any --only-binary=:all: "${package}==${version}" 2>/dev/null || \
        pip download --no-deps "${package}==${version}" 2>/dev/null
        wheel_name=$(ls ${package//-/_}*.whl 2>/dev/null | head -1)
    fi

    if [ -n "$wheel_name" ] && [ -f "$wheel_name" ]; then
        echo "  Extracting $wheel_name to python_modules..."
        unzip -q -o "$wheel_name" -d "$PYTHON_MODULES"
        rm -f "$wheel_name"
    else
        echo "  ERROR: Failed to download $package $version"
        return 1
    fi

    cd "$PROJECT_DIR"
}

# MARK: - Build and Copy Stubs

build_stub() {
    local stub_name=$1
    local stub_dir="$STUBS_DIR/${stub_name}-stub"

    if [ ! -d "$stub_dir" ]; then
        echo "ERROR: Stub directory not found: $stub_dir"
        return 1
    fi

    echo "Building $stub_name stub..."
    local src_package="$stub_dir/src/$stub_name"
    if [ -d "$src_package" ]; then
        echo "  Copying $stub_name to python_modules..."
        rm -rf "$PYTHON_MODULES/$stub_name"
        cp -r "$src_package" "$PYTHON_MODULES/"
    else
        echo "  ERROR: Source package not found: $src_package"
        return 1
    fi
}

# MARK: - Main

echo ""
echo "Step 1: Downloading and extracting wheels..."

rm -rf "$PYTHON_MODULES/langchain_core" "$PYTHON_MODULES/langchain_core-"*.dist-info 2>/dev/null || true
rm -rf "$PYTHON_MODULES/langgraph" "$PYTHON_MODULES/langgraph-"*.dist-info 2>/dev/null || true
rm -rf "$PYTHON_MODULES/langgraph_sdk" "$PYTHON_MODULES/langgraph_sdk-"*.dist-info 2>/dev/null || true
rm -rf "$PYTHON_MODULES/langgraph_checkpoint" "$PYTHON_MODULES/langgraph_checkpoint-"*.dist-info 2>/dev/null || true
rm -rf "$PYTHON_MODULES/langchain_protocol" "$PYTHON_MODULES/langchain_protocol-"*.dist-info 2>/dev/null || true

download_and_extract_wheel "langchain-core" "$LANGCHAIN_CORE_VERSION"
download_and_extract_wheel "langgraph" "$LANGGRAPH_VERSION"
download_and_extract_wheel "langgraph-sdk" "$LANGGRAPH_SDK_VERSION"
download_and_extract_wheel "langgraph-checkpoint" "$LANGGRAPH_CHECKPOINT_VERSION"
download_and_extract_wheel "langchain-protocol" "$LANGCHAIN_PROTOCOL_VERSION"

echo ""
echo "Step 2: Fixing namespace packages for Pyodide..."
# langgraph and langgraph-checkpoint use PEP 420 namespace packages (no __init__.py).
# Pyodide requires explicit __init__.py files for all packages.
find "$PYTHON_MODULES/langgraph" -type d ! -path '*/.*' | while read dir; do
    if [ ! -f "$dir/__init__.py" ]; then
        echo "  Creating $dir/__init__.py"
        touch "$dir/__init__.py"
    fi
done

echo ""
echo "Step 3: Building and copying stubs..."

rm -rf "$PYTHON_MODULES/xxhash" 2>/dev/null || true
rm -rf "$PYTHON_MODULES/ormsgpack" 2>/dev/null || true
rm -rf "$PYTHON_MODULES/uuid_utils" "$PYTHON_MODULES/uuid_utils-"*.dist-info 2>/dev/null || true
rm -rf "$PYTHON_MODULES/websockets" "$PYTHON_MODULES/websockets-"*.dist-info 2>/dev/null || true

build_stub "xxhash"
build_stub "ormsgpack"
build_stub "uuid_utils"
build_stub "websockets"

# Cleanup
rm -rf "$TEMP_DIR"

echo ""
echo "=== Setup complete ==="
echo ""
echo "Installed packages:"
ls -d "$PYTHON_MODULES/langchain_core"* "$PYTHON_MODULES/langgraph"* "$PYTHON_MODULES/langchain_protocol"* 2>/dev/null | xargs -n1 basename
echo ""
echo "Installed stubs:"
ls -d "$PYTHON_MODULES/xxhash" "$PYTHON_MODULES/ormsgpack" "$PYTHON_MODULES/uuid_utils" "$PYTHON_MODULES/websockets" 2>/dev/null | xargs -n1 basename
echo ""
echo "Next steps:"
echo "  1. Run 'uv run pywrangler sync' to install remaining dependencies"
echo "  2. Run 'uv run pywrangler dev' to start the dev server"
