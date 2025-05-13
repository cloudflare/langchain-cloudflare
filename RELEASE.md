# Langchain Cloudflare Releases

## Prep the Release

- Create a PR to bump the version and update the changelog, including today's date.
  Bump the minor version for new features, patch for a bug fix.

- Update the version in the `pyproject.toml` file in the relevant library directory:
  - For `langchain-cloudflare`: `libs/langchain-cloudflare/pyproject.toml`
  - For `langgraph-checkpoint-cloudflare-d1`: `libs/langgraph-checkpoint-cloudflare-d1/pyproject.toml`

- Merge the PR.

## Run the Release Workflow

- Go to the release [workflow](https://github.com/langchain-ai/langchain-cloudflare/actions/workflows/_release.yml).

- Click "Run Workflow".

- Choose the appropriate library from the dropdown:
  - `libs/langchain-cloudflare`
  - `libs/langgraph-checkpoint-cloudflare-d1`

- Click "Run Workflow".

- The workflow will:
  1. Run lint and format checks
  2. Run tests
  3. Build the package
  4. Publish to PyPI
  5. Create the GitHub Release with auto-generated release notes

## Finish the Release

- Return to the release action and wait for it to complete successfully.

## Troubleshooting

### PyPI Trusted Publishing Setup

Make sure trusted publishing is configured on PyPI for each package:
- Go to PyPI project settings
- Add a trusted publisher for GitHub Actions
- Repository: `langchain-ai/langchain-cloudflare`
- Workflow filename: `_release.yml`
- Environment name: (leave blank)

### Version Conflicts

If you encounter version conflicts:
- Make sure the version in `pyproject.toml` is unique and hasn't been published before
- Follow semantic versioning: MAJOR.MINOR.PATCH

### Failed Tests in Release

If tests fail during release:
- Fix the issues in a new PR
- Update the version number again (since the previous version number was already tagged)
- Re-run the release workflow
