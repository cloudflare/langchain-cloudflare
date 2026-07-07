# MARK: - Imports
from enum import Enum


# MARK: - StrEnum
class StrEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


# MARK: - Token Errors
class TokenErrors(StrEnum):
    """Error messages for missing or insufficient API token configuration."""

    NO_ACCOUNT_ID_SET = (
        "A Cloudflare account ID must be provided either through "
        "the account_id parameter or "
        "CF_ACCOUNT_ID environment variable. "
        "Alternatively, when running in a Python Worker, you can "
        "pass the 'binding' parameter (env.VECTORIZE) instead."
    )

    INSUFFICIENT_AI_TOKENS = (
        "A Cloudflare API token must be provided either through "
        "the api_token parameter or CF_AI_API_TOKEN environment variable. "
        "Or pass the 'binding' parameter (env.AI) in a Python Worker."
    )

    INSUFFICIENT_VECTORIZE_TOKENS = (
        "Not enough API token values provided. "
        "Please provide a global `api_token` or `vectorize_api_token` "
        "through parameters or environment variables "
        "(CF_API_TOKEN, CF_VECTORIZE_API_TOKEN). "
        "Alternatively, when running in a Python Worker, you can "
        "pass the 'binding' parameter (env.VECTORIZE) instead."
    )

    NO_GLOBAL_TOKEN_WITH_D1_TOKEN = (
        "`d1_database_id` provided, but no global `api_token` provided "
        "and no `d1_api_token` provided. Please set these through parameters "
        "or environment variables (CF_API_TOKEN, CF_D1_API_TOKEN)."
    )

    INSUFFICIENT_AI_SEARCH_TOKENS = (
        "A Cloudflare AI Search API token (with the AI Search:Run permission) "
        "must be provided either through the api_token parameter or the "
        "CF_AI_SEARCH_API_TOKEN (or CF_API_TOKEN) environment variable. "
        "Alternatively, when running in a Python Worker, you can pass the "
        "'binding' parameter (a dedicated ai_search binding) instead."
    )

    INSUFFICIENT_AI_SEARCH_ADMIN_TOKENS = (
        "A Cloudflare AI Search API token (with AI Search:Edit and "
        "AI Search:Run permissions) must be provided either through the "
        "api_token parameter or the CF_AI_SEARCH_API_TOKEN (or CF_API_TOKEN) "
        "environment variable. Alternatively, when running in a Python Worker, "
        "you can pass an ai_search_namespaces binding."
    )

    NO_AI_SEARCH_INSTANCE = (
        "An AI Search instance_name must be provided through the instance_name "
        "parameter or the CF_AI_SEARCH_INSTANCE_NAME environment variable when "
        "using the REST API. Alternatively, when running in a Python Worker, "
        "you can pass the 'binding' parameter instead."
    )
