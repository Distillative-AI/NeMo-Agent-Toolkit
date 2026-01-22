# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Dynamo LLM provider with automatic prefix injection for KV cache optimization.

This module provides a specialized OpenAI-compatible LLM that sends Dynamo prefix hints
for optimal KV cache management and request routing. The prefix parameters are optimizable
via the NAT optimizer.

The implementation uses httpx event hooks to inject hints at the HTTP transport level,
making it framework-agnostic (works with LangChain, LlamaIndex, etc.).

Transport Mechanisms
--------------------

This module supports two transport mechanisms for routing hints, used simultaneously
for maximum compatibility:

1. **HTTP Headers** (``x-prefix-*``): For the generalized Thompson Sampling setup
   that uses custom ``frontend.py`` which reads headers directly.
   *DEPRECATED: Will be removed when start_dynamo_unified_thompson_hints.sh is retired.*

2. **nvext.annotations** (in request body): For the optimized Thompson Sampling setup
   that uses the default Dynamo frontend with custom ``processor.py`` which reads
   annotations from the preprocessed request. *This is the preferred mechanism.*

Dynamo Prefix Parameters
-------------------------

prefix_osl (Output Sequence Length)
    Hint for expected response length:

    - LOW: decode_cost=1.0, short responses
    - MEDIUM: decode_cost=2.0, typical responses
    - HIGH: decode_cost=3.0, long responses

prefix_iat (Inter-Arrival Time)
    Hint for request pacing:

    - LOW: iat_factor=1.5, rapid bursts -> high worker stickiness
    - MEDIUM: iat_factor=1.0, normal pacing
    - HIGH: iat_factor=0.6, slow requests -> more exploration

prefix_total_requests
    Expected requests per conversation:

    - Higher values increase KV cache affinity and worker stickiness
    - Lower values allow more load balancing
"""

import json
import logging
import uuid
from collections.abc import Callable
from collections.abc import Coroutine
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING
from typing import Any
from typing import Literal

if TYPE_CHECKING:
    import httpx

from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.llm import LLMProviderInfo
from nat.cli.register_workflow import register_llm_provider
from nat.data_models.optimizable import OptimizableField
from nat.data_models.optimizable import SearchSpace
from nat.llm.openai_llm import OpenAIModelConfig

logger = logging.getLogger(__name__)

# Define valid prefix hint values
PrefixLevel = Literal["LOW", "MEDIUM", "HIGH"]

# =============================================================================
# CONTEXT MANAGEMENT FOR DYNAMO PREFIX ID
# =============================================================================


class DynamoPrefixContext:
    """
    Singleton class for managing Dynamo prefix IDs across LLM calls.

    This allows evaluation code to set a prefix ID that persists across all LLM
    calls for a single evaluation question (multi-turn conversation).

    Usage::

        from nat.llm.dynamo_llm import DynamoPrefixContext

        # Set prefix ID at the start of each evaluation question
        DynamoPrefixContext.set("eval-q001-abc123")

        # ... perform LLM calls ...

        # Clear when done
        DynamoPrefixContext.clear()

        # Or use as a context manager
        with DynamoPrefixContext.scope("eval-q001-abc123"):
            # ... perform LLM calls ...
    """

    _current_prefix_id: ContextVar[str | None] = ContextVar('dynamo_prefix_id', default=None)

    @classmethod
    def set(cls, prefix_id: str) -> None:
        """
        Set the Dynamo prefix ID for the current context.

        Call this at the start of each evaluation question to ensure all LLM calls
        for that question share the same prefix ID (enabling KV cache reuse).

        Args:
            prefix_id: The unique prefix ID (e.g., "eval-q001-abc123")
        """
        cls._current_prefix_id.set(prefix_id)
        logger.debug("Set Dynamo prefix ID: %s", prefix_id)

    @classmethod
    def clear(cls) -> None:
        """Clear the current Dynamo prefix ID context."""
        cls._current_prefix_id.set(None)
        logger.debug("Cleared Dynamo prefix ID")

    @classmethod
    def get(cls) -> str | None:
        """Get the current Dynamo prefix ID from context, if any."""
        return cls._current_prefix_id.get()

    @classmethod
    @contextmanager
    def scope(cls, prefix_id: str) -> Iterator[None]:
        """
        Context manager for scoped prefix ID usage.

        Automatically sets the prefix ID on entry and clears it on exit,
        ensuring proper cleanup even if exceptions occur.

        Args:
            prefix_id: The unique prefix ID for this scope

        Yields:
            None

        Usage:
            with DynamoPrefixContext.scope("eval-q001"):
                # All LLM calls here will use "eval-q001" prefix
                await llm.ainvoke(...)
        """
        cls.set(prefix_id)
        try:
            yield
        finally:
            cls.clear()


# =============================================================================
# DYNAMO MODEL CONFIGURATION
# =============================================================================


class DynamoModelConfig(OpenAIModelConfig, name="dynamo"):
    """
    A Dynamo LLM provider with automatic prefix hint injection for KV cache optimization.

    This is a specialized OpenAI-compatible LLM that sends Dynamo prefix hints
    for optimal KV cache management and request routing. Prefix hints are enabled
    by default using the template "nat-dynamo-{uuid}". The prefix routing parameters
    (prefix_total_requests, prefix_osl, prefix_iat) are optimizable via the NAT optimizer.

    Hints are sent via both HTTP headers (``x-prefix-*``) and ``nvext.annotations``
    in the request body for compatibility with different Dynamo setups:

    - **Generalized Thompson Sampling** (custom frontend.py): Reads HTTP headers
    - **Optimized Thompson Sampling** (default frontend + processor.py): Reads nvext.annotations

    To disable prefix hints, set prefix_template to null/None in your config.
    """

    # =========================================================================
    # DYNAMO PREFIX PARAMETERS
    # =========================================================================

    prefix_template: str | None = Field(
        default="nat-dynamo-{uuid}",
        description="Template for prefix ID. The {uuid} placeholder will be replaced with a unique ID. "
        "Prefix headers are sent by default for KV cache optimization. "
        "Set to null/None to disable prefix header injection.",
    )

    prefix_total_requests: int = OptimizableField(
        default=10,
        ge=1,
        le=50,
        description=("Expected number of requests for this conversation/prefix. "
                     "Higher values increase worker stickiness and KV cache locality. "
                     "Lower values allow more load balancing across workers."),
        space=SearchSpace(low=1, high=20, step=5))

    prefix_osl: PrefixLevel = OptimizableField(default="MEDIUM",
                                               description=("Output Sequence Length hint for the Dynamo router. "
                                                            "LOW=short responses (decode_cost=1.0), "
                                                            "MEDIUM=typical (decode_cost=2.0), "
                                                            "HIGH=long responses (decode_cost=3.0)."),
                                               space=SearchSpace(values=["LOW", "MEDIUM", "HIGH"]))

    prefix_iat: PrefixLevel = OptimizableField(default="MEDIUM",
                                               description=("Inter-Arrival Time hint for the Dynamo router. "
                                                            "LOW=rapid bursts (iat_factor=1.5, high stickiness), "
                                                            "MEDIUM=normal (iat_factor=1.0), "
                                                            "HIGH=slow requests (iat_factor=0.6, more exploration)."),
                                               space=SearchSpace(values=["LOW", "MEDIUM", "HIGH"]))

    request_timeout: float = Field(
        default=600.0,
        gt=0.0,
        description="HTTP request timeout in seconds for LLM requests.",
    )

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    @staticmethod
    def get_dynamo_field_names() -> frozenset[str]:
        """
        Get the set of Dynamo-specific field names for model_dump exclusion.

        Use this when building config dicts for framework clients to exclude
        Dynamo-specific parameters that should not be passed to the underlying client.

        Returns:
            A frozenset of Dynamo-specific field names.

        Example::

            config_dict = config.model_dump(
                exclude={"type", "thinking", *DynamoModelConfig.get_dynamo_field_names()},
                ...
            )
        """
        return frozenset({
            "prefix_template",
            "prefix_total_requests",
            "prefix_osl",
            "prefix_iat",
            "request_timeout",
        })


# =============================================================================
# HTTPX EVENT HOOK FOR HEADER INJECTION
# =============================================================================


def _create_dynamo_request_hook(
    prefix_template: str | None,
    total_requests: int,
    osl: str,
    iat: str,
) -> Callable[["httpx.Request"], Coroutine[Any, Any, None]]:
    """
    Create an httpx event hook that injects Dynamo prefix hints into requests.

    This hook is called before each HTTP request is sent, allowing us to inject
    hints dynamically. The prefix ID is generated ONCE when the hook is created,
    ensuring all requests from the same client share the same prefix ID. This enables
    Dynamo's KV cache optimization across multi-turn conversations.

    The context variable can override this for scenarios where you need different
    prefix IDs (e.g., per-question in batch evaluation).

    Hints are injected via TWO transport mechanisms for maximum compatibility:

    1. **HTTP Headers** (``x-prefix-*``): For the generalized Thompson Sampling setup
       that uses custom ``frontend.py`` which reads headers directly.

    2. **nvext.annotations** (in request body): For the optimized Thompson Sampling
       setup that uses the default Dynamo frontend with custom ``processor.py``
       which reads annotations from the preprocessed request.

    Args:
        prefix_template: Template string with {uuid} placeholder
        total_requests: Expected number of requests for this prefix
        osl: Output sequence length hint (LOW/MEDIUM/HIGH)
        iat: Inter-arrival time hint (LOW/MEDIUM/HIGH)

    Returns:
        An async function suitable for use as an httpx event hook.
    """
    # Generate the default prefix ID ONCE when the hook is created
    # This ensures all requests from this client share the same prefix ID
    unique_id = uuid.uuid4().hex[:16]
    if prefix_template:
        default_prefix_id = prefix_template.format(uuid=unique_id)
    else:
        default_prefix_id = f"nat-dynamo-{unique_id}"

    logger.debug("Created Dynamo request hook with default prefix ID: %s", default_prefix_id)

    async def on_request(request):
        """Inject Dynamo prefix hints into request headers AND body."""
        # Check context variable first (allows per-question override in batch evaluation)
        context_prefix_id = DynamoPrefixContext.get()

        if context_prefix_id:
            prefix_id = context_prefix_id
            logger.debug("Using context prefix ID: %s", prefix_id)
        else:
            # Use the pre-generated prefix ID (same for all requests from this client)
            prefix_id = default_prefix_id
            logger.debug("Using default prefix ID: %s", prefix_id)

        # =====================================================================
        # Transport 1: HTTP Headers (for generalized Thompson Sampling setup)
        # The custom frontend.py reads these headers directly.
        #
        # DEPRECATION NOTE: This transport mechanism exists solely for backwards
        # compatibility with start_dynamo_unified_thompson_hints.sh which uses
        # custom frontend.py/processor.py that read x-prefix-* headers.
        # Once that setup is deprecated in favor of the optimized setup
        # (start_dynamo_optimized_thompson_hints.sh), this header injection
        # can be removed entirely - only nvext.annotations will be needed.
        #
        # AI PROMPT TO REMOVE HTTP HEADERS (use when generalized setup is deprecated):
        # "Remove the HTTP header injection (x-prefix-*) from dynamo_llm.py.
        # Keep only the nvext.annotations transport mechanism. Update docstrings
        # to remove references to HTTP headers and the generalized Thompson
        # Sampling setup. The start_dynamo_unified_thompson_hints.sh script
        # and its custom frontend.py/processor.py are now deprecated."
        # =====================================================================
        request.headers["x-prefix-id"] = prefix_id
        request.headers["x-prefix-total-requests"] = str(total_requests)
        request.headers["x-prefix-osl"] = osl.upper()
        request.headers["x-prefix-iat"] = iat.upper()

        # =====================================================================
        # Transport 2: nvext.annotations (for optimized Thompson Sampling setup)
        # The default Dynamo frontend passes these through to processor.py
        # which extracts them from the PreprocessedRequest.annotations field.
        # =====================================================================
        if request.method == "POST" and request.content:
            try:
                body = json.loads(request.content.decode("utf-8"))
                if isinstance(body, dict):
                    # Build annotations list in "key:value" format
                    annotations = [
                        f"prefix_id:{prefix_id}",
                        f"total_requests:{total_requests}",
                        f"osl:{osl.upper()}",
                        f"iat:{iat.upper()}",
                    ]

                    # Add/merge nvext.annotations
                    if "nvext" not in body:
                        body["nvext"] = {}
                    if not isinstance(body["nvext"], dict):
                        body["nvext"] = {}

                    # Preserve any existing annotations and add ours
                    existing = body["nvext"].get("annotations", [])
                    if not isinstance(existing, list):
                        existing = []

                    # Our annotations take precedence (placed first)
                    body["nvext"]["annotations"] = annotations + [
                        a for a in existing
                        if not any(a.startswith(f"{key}:") for key in ["prefix_id", "total_requests", "osl", "iat"])
                    ]

                    # Re-encode the body
                    new_content = json.dumps(body).encode("utf-8")
                    # Update the request content (httpx allows this via _content)
                    request._content = new_content
                    request.headers["content-length"] = str(len(new_content))

                    logger.debug("Injected nvext.annotations: %s", body["nvext"]["annotations"])

            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                # Not JSON or encoding issue - skip body injection, headers still work
                logger.debug("Could not inject nvext.annotations (body not JSON): %s", e)

        logger.debug("Injected Dynamo hints: prefix_id=%s, total_requests=%d, osl=%s, iat=%s",
                     prefix_id,
                     total_requests,
                     osl.upper(),
                     iat.upper())

    return on_request


def create_httpx_client_with_dynamo_hooks(
    prefix_template: str | None,
    total_requests: int,
    osl: str,
    iat: str,
    timeout: float = 600.0,
) -> "httpx.AsyncClient":
    """
    Create an httpx.AsyncClient with Dynamo prefix hint injection.

    This client can be passed to the OpenAI SDK to inject hints at the HTTP level,
    making it framework-agnostic. Hints are injected via both HTTP headers and
    nvext.annotations in the request body for maximum compatibility with different
    Dynamo setups:

    - **Generalized setup** (custom frontend.py): Reads ``x-prefix-*`` HTTP headers
    - **Optimized setup** (default frontend + custom processor.py): Reads
      ``nvext.annotations`` from the request body

    Args:
        prefix_template: Template string with {uuid} placeholder
        total_requests: Expected number of requests for this prefix
        osl: Output sequence length hint (LOW/MEDIUM/HIGH)
        iat: Inter-arrival time hint (LOW/MEDIUM/HIGH)
        timeout: HTTP request timeout in seconds

    Returns:
        An httpx.AsyncClient configured with Dynamo hint injection.
    """
    import httpx

    request_hook = _create_dynamo_request_hook(prefix_template, total_requests, osl, iat)

    return httpx.AsyncClient(
        event_hooks={"request": [request_hook]},
        timeout=httpx.Timeout(timeout),
    )


# =============================================================================
# PROVIDER REGISTRATION
# =============================================================================
# Note: Client registrations for each framework (LangChain, LlamaIndex, etc.)
# are in the respective plugin packages under packages/nvidia_nat_<framework>/


@register_llm_provider(config_type=DynamoModelConfig)
async def dynamo_llm(config: DynamoModelConfig, _builder: Builder):
    """Register the Dynamo LLM provider."""
    yield LLMProviderInfo(
        config=config,
        description="A Dynamo-optimized model with automatic prefix headers for KV cache management.",
    )
