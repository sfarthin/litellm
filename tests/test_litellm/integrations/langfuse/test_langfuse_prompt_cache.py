from typing import Dict

from unittest.mock import patch

from litellm.integrations.langfuse.langfuse_prompt_management import (
    LangfusePromptManagement,
    langfuse_client_init,
    langfuse_prompt_compilation_cache,
    langfuse_prompt_existence_cache,
)
from litellm.types.llms.openai import ChatCompletionSystemMessage


def _clear_dual_cache(cache) -> None:
    """Helper to clear both the in-memory and base cache layers."""

    in_memory_cache = getattr(cache, "in_memory_cache", None)
    if in_memory_cache is not None:
        in_memory_cache.cache_dict.clear()
        in_memory_cache.ttl_dict.clear()


class TestLangfusePromptCache:
    def setup_method(self) -> None:
        langfuse_client_init.cache_clear()
        _clear_dual_cache(langfuse_prompt_compilation_cache)
        _clear_dual_cache(langfuse_prompt_existence_cache)

    def test_prompt_caching_reuses_compiled_prompt(self):
        class FakePromptClient:
            def __init__(self, config: Dict[str, str]):
                self.config = config

            def compile(self, **kwargs):
                value = kwargs.get("name", "world")
                return [
                    ChatCompletionSystemMessage(
                        role="system", content=f"hello {value}"
                    )
                ]

        fake_client = object()
        with patch(
            "litellm.integrations.langfuse.langfuse_prompt_management.langfuse_client_init",
            return_value=fake_client,
        ):
            langfuse_prompt_management = LangfusePromptManagement()

        langfuse_prompt_management._prompt_client_cache.clear()

        fake_prompt_client = FakePromptClient(
            {"model": "langfuse/test", "temperature": 0.1}
        )
        call_count = 0

        def fake_get_prompt(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return fake_prompt_client

        dynamic_params = {
            "langfuse_public_key": "pk",
            "langfuse_secret": "sk",
            "langfuse_host": "https://example.com",
        }

        with patch.object(
            langfuse_prompt_management,
            "_get_prompt_from_id",
            side_effect=fake_get_prompt,
        ), patch(
            "litellm.integrations.langfuse.langfuse_prompt_management.langfuse_client_init",
            return_value=fake_client,
        ):
            first_response = langfuse_prompt_management.get_chat_completion_prompt(
                model="langfuse/test",
                messages=[],
                non_default_params={},
                prompt_id="prompt-1",
                prompt_variables={"name": "ada"},
                dynamic_callback_params=dynamic_params,
            )
            assert call_count == 2

            second_response = langfuse_prompt_management.get_chat_completion_prompt(
                model="langfuse/test",
                messages=[],
                non_default_params={},
                prompt_id="prompt-1",
                prompt_variables={"name": "ada"},
                dynamic_callback_params=dynamic_params,
            )

            assert call_count == 2, "cached prompt should avoid extra lookups"
            assert first_response == second_response


class TestLangfusePromptExistenceCache:
    def setup_method(self) -> None:
        langfuse_client_init.cache_clear()
        _clear_dual_cache(langfuse_prompt_compilation_cache)
        _clear_dual_cache(langfuse_prompt_existence_cache)

    def test_prompt_existence_cache_avoids_rechecks(self):
        fake_client = object()
        with patch(
            "litellm.integrations.langfuse.langfuse_prompt_management.langfuse_client_init",
            return_value=fake_client,
        ):
            langfuse_prompt_management = LangfusePromptManagement()

        langfuse_prompt_management._prompt_client_cache.clear()

        prompt_call_count = 0

        def fake_get_prompt(*args, **kwargs):
            nonlocal prompt_call_count
            prompt_call_count += 1
            return object()

        dynamic_params = {
            "langfuse_public_key": "pk",
            "langfuse_secret": "sk",
            "langfuse_host": "https://example.com",
        }

        with patch.object(
            langfuse_prompt_management,
            "_get_prompt_from_id",
            side_effect=fake_get_prompt,
        ):
            should_run_first = langfuse_prompt_management.should_run_prompt_management(
                prompt_id="prompt-1",
                dynamic_callback_params=dynamic_params,
            )

            assert should_run_first is True
            assert prompt_call_count == 1

            should_run_second = langfuse_prompt_management.should_run_prompt_management(
                prompt_id="prompt-1",
                dynamic_callback_params=dynamic_params,
            )

            assert should_run_second is True
            assert (
                prompt_call_count == 1
            ), "prompt existence should be memoized across calls"
