from __future__ import annotations

import pytest

from memory_router.classifier import classify
from memory_router.config import Config
from memory_router.router import Router


class _UnavailableProvider:
    name = "stub"

    def is_available(self) -> bool:
        return False

    def complete(self, *args, **kwargs):  # pragma: no cover - not expected
        raise AssertionError("complete() should not be called")


def test_api_mode_fails_closed_without_remote_provider():
    router = Router(Config(mode="api"))
    router.providers = {name: _UnavailableProvider() for name in router.providers}

    with pytest.raises(RuntimeError, match="API mode requires"):
        router.route(classify("Explain bond convexity"))
