"""Tests for the Web3ConnectionFactory connection flow and caching."""

from types import SimpleNamespace

import pytest

from on1builder.utils.web3_factory import Web3ConnectionFactory


@pytest.fixture(autouse=True)
def _reset_factory_cache():
    Web3ConnectionFactory._connections.clear()
    yield
    Web3ConnectionFactory._connections.clear()


@pytest.mark.asyncio
async def test_web3_factory_accepts_string_keyed_urls(monkeypatch):
    stub_settings = SimpleNamespace(
        websocket_urls={},
        rpc_urls={"1": "http://example"},
        poa_chains=[],
    )
    monkeypatch.setattr("on1builder.config.loaders.get_settings", lambda: stub_settings)

    created = {}

    async def _fake_http(cls, chain_id, http_url):
        created["args"] = (chain_id, http_url)
        return object()

    async def _fake_test(_cls, _web3):
        return True

    monkeypatch.setattr(
        Web3ConnectionFactory,
        "_create_http_connection",
        classmethod(_fake_http),
    )
    monkeypatch.setattr(
        Web3ConnectionFactory, "_test_connection", classmethod(_fake_test)
    )

    result = await Web3ConnectionFactory.create_connection(1, force_new=True)
    assert result is not None
    assert created["args"] == (1, "http://example")


@pytest.mark.asyncio
async def test_web3_factory_caches_connections(monkeypatch):
    stub_settings = SimpleNamespace(
        websocket_urls={},
        rpc_urls={1: "http://example"},
        poa_chains=[],
    )
    monkeypatch.setattr("on1builder.config.loaders.get_settings", lambda: stub_settings)

    created = {"count": 0}

    async def _fake_http(cls, chain_id, http_url):
        created["count"] += 1
        return {"chain_id": chain_id, "url": http_url}

    async def _fake_test(_cls, _web3):
        return True

    monkeypatch.setattr(
        Web3ConnectionFactory,
        "_create_http_connection",
        classmethod(_fake_http),
    )
    monkeypatch.setattr(
        Web3ConnectionFactory, "_test_connection", classmethod(_fake_test)
    )

    first = await Web3ConnectionFactory.create_connection(1, force_new=True)
    second = await Web3ConnectionFactory.create_connection(1)

    assert first is second
    assert created["count"] == 1
