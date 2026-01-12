"""Intent-focused tests for MainOrchestrator health checks without full startup."""

import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from on1builder.core.main_orchestrator import MainOrchestrator


class DummyWorker:
    def __init__(self, chain_id, last_heartbeat=None):
        self.chain_id = chain_id
        self.last_heartbeat = last_heartbeat or datetime.now() - timedelta(minutes=10)

    async def stop(self):
        return None


class DummyBalanceManager:
    def __init__(self, eth_balance):
        self.eth_balance = eth_balance

    async def get_balance(self, token):
        return self.eth_balance if token == "ETH" else Decimal("0")


@pytest.mark.asyncio
async def test_check_system_health_flags_unhealthy_and_low_balance(monkeypatch):
    orch = MainOrchestrator.__new__(MainOrchestrator)
    orch._workers = [DummyWorker(1)]
    orch._balance_managers = {1: DummyBalanceManager(Decimal("0.005"))}
    orch._startup_time = datetime.now()
    orch._notification_service = SimpleNamespace()
    orch._send_alert = AsyncMock()

    await orch._check_system_health()

    # Expect at least one alert (unresponsive worker or low ETH)
    assert orch._send_alert.await_count >= 1


@pytest.mark.asyncio
async def test_main_orchestrator_startup_avoids_blocking_sleep(monkeypatch):
    def _raise_sleep(_):
        raise AssertionError("time.sleep should not be called during startup")

    stub_manager = SimpleNamespace(get_config=lambda: SimpleNamespace(chains=[1]))

    monkeypatch.setattr(
        "on1builder.core.main_orchestrator.get_config_manager",
        lambda: stub_manager,
    )
    monkeypatch.setattr(
        "on1builder.core.main_orchestrator.initialize_global_config", lambda: None
    )
    monkeypatch.setattr(
        "on1builder.core.main_orchestrator.NotificationService",
        lambda: SimpleNamespace(),
    )
    monkeypatch.setattr(
        "on1builder.core.main_orchestrator.DatabaseInterface",
        lambda: SimpleNamespace(),
    )
    monkeypatch.setattr("on1builder.core.main_orchestrator.time.sleep", _raise_sleep)

    orch = MainOrchestrator()
    orch._initialize_database = AsyncMock()
    orch._initialize_workers = AsyncMock()
    orch._start_services = AsyncMock()
    orch._send_alert = AsyncMock()
    orch._shutdown = AsyncMock()
    orch._shutdown_event.set()

    monkeypatch.setattr(
        "on1builder.core.main_orchestrator.asyncio.sleep",
        AsyncMock(return_value=None),
    )

    await orch.run()


@pytest.mark.asyncio
async def test_main_orchestrator_memory_optimizer_lifecycle(monkeypatch):
    orch = MainOrchestrator.__new__(MainOrchestrator)
    orch._workers = []
    orch._balance_managers = {}
    orch._multi_chain_orchestrator = None
    orch._performance_monitor_task = None
    orch._db_interface = None
    orch._notification_service = None
    orch._shutdown_event = asyncio.Event()
    orch._is_running = False

    init_memory = AsyncMock()
    cleanup_memory = AsyncMock()
    monkeypatch.setattr(
        "on1builder.core.main_orchestrator.initialize_memory_optimization",
        init_memory,
    )
    monkeypatch.setattr(
        "on1builder.core.main_orchestrator.cleanup_memory_optimization",
        cleanup_memory,
    )

    orch._performance_monitor_loop = AsyncMock()
    await orch._start_services()
    init_memory.assert_awaited_once()

    orch._generate_final_report = AsyncMock()
    monkeypatch.setattr(
        "on1builder.core.main_orchestrator.ExternalAPIManager",
        lambda: SimpleNamespace(close=AsyncMock()),
    )
    monkeypatch.setattr(
        "on1builder.core.main_orchestrator.Web3ConnectionFactory.close_all_connections",
        AsyncMock(),
    )

    await orch._shutdown()
    cleanup_memory.assert_awaited_once()
