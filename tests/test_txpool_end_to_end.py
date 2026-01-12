"""End-to-end tests for txpool scanning and strategy selection."""

from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from on1builder.engines.strategy_executor import StrategyExecutor
from on1builder.monitoring.txpool_scanner import TxPoolScanner


class DummyBalanceManager:
    def __init__(self, summary):
        self._summary = summary
        self.recorded = []

    async def get_balance_summary(self):
        return self._summary

    async def calculate_optimal_gas_price(self, _expected_profit):
        return 7, True

    async def record_profit(self, profit_amount, strategy_name, *_, **__):
        self.recorded.append((profit_amount, strategy_name))


class DummyTxManager:
    def __init__(self):
        self.calls = []

    async def execute_arbitrage(self, opportunity):
        self.calls.append(("arbitrage", opportunity))
        return {"success": True, "profit_eth": 0.02, "gas_used": 120_000}

    async def execute_front_run(self, opportunity):
        self.calls.append(("front_run", opportunity))
        return {"success": True, "profit_eth": 0.05, "gas_used": 140_000}

    async def execute_back_run(self, opportunity):
        self.calls.append(("back_run", opportunity))
        return {"success": True, "profit_eth": 0.01, "gas_used": 100_000}

    async def execute_sandwich(self, opportunity):
        self.calls.append(("sandwich", opportunity))
        return {"success": False, "profit_eth": 0.0, "gas_used": 180_000}

    async def execute_flashloan_arbitrage(self, opportunity):
        self.calls.append(("flashloan_arbitrage", opportunity))
        return {"success": True, "profit_eth": 0.08, "gas_used": 200_000}


class DummyWeb3:
    def __init__(self, tx):
        self.eth = SimpleNamespace(
            get_transaction=AsyncMock(return_value=tx), chain_id=1
        )

    def from_wei(self, value, unit):
        if unit == "ether":
            return Decimal(value) / Decimal(10**18)
        return value


class DummyABIRegistry:
    def get_monitored_tokens(self, _chain_id):
        return {}


@pytest.mark.asyncio
async def test_txpool_scanner_executes_selected_strategies(monkeypatch):
    router_address = "0x7a250d5630b4cf539739df2c5dacb4c659f2488d"
    stub_settings = SimpleNamespace(
        contracts=SimpleNamespace(uniswap_v2_router={"1": router_address}),
        chains=[1],
        websocket_urls={1: "ws://private"},
        rpc_urls={1: "http://private"},
        allow_unsimulated_trades=True,
        connection_retry_delay=0.1,
    )
    monkeypatch.setattr("on1builder.monitoring.txpool_scanner.settings", stub_settings)
    monkeypatch.setattr(
        "on1builder.monitoring.txpool_scanner.ABIRegistry", lambda: DummyABIRegistry()
    )

    strategy_settings = SimpleNamespace(
        ml_exploration_rate=0.0,
        ml_learning_rate=0.01,
        ml_decay_rate=0.99,
        allow_unsimulated_trades=True,
        ml_update_frequency=9999,
        simulation_concurrency=1,
        mev_strategies_enabled=True,
        front_running_enabled=True,
        back_running_enabled=True,
        sandwich_attacks_enabled=True,
        flashloan_enabled=True,
    )
    monkeypatch.setattr(
        "on1builder.engines.strategy_executor.settings", strategy_settings
    )

    import eth_abi

    amount_in = 2 * 10**18
    amount_out_min = 1 * 10**18
    path = [
        "0x" + "1" * 40,
        "0x" + "2" * 40,
    ]
    recipient = "0x" + "3" * 40
    deadline = 1_700_000_000
    encoded = eth_abi.encode(
        ["uint256", "uint256", "address[]", "address", "uint256"],
        [amount_in, amount_out_min, path, recipient, deadline],
    )
    tx_input = bytes.fromhex("38ed1739" + encoded.hex())

    tx = {
        "hash": bytes.fromhex("11" * 32),
        "from": "0xdead",
        "to": router_address,
        "value": 6 * 10**18,
        "gasPrice": 60 * 10**9,
        "gas": 21000,
        "input": tx_input,
    }
    web3 = DummyWeb3(tx)
    balance_summary = {
        "balance": 2.0,
        "balance_tier": "medium",
        "wallet_address": "0xabc",
        "max_investment": 1.0,
        "profit_threshold": 0.05,
        "flashloan_recommended": False,
        "emergency_mode": False,
    }
    tx_manager = DummyTxManager()
    executor = StrategyExecutor(tx_manager, DummyBalanceManager(balance_summary))

    scanner = TxPoolScanner(web3, executor, chain_id=1)
    await scanner._process_tx_hash(tx["hash"])

    called = {name for name, _ in tx_manager.calls}
    assert {"front_run", "back_run", "arbitrage"} <= called

    for name, opportunity in tx_manager.calls:
        if name in {"front_run", "back_run"}:
            target_tx = opportunity.get("target_tx", {})
            assert target_tx.get("gasPrice") == tx["gasPrice"]
            assert target_tx.get("hash") == tx["hash"].hex()
        assert opportunity.get("dex") == "uniswap_v2"
        assert opportunity.get("path") == path
        assert opportunity.get("amount_in") == amount_in
        assert opportunity.get("profit_potential", 0) > 0


@pytest.mark.asyncio
async def test_txpool_scanner_normalizes_input_data_string(monkeypatch):
    router_address = "0x7a250d5630b4cf539739df2c5dacb4c659f2488d"
    stub_settings = SimpleNamespace(
        contracts=SimpleNamespace(uniswap_v2_router={"1": router_address}),
        chains=[1],
        websocket_urls={1: "ws://private"},
        rpc_urls={1: "http://private"},
        allow_unsimulated_trades=True,
        connection_retry_delay=0.1,
    )
    monkeypatch.setattr("on1builder.monitoring.txpool_scanner.settings", stub_settings)
    monkeypatch.setattr(
        "on1builder.monitoring.txpool_scanner.ABIRegistry", lambda: DummyABIRegistry()
    )

    tx = {
        "hash": bytes.fromhex("22" * 32),
        "from": "0xdead",
        "to": router_address,
        "value": 2 * 10**18,
        "gasPrice": 60 * 10**9,
        "gas": 21000,
        "input": "38ed1739",
    }
    web3 = DummyWeb3(tx)
    executor = SimpleNamespace(
        simulate_opportunities_batch=AsyncMock(), execute_opportunity=AsyncMock()
    )
    scanner = TxPoolScanner(web3, executor, chain_id=1)

    analysis = scanner._analyze_transaction_comprehensive(tx)
    assert analysis["input_data"].startswith("0x")
    assert analysis["target_dex"] == "uniswap_v2"


@pytest.mark.asyncio
async def test_txpool_scanner_simulates_when_required(monkeypatch):
    router_address = "0x7a250d5630b4cf539739df2c5dacb4c659f2488d"
    stub_settings = SimpleNamespace(
        contracts=SimpleNamespace(uniswap_v2_router={"1": router_address}),
        chains=[1],
        websocket_urls={1: "ws://private"},
        rpc_urls={1: "http://private"},
        allow_unsimulated_trades=False,
        connection_retry_delay=0.1,
    )
    monkeypatch.setattr("on1builder.monitoring.txpool_scanner.settings", stub_settings)
    monkeypatch.setattr(
        "on1builder.monitoring.txpool_scanner.ABIRegistry", lambda: DummyABIRegistry()
    )

    import eth_abi

    amount_in = 2 * 10**18
    amount_out_min = 1 * 10**18
    path = [
        "0x" + "1" * 40,
        "0x" + "2" * 40,
    ]
    recipient = "0x" + "3" * 40
    deadline = 1_700_000_000
    encoded = eth_abi.encode(
        ["uint256", "uint256", "address[]", "address", "uint256"],
        [amount_in, amount_out_min, path, recipient, deadline],
    )
    tx_input = bytes.fromhex("38ed1739" + encoded.hex())

    tx = {
        "hash": bytes.fromhex("33" * 32),
        "from": "0xdead",
        "to": router_address,
        "value": 2 * 10**18,
        "gasPrice": 60 * 10**9,
        "gas": 21000,
        "input": tx_input,
    }
    web3 = DummyWeb3(tx)
    executor = SimpleNamespace(
        simulate_opportunities_batch=AsyncMock(side_effect=lambda opps: opps),
        execute_opportunity=AsyncMock(return_value={"success": True}),
    )
    scanner = TxPoolScanner(web3, executor, chain_id=1)

    await scanner._process_tx_hash(tx["hash"])

    assert executor.simulate_opportunities_batch.await_count == 1
    assert executor.execute_opportunity.await_count >= 1


@pytest.mark.asyncio
async def test_txpool_scanner_detects_uniswap_v3(monkeypatch):
    router_address = "0x1f98431c8ad98523631ae4a59f267346ea31f984"
    stub_settings = SimpleNamespace(
        contracts=SimpleNamespace(uniswap_v3_router={"1": router_address}),
        chains=[1],
        websocket_urls={1: "ws://private"},
        rpc_urls={1: "http://private"},
        allow_unsimulated_trades=True,
        connection_retry_delay=0.1,
    )
    monkeypatch.setattr("on1builder.monitoring.txpool_scanner.settings", stub_settings)
    monkeypatch.setattr(
        "on1builder.monitoring.txpool_scanner.ABIRegistry", lambda: DummyABIRegistry()
    )

    import eth_abi

    token_in = "0x" + "1" * 40
    token_out = "0x" + "2" * 40
    fee = 3000
    recipient = "0x" + "3" * 40
    deadline = 1_700_000_000
    amount_in = 2 * 10**18
    amount_out_min = 1 * 10**18
    sqrt_price_limit = 0
    encoded = eth_abi.encode(
        ["(address,address,uint24,address,uint256,uint256,uint256,uint160)"],
        [
            (
                token_in,
                token_out,
                fee,
                recipient,
                deadline,
                amount_in,
                amount_out_min,
                sqrt_price_limit,
            )
        ],
    )
    tx_input = bytes.fromhex("414bf389" + encoded.hex())

    tx = {
        "hash": bytes.fromhex("44" * 32),
        "from": "0xdead",
        "to": router_address,
        "value": 3 * 10**18,
        "gasPrice": 60 * 10**9,
        "gas": 21000,
        "input": tx_input,
    }
    web3 = DummyWeb3(tx)
    executor = SimpleNamespace(
        simulate_opportunities_batch=AsyncMock(side_effect=lambda opps: opps),
        execute_opportunity=AsyncMock(return_value={"success": True}),
    )
    scanner = TxPoolScanner(web3, executor, chain_id=1)

    analysis = scanner._analyze_transaction_comprehensive(tx)
    assert analysis["target_dex"] == "uniswap_v3"
    assert analysis["pool_fee"] == fee

    opportunities = await scanner._analyze_for_opportunities(analysis)
    assert opportunities
    for opp in opportunities:
        assert opp.get("dex") == "uniswap_v3"
        assert opp.get("fee") == fee
