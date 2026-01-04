# ON1Builder

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![Status](https://img.shields.io/badge/Status-Active-success)
![Tests](https://img.shields.io/badge/Tests-Manual%20%2F%20Local-inactive)
![Coverage](https://img.shields.io/badge/Coverage-TBD-lightgrey)

Async, multi-chain MEV/arbitrage engine with safety rails, flashloan support, and live telemetry. Minimal external dependencies (only Etherscan key needed; prices use free, keyless feeds).

---

## Quick Snapshot

| Track | Summary |
| ----- | ------- |
| Core Focus | MEV searcher: arbitrage, back/front-run, flashloans |
| Chains | Ethereum ready (public RPC OK); multi-chain capable |
| Safety | Slippage caps, gas ceilings, balance tiers, emergency stop |
| Telemetry | Heartbeats, perf summaries, structured logs, notifications |

## Feature Highlights


+----------------+--------------------------------------------------+
| Execution      | Async orchestrator, per-chain workers, ML bias   |
| Strategies     | Arbitrage, back/front-run, flashloans (Aave V3)  |
| Market Data    | Keyless price fetch (well-known set), on-chain   |
| Risk           | Gas caps, slippage limits, balance tiers, halt   |
| Monitoring     | Heartbeats, logs, alerts, perf snapshots         |
+----------------+--------------------------------------------------+


## Fast Start

### 1. Clone & env
```bash
git clone https://github.com/John0n1/ON1Builder.git
cd ON1Builder
```
- Windows
```bash
python -m venv .venv
. .venv/Scripts/activate
pip install -r requirements.txt
```
- Linux/MacOS
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure (minimum)
```bash
copy .env.example .env
# edit:
#   WALLET_KEY, WALLET_ADDRESS
#   RPC_URL_1="https://ethereum-rpc.publicnode.com"
#   WEBSOCKET_URL_1="wss://ethereum-rpc.publicnode.com"   # public WS is auto-skipped for txpool
#   ETHERSCAN_API_KEY=...
```

### 3. Run
```bash
python -m on1builder status check
python -m on1builder run start
```

> With public RPC/WS, txpool scanning is disabled on purpose (unreliable pending tx support). Provide a private WS endpoint if you want pending tx monitoring.

## Architecture Map




## Configuration Cheat Sheet

| Setting | Description |
| ------- | ----------- |
| `WALLET_KEY`, `WALLET_ADDRESS` | Required for signing/monitoring |
| `RPC_URL_1` | HTTP RPC endpoint (public OK) |
| `WEBSOCKET_URL_1` | WS endpoint; use private if you want txpool scanning |
| `ETHERSCAN_API_KEY` | Optional but recommended for ABI/tx metadata |
| `MIN_PROFIT_ETH` | Profit floor per trade (ETH) |
| `MAX_GAS_PRICE_GWEI` | Hard gas ceiling |
| `NOTIFICATION_CHANNELS` | `slack,telegram,discord,email` (blank = off) |

Full list lives in `.env.example`.

## Running & Monitoring

```bash
# Validate config
python -m on1builder status check

# Start bot
python -m on1builder run start

# View logs
tail -f logs/on1builder.log          # *nix
Get-Content logs\\on1builder.log -Wait  # Windows
```

Heartbeats report balance tier, pending tx count (0 if txpool scanner is disabled), and memory usage.

## Development

```bash
python -m pytest            # fast/local tests
RUN_LIVE_API_TESTS=1 python -m pytest tests/test_external_api_logic.py
black src tests && flake8 src tests && mypy src
```

Pre-commit hooks are configured in `.pre-commit-config.yaml`.

## Safety Notes

- Public WS endpoints are auto-skipped for txpool scanning to avoid noisy failures.
- Price lookups are limited to a small, well-known token set; repeated failures are silenced after blacklisting.
- Emergency balance tiers keep the bot idle when funds are low.

## License

MIT - see [LICENSE](LICENSE).

## Disclaimer

Use at your own risk. No warranty. MEV strategies can be volatile and may incur losses. Keep keys safe; never use production keys on public demos.
