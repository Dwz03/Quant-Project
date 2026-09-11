# Quant Trading Project

## 1. Project Overview

This repository is a modular quantitative trading project combining strategy research, backtesting, portfolio and risk management, broker integration, and automated Alpaca paper trading. It is an educational and engineering-focused system: research code can be evaluated independently, while the automated path converts strategy target weights into risk-checked broker orders.

The automated broker path is **paper trading only**. The runner rejects brokers that do not explicitly declare paper-mode capability, and `AlpacaPaperBroker` constructs its Alpaca client with paper mode enabled.

## 2. Architecture

The automated execution path is:

```text
Alpaca market data
        |
        v
TradingStrategy.generate_target_weights(history)
        |
        v
   target_weights
        |
        v
    Rebalancer
        |
        v
    RiskManager
        |
        v
   TradingEngine
        |
        v
 AlpacaPaperBroker
        |
        v
order status and broker-state reconciliation
        |
        v
     Portfolio
```

Research and backtesting are separate from broker execution. The repository contains vectorized and event-driven backtesting components, performance metrics, validation utilities, and research implementations for mean reversion, pairs, and PCA-based statistical arbitrage. These paths do not need to submit broker orders.

### Research experiment evaluation semantics

The reusable research experiment layer uses chronological train, validation, and test splits. Parameter selection may inspect both training and validation data, so reported validation performance is model-selection performance rather than an unbiased final out-of-sample estimate. The test split is not available to parameter selection; its held-out performance is the final untouched evaluation.

Target-weight evaluation requires finite close-to-close returns for every asset in the supplied evaluation universe, even when an asset has zero target weight. Missing or infinite returns fail the experiment explicitly rather than being treated as zero contribution during portfolio aggregation.

The Week 11 standardized validation and robustness phase is complete. See the [research protocol](research/PROTOCOL.md) for the project-wide evidence standard and the [Momentum v1 research note](research/notes/momentum_v1.md) for the first completed strategy decision record.

## 3. Core Components

- **Strategy interfaces and factory** — `TradingStrategy` defines the target-weight interface used by automated execution. `build_strategy()` selects a supported implementation from configuration. The older signal-oriented `Strategy` interface remains available for event-driven experiments.
- **TradingEngine** — connects target weights, rebalancing, risk checks, and broker submission. Broker batches use cumulative and conservative risk projections, deterministic client order IDs, and one-order-per-symbol validation.
- **Rebalancer** — compares target weights with confirmed portfolio positions and creates sell orders followed by buy orders.
- **RiskManager** — checks projected position size, available cash, gross exposure, and leverage.
- **Portfolio and Position** — maintain confirmed cash, positions, average cost, realized and unrealized P&L, and exposure calculations. In paper execution, broker account and position data are the confirmed source of truth.
- **AlpacaPaperBroker** — adapts Alpaca paper-account operations for order submission, status lookup, positions, account state, market clock, asset shortability, open orders, and bounded order history.
- **AlpacaMarketDataAdapter** — requests IEX historical bars and latest trades and rejects missing or stale latest prices.
- **PaperTradingEngine** — schedules the daily strategy, synchronizes broker state, blocks on outstanding orders, classifies durable cycle state, waits for new orders, and prevents duplicate daily decisions.
- **ExecutionHandler** — simulates fills, commission, slippage, and partial fill ratios in backtests and local execution tests. It does not manufacture fills in the Alpaca broker path.

## 4. Strategies

The strategy factory currently supports:

- **Momentum** (`momentum`) — compares the latest price with a configurable historical lookback and assigns long, short, or flat target weights.
- **Moving average** (`moving_average`, `ma`) — compares short- and long-window averages and produces directional target weights.
- **Mean reversion** (`mean_reversion`, `mr`) — compares the current price with a trailing mean and targets a contrarian position.
- **Pairs trading** (`pairs`) — requires exactly two symbols, estimates a hedge relationship, and uses spread z-scores to form market-relative target weights.
- **PCA residual statistical arbitrage** (`pca`, `pca_residual`) — uses PCA residual signals across at least two assets and normalizes them into portfolio weights.

These implementations are research and engineering examples. Their presence does not imply validated profitability or readiness for live capital.

## 5. Paper-Trading Safety

The automated paper path includes these safeguards:

- `AlpacaPaperBroker` explicitly exposes `is_paper = True`, and the runner fails closed unless the broker explicitly reports paper mode.
- Risk checks are batch-aware: previously approved orders affect validation of later orders in the same batch.
- A cumulative-fill projection tracks the endpoint if submitted orders fill, including cumulative position, leverage, and shortability effects.
- A conservative-risk projection reserves pending BUY cash, does not make unfilled SELL proceeds available, and grants no pending order premature exposure-reduction credit.
- Rejected or failed submissions do not change either projection.
- Duplicate symbols in one daily broker batch are rejected before submission, preserving the one-client-order-ID-per-symbol invariant.
- Deterministic client order IDs use the trading date, strategy identity, and symbol.
- Confirmed broker positions are synchronized separately from outstanding orders. Open and partially filled orders block a new rebalance batch.
- Order-wait timeouts return a pending state rather than terminating the scheduler or generating an immediate replacement batch.
- Alpaca all-status order history provides restart-safe state for submitted daily cycles. Filled and terminal-but-incomplete batches are distinguished, and neither is retried automatically that day.
- An atomically replaced JSON marker under `.state/` preserves the one-decision-per-day rule when a cycle produces no broker order record.
- Unexpected cycle failures activate an in-process kill switch and stop further strategy execution.

These controls are intentionally conservative and are not a replacement for a production order-management system.

## 6. Project Structure

```text
quant_project/
├── README.md
├── requirements.txt
├── main.py                         # local research/backtest entry point
├── src/
│   ├── strategy.py                # signal and target-weight strategies
│   ├── strategy_factory.py
│   ├── trading_engine.py
│   ├── paper_trading_engine.py
│   ├── alpaca_broker.py
│   ├── market_data.py
│   ├── rebalancer.py
│   ├── risk_manager.py
│   ├── portfolio.py
│   ├── execution.py
│   ├── backtest.py
│   ├── metrics.py
│   └── research/
│       ├── mean_reversion.py
│       ├── pairs.py
│       └── validation.py
├── scripts/
│   ├── run_paper_strategy.py      # automated paper-only runner
│   ├── paper_dry_run.py
│   └── daily_report.py
├── research/
│   ├── run_validation_experiment.py
│   └── *.md                       # research notes and reports
├── tests/
│   └── test_*.py                  # automated offline tests
├── practice/                       # isolated learning exercises
├── data/                           # local market-data files
└── .state/                         # ignored runtime cycle state
```

Generated data files, logs, reports, caches, the virtual environment, and `.env` are omitted from the diagram. The `.state/` directory is runtime-only and gitignored.

## 7. Installation

Create and activate a virtual environment, then install the pinned repository requirements:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate it with:

```powershell
.venv\Scripts\Activate.ps1
```

Declared dependencies include `alpaca-py`, pandas, NumPy, SciPy, scikit-learn, Statsmodels, pytest, python-dotenv, Matplotlib, and yfinance. Exact versions are defined in `requirements.txt`.

## 8. Environment Variables

The automated runner references these variable names:

- `ALPACA_API_KEY` — required.
- `ALPACA_SECRET_KEY` — required.
- `SYMBOLS` — optional comma-separated universe; defaults to `AAPL,MSFT,GOOG`.
- `STRATEGY_NAME` — optional factory strategy name; defaults to `momentum`.
- `STRATEGY_LOOKBACK` — optional momentum lookback; defaults to `20`.
- `STRATEGY_TARGET_WEIGHT` — optional momentum target weight; defaults to `0.01`.
- `STRATEGY_ALLOW_SHORT` — optional momentum shorting flag; defaults to `false`.

The current factory applies those three strategy parameters to momentum. Other factory strategies currently use their defined factory defaults.

Credentials may be loaded from a local `.env` file. `.env` must never be committed and is gitignored. Never place real credentials in source files, tests, shell history, or documentation.

## 9. Running Tests

After activating the virtual environment, run the suite from the project root:

```bash
PYTHONPATH=. python -m pytest -q tests
```

Broker, paper-engine, reconciliation, restart-safety, idempotency, and batch-risk tests use fake clients and brokers. They do not require real Alpaca credentials or order submission.

### Running the momentum research example

The momentum research runner uses the existing adjusted daily-price loader and standardized experiment framework. By default it evaluates `SPY`, `QQQ`, `AAPL`, and `MSFT` from `2021-01-01` through `2026-01-01`, with 60%/20%/20% chronological splits, lookbacks of 5, 10, 20, and 60 days, a fixed long-only target weight of 0.25, and a proportional transaction-cost rate of 0.001.

Run it manually from the repository root:

```bash
python -m research.run_momentum_research
```

Exact cached files are reused when present; otherwise the existing loader downloads auto-adjusted daily data through yfinance. The terminal report compares Momentum with Equal-Weight Constant Allocation and SPY Buy & Hold on the same validation and test dates. Generated `candidate_results.csv`, `summary.csv`, and `benchmark_comparison.csv` files are written under `research_results/momentum/`, which is gitignored.

Run the expanding-window robustness experiment separately with:

```bash
python -m research.run_momentum_research --walk-forward
```

Its defaults use 252 training observations followed by 63 validation and 63 unseen test observations, advancing 63 observations per fold. It writes `walk_forward_folds.csv`, `walk_forward_summary.csv`, and `walk_forward_benchmark_comparison.csv` to the same gitignored results directory.

## 10. Running the Automated Paper Bot

Configure the required environment variables without exposing credentials on the command line, activate the virtual environment, and run:

```bash
python scripts/run_paper_strategy.py
```

The runner:

1. constructs and verifies an explicitly paper-mode Alpaca broker;
2. loads the configured symbols and strategy;
3. synchronizes confirmed account positions and checks outstanding orders;
4. waits outside the execution window;
5. runs one daily strategy decision during the final 15 minutes before market close;
6. submits risk-approved orders and reconciles their terminal state.

The scheduler polls once per minute. Open or partially filled orders block new decisions. Runtime cycle state is stored at the stable project path `.state/paper_cycle.json`; `.state/` is gitignored.

This command is for Alpaca paper trading only. It does not provide a live-trading mode.

## 11. Current Limitations

- Automated broker execution supports paper trading only.
- Scheduling is daily and low frequency, with a fixed pre-close execution window.
- This is not a production or institutional order-management system.
- The Alpaca paper account should ideally be dedicated to this bot; unrelated open orders conservatively block execution.
- Confirmed positions outside `SYMBOLS` may require additional price and universe handling before rebalancing.
- Runtime state uses a local file, so the project directory must be writable and retained across restarts.
- Risk and pending-order assumptions intentionally favor conservative rejection over maximizing capital usage.
- Factory configuration is not yet uniformly parameterized across every strategy.
- Research validation is still being expanded, and no profitability claim is made.

## 12. Research Roadmap

These are future research directions, not claims about completed functionality:

- regression-based alpha research;
- broader momentum research;
- expanded mean-reversion and pairs research;
- further PCA statistical-arbitrage validation;
- factor research and systematic factor mining.

## 13. Disclaimer

This repository is educational and research software. It is not financial advice, an investment recommendation, or a guarantee of trading performance. Paper-trading behavior does not ensure equivalent live-market results.

## Strategy Registry

The repository contains several strategy implementations, but implementation
does not imply validated profitability or paper-trading readiness.

Strategies move through the following lifecycle:

Research
    -> Validation
    -> Historical Holdout
    -> Paper Integration
    -> Paper Forward Testing
    -> Guarded Live Deployment

A strategy should only enter paper trading after its research specification
has been frozen and it has passed the required validation gates.

| Strategy | CLI Name | Status | Signal Timing | Execution Timing | Paper Order Mode |
|---|---|---|---|---|---|
| Moving Average | `moving_average` | Paper-integrated | Completed daily bars | Final 15 minutes before US market close | Quantity |
| Volatility 20 V1 | `volatility_20` | Historical Holdout PASS / Paper Preview PASS | Completed Close_t | First 5 minutes after next US market open | Notional |
| Momentum | `momentum` | Engineering / research example | Strategy dependent | Not yet promoted to validated paper strategy | Quantity |
| Mean Reversion | `mean_reversion` | Engineering / research example | Strategy dependent | Not yet promoted to validated paper strategy | Quantity |
| Pairs | `pairs` | Research implementation | Strategy dependent | Not yet promoted to validated paper strategy | Quantity |
| PCA Residual | `pca_residual` | Research implementation | Strategy dependent | Not yet promoted to validated paper strategy | Quantity |

The execution schedule is considered part of the strategy specification.
Different strategies therefore do not necessarily trade at the same time.

---

## Paper Trading

The automated broker path is paper trading only.

Before allowing a strategy to submit paper orders, run it in preview mode:

```bash
.venv/bin/python scripts/run_paper_strategy.py \
    --strategy <strategy_name> \
    --preview

.venv/bin/python scripts/run_paper_strategy.py \
    --strategy <strategy_name>

.venv/bin/python scripts/run_paper_strategy.py \
    --strategy moving_average \
    --preview

.venv/bin/python scripts/run_paper_strategy.py \
    --strategy moving_average

.venv/bin/python scripts/run_paper_strategy.py \
    --strategy volatility_20 \
    --preview

.venv/bin/python scripts/run_paper_strategy.py \
    --strategy volatility_20

    

