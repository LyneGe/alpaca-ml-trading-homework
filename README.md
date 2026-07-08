# Machine Learning Trading Signal with Alpaca

This homework builds a simple machine learning trading signal using Alpaca market data, evaluates it with a backtest, and demonstrates it in Alpaca paper trading.

## Important Safety Note

This project uses **Alpaca paper trading only**. No real money is used.

## Files

- `ml_trading_signal.ipynb` — main notebook for data collection, feature engineering, PCA, machine learning, backtesting, metrics, and visualizations.
- `paper_trade.py` — paper trading demo script that fetches latest data, computes features, applies PCA, generates a signal, and submits paper orders only.
- `charts/` — saved charts from the notebook.
- `requirements.txt` — Python dependencies.
- `.env.example` — example environment variable file.

## Strategy Summary

The model predicts whether the next-day return will be positive.

Target:

- `1` if next-day return > 0
- `0` otherwise

Signal rule:

- Long if model probability > 0.6
- Flat if model probability <= 0.6

The strategy is long-only, uses no leverage, and does not short sell.

## Backtest Setup

- Initial capital: $100,000
- Strategy: ML signal
- Benchmark: Buy and Hold
- No leverage
- No short selling

Performance metrics:

- Total Return
- CAGR
- Volatility
- Sharpe Ratio
- Sortino Ratio
- Max Drawdown
- Win Rate

## How to Run

Install requirements:

```bash
pip install -r requirements.txt
```

Create your real `.env` file based on `.env.example`:

```txt
ALPACA_API_KEY=your_paper_api_key
ALPACA_SECRET_KEY=your_paper_secret_key
```

Run the notebook:

```bash
jupyter notebook ml_trading_signal.ipynb
```

Run the paper trading demo:

```bash
python paper_trade.py
```

## Video Link

Paste your video link here:

[Video Link](https://drive.google.com/file/d/1fdlm4GnzhLjYq8kj7EosBHCq2MeW8Rva/view?usp=sharing)

## Required Video Statement

In the video, state clearly:

> This is paper trading only — no real money is used.
