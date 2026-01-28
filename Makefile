
python scripts/fetch_historical_data.py
python scripts/run_backtest.py


python scripts/run_optimizer.py --strategy trend_following --pair BTCUSDT
python scripts/run_optimizer.py --strategy mean_reversion --pair BTCUSDT
python scripts/run_optimizer.py --strategy momentum --pair BTCUSDT


streamlit run dashboard/app.py


# Mode paper trading (défaut)
python -m src.live_bot --pairs BTC/USDT ETH/USDT --once
# Avec toutes les options
python -m src.live_bot --pairs BTC/USDT --strategies momentum trend_following --interval 4
# Mode live (attention!)
python -m src.live_bot --live --pairs BTC/USDT