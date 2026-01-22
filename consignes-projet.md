# 📖 CONSIGNES PROJET - Trading Algorithmique Crypto Swing

**Document de référence pour le développement du bot de trading algorithmique.**  
À consulter avant chaque phase et à transmettre aux IAs qui vous aident.

---

## 📋 Table des matières

1. [Principes Généraux](#principes-généraux)
2. [Standards de Code (PEP 8)](#standards-de-code-pep-8)
3. [Gestion des Secrets & Sécurité](#gestion-des-secrets--sécurité)
4. [Phase 1 : BACKTEST](#phase-1--backtest)
5. [Phase 2 : OPTIMISATION](#phase-2--optimisation)
6. [Phase 3 : PAPER TRADING](#phase-3--paper-trading)
7. [Testing avec Pytest](#testing-avec-pytest)
8. [Logging & Monitoring](#logging--monitoring)
9. [Git & Versioning](#git--versioning)
10. [Checklist Intégrateur IA](#checklist-intégrateur-ia)

---

## Principes Généraux

### 🎯 Règle #1 : Pas d'argent réel jusqu'à preuve absolue

**Principe** : Backtest → Optimisation → Paper Trading (4+ semaines) → LIVE

**À FAIRE** ✅ :
```python
# Phase 1 : Backtest uniquement
if __name__ == "__main__":
    config = load_config("config/backtest_config.json")
    data = fetch_historical_data(config['pairs'], config['start_date'], config['end_date'])
    backtest_results = run_backtest(data, strategy, initial_capital=2000)
    # Exporte résultats en CSV, analyse en Jupyter
    export_backtest_results(backtest_results, "data/backtest_results/")

# Phase 2 : Paper trading UNIQUEMENT
bot = LiveBot(strategy, pairs, paper_trading=True)  # Jamais False ici
bot.run()
```

**À ÉVITER** ❌ :
```python
# ❌ JAMAIS : Passer directement au live sans validation complète
bot = LiveBot(strategy, pairs, paper_trading=False)  # DANGER !
bot.run()

# ❌ JAMAIS : Modifier une stratégie "juste un peu" en live
# Toute modification = retour à backtest + paper trading
```

---

### 🎯 Règle #2 : Overfitting = ennemi #1

**Principe** : Une stratégie qui performe à 200% en backtest mais 5% en live = OVERFITTING.

**À FAIRE** ✅ :
```python
# Walk-forward analysis : valide sur PLUSIEURS périodes disjointes
train_period = "2022-01-01:2023-06-30"
test_period = "2023-07-01:2023-12-31"
validation_period = "2024-01-01:2024-06-30"

for period in [test_period, validation_period]:
    backtest_results = run_backtest_on_period(data, strategy, period)
    # Vérifie que le Sharpe Ratio et Win Rate restent stables
    assert backtest_results['sharpe_ratio'] > 1.0, "Sharpe dégradé en validation"
    assert backtest_results['win_rate'] > 45, "Win rate trop faible"
```

**À ÉVITER** ❌ :
```python
# ❌ JAMAIS : Optimiser sur TOUS les données sans validation
backtest(data, strategy_params)  # Data = 2 ans complètes
# Résultat : overfitting garanti

# ❌ JAMAIS : Ajouter de nouveaux indicateurs basé sur "ce que j'ai vu"
# Les stratégies avec 20+ indicateurs overfittent toujours
```

**Détection du surapprentissage** :
| Métrique | Signal OK ✅ | Signal Danger ⚠️ |
|----------|-------------|-----------------|
| Sharpe Ratio Backtest vs Paper | Différence < 0.5 | Différence > 1.0 |
| Win Rate | Backtest 55% → Paper 52% | Backtest 75% → Paper 30% |
| Total Return | Backtest 45% → Paper 40% | Backtest 150% → Paper 5% |
| Drawdown Max | Semblable (+/- 3%) | Très différent (> 10%) |

---

### 🎯 Règle #3 : Logs & Traçabilité

**Principe** : Chaque trade, chaque signal, chaque erreur doit être loggé.

**À FAIRE** ✅ :
```python
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"logs/trading_{datetime.now().strftime('%Y%m%d')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Log chaque événement important
logger.info(f"Signal générée : BUY BTCUSDT @ 43500 USDT")
logger.warning(f"Position non fermée après 48h : ETHUSDT")
logger.error(f"Connexion API Binance échouée, retry dans 5s")

# Exporte aussi en CSV pour dashboard
logger.info("Trade fermée : PnL = +2.5%, résultat exporté en CSV")
```

**À ÉVITER** ❌ :
```python
# ❌ JAMAIS : Pas de logs du tout
def buy(pair, price):
    order = exchange.create_order(pair, price)  # On ne sait pas ce qui s'est passé
    return order

# ❌ JAMAIS : Logs non structurés
print("BTCUSDT bought")  # Pas de timestamp, pas de prix, pas de traçabilité
print("Error")  # Quel erreur ? Quelle ligne ?
```

---

## Standards de Code (PEP 8)

### Structure générale

**À FAIRE** ✅ :
```python
"""
Module: strategies.py
Description: Implémentation des stratégies de trading (Trend Following, Mean Reversion, MACD)
Author: [Ton nom]
Date: 2025-01-22
Version: 1.0
"""

import logging
from typing import Tuple, Dict, Optional
from abc import ABC, abstractmethod

import pandas as pd
import numpy as np
import ta

logger = logging.getLogger(__name__)


class Strategy(ABC):
    """Classe abstraite pour toutes les stratégies."""
    
    def __init__(self, name: str):
        """
        Initialize Strategy.
        
        Args:
            name (str): Nom de la stratégie
        """
        self.name = name
        logger.info(f"Strategy initialized: {name}")
    
    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Génère les signaux de trading.
        
        Args:
            df (pd.DataFrame): DataFrame OHLCV avec colonnes [open, high, low, close, volume]
        
        Returns:
            pd.DataFrame: DataFrame augmentée avec colonne 'signal' (1=BUY, -1=SELL, 0=HOLD)
        
        Raises:
            ValueError: Si colonnes manquantes
        """
        pass


class TrendFollowing(Strategy):
    """Stratégie de suivi de tendance SMA."""
    
    def __init__(
        self,
        sma_short: int = 20,
        sma_long: int = 50,
        name: str = "trend_following"
    ):
        """Initialise la stratégie Trend Following.
        
        Args:
            sma_short (int): Période de la SMA courte (défaut: 20)
            sma_long (int): Période de la SMA longue (défaut: 50)
            name (str): Nom de la stratégie
        
        Raises:
            ValueError: Si sma_short >= sma_long
        """
        super().__init__(name)
        
        if sma_short >= sma_long:
            raise ValueError(f"SMA court ({sma_short}) doit être < SMA long ({sma_long})")
        
        self.sma_short = sma_short
        self.sma_long = sma_long
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Génère les signaux de suivi de tendance."""
        required_cols = ['close']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Colonnes manquantes : {required_cols}")
        
        df = df.copy()
        
        df['sma_short'] = ta.trend.sma_indicator(df['close'], self.sma_short)
        df['sma_long'] = ta.trend.sma_indicator(df['close'], self.sma_long)
        
        # Croisement : SMA court passe au-dessus de SMA long = BUY
        df['signal'] = np.where(
            df['sma_short'] > df['sma_long'],
            1,  # BUY
            -1   # SELL
        )
        
        return df
```

**À ÉVITER** ❌ :
```python
# ❌ Pas de documentation
def trend_following(df):
    sma_short = df['close'].rolling(20).mean()
    sma_long = df['close'].rolling(50).mean()
    signal = np.where(sma_short > sma_long, 1, -1)
    return signal

# ❌ Nommage non PEP8
def TRENDFOLLOW(df):
    pass

def trend_following_v2_optimized_final_v3(df):
    pass

# ❌ Fonctions trop longues (>50 lignes) sans découpe
def backtest(data, strategy, optimize=True, validation=False, walk_forward=True):
    # 200 lignes de code sans structure
    pass
```

### Règles PEP 8 essentielles

| Élément | Standard ✅ | À ÉVITER ❌ |
|---------|-----------|----------|
| **Noms de variables** | `entry_price`, `rsi_period` | `ep`, `p`, `entrPrice` |
| **Noms de fonctions** | `calculate_drawdown()` | `calc()`, `CalcDrawdown()` |
| **Noms de classes** | `TrendFollowing`, `TradeLogger` | `trendFollowing`, `trade_logger` |
| **Constantes** | `MAX_POSITION_SIZE = 0.02` | `max_position_size = 0.02` |
| **Lignes** | Max 79 caractères | `this_is_a_very_long_line_that_exceeds_79_chars_and_makes_code_hard_to_read()` |
| **Import** | `import pandas as pd` | `from pandas import *` |
| **Espacement** | 2 lignes vides entre fonctions | `def f1():\n    pass\ndef f2():\n    pass` |

---

## Gestion des Secrets & Sécurité

### 🔐 Règle #1 : JAMAIS de clés en dur

**À FAIRE** ✅ :
```python
# .env (JAMAIS committer ce fichier)
BINANCE_API_KEY=your_actual_key_here
BINANCE_API_SECRET=your_actual_secret_here
DATABASE_URL=sqlite:///trading.db

# .gitignore (toujours inclure)
.env
.env.local
config/secrets.json
*.log
data/live_trading/*.csv
```

**Python code** :
```python
import os
from dotenv import load_dotenv

# Charge depuis .env
load_dotenv()

API_KEY = os.getenv("BINANCE_API_KEY")
API_SECRET = os.getenv("BINANCE_API_SECRET")

if not API_KEY or not API_SECRET:
    raise ValueError("❌ Clés API manquantes dans .env")

# Utilise seulement en variables, jamais en dur
exchange = ccxt.binance({
    'apiKey': API_KEY,
    'secret': API_SECRET,
    'enableRateLimit': True
})
```

**À ÉVITER** ❌ :
```python
# ❌ JAMAIS
API_KEY = "sk_live_abcd1234..."  # Clé en dur !
API_SECRET = "secret_xyz789..."

# ❌ JAMAIS committer
git add .env
git commit -m "Added API keys"  # CATASTROPHE

# ❌ JAMAIS partager via email/Slack
# "Voici ma clé API..."
```

### 🔐 Règle #2 : Structure des secrets

```
crypto-swing-bot/
├── .env.example          # Template à committer
├── .env                  # ⚠️ À .gitignore
├── config/
│   ├── strategies.json   # Public (config stratégies)
│   ├── pairs.json        # Public (paires à trader)
│   └── .secrets.json     # ⚠️ À .gitignore (clés DB, etc.)
└── .gitignore
```

**.env.example** (à committer) :
```
# Binance API (paper trading testnet)
BINANCE_API_KEY=your_testnet_key_here
BINANCE_API_SECRET=your_testnet_secret_here
BINANCE_TESTNET_ENABLED=true

# Base de données
DATABASE_URL=sqlite:///data/trading.db

# Logging
LOG_LEVEL=INFO
LOG_DIR=logs/
```

---

## Phase 1 : BACKTEST

### 📋 Checklist Démarrage

- [ ] Récupérer 2 ans de données historiques (BTCUSDT, ETHUSDT, etc.)
- [ ] Implémenter **3 stratégies simples** minimum (SMA, RSI, MACD)
- [ ] Chaque stratégie teste ses signaux sur les données
- [ ] Logger les résultats en CSV (`backtest_results/`)
- [ ] Analyse Jupyter des 3 stratégies

### Récupération des données

**À FAIRE** ✅ :
```python
# src/data_fetcher.py
import ccxt
import pandas as pd
from typing import List, Tuple
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class BinanceDataFetcher:
    """Récupère les données OHLCV depuis Binance."""
    
    def __init__(self, testnet: bool = False):
        """Initialize Binance Data Fetcher."""
        self.exchange = ccxt.binance({
            'sandbox': testnet,
            'enableRateLimit': True
        })
    
    def fetch_ohlcv(
        self,
        pair: str,
        timeframe: str = '4h',
        start_date: str = '2023-01-01',
        end_date: str = '2025-01-22'
    ) -> pd.DataFrame:
        """
        Récupère OHLCV historique pour une paire.
        
        Args:
            pair (str): Paire trading (ex: 'BTC/USDT')
            timeframe (str): Timeframe (ex: '4h', '1d')
            start_date (str): Date de début (YYYY-MM-DD)
            end_date (str): Date de fin (YYYY-MM-DD)
        
        Returns:
            pd.DataFrame: DataFrame OHLCV
        
        Raises:
            Exception: Si erreur API
        """
        try:
            since = self.exchange.parse8601(f"{start_date}T00:00:00Z")
            until = self.exchange.parse8601(f"{end_date}T23:59:59Z")
            
            all_candles = []
            
            while since < until:
                logger.info(f"Fetching {pair} from {datetime.fromtimestamp(since/1000)}")
                
                candles = self.exchange.fetch_ohlcv(
                    pair,
                    timeframe,
                    since,
                    limit=500
                )
                
                if not candles:
                    break
                
                all_candles.extend(candles)
                since = candles[-1][0] + 1
            
            df = pd.DataFrame(
                all_candles,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            logger.info(f"✅ Fetched {len(df)} candles for {pair}")
            return df
        
        except Exception as e:
            logger.error(f"❌ Erreur fetching {pair}: {e}")
            raise
    
    def validate_data(self, df: pd.DataFrame, pair: str) -> bool:
        """Valide que les données sont complètes (pas de trous)."""
        required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            logger.error(f"Colonnes manquantes pour {pair}")
            return False
        
        if df['close'].isnull().any():
            logger.error(f"Valeurs NULL dans {pair}")
            return False
        
        logger.info(f"✅ Data validée pour {pair}")
        return True
```

**À ÉVITER** ❌ :
```python
# ❌ Pas de gestion d'erreur
def fetch_data(pair, start, end):
    candles = exchange.fetch_ohlcv(pair, '4h', start)
    return pd.DataFrame(candles)

# ❌ Pas de limitation de rate
# Binance : max 1200 requests / min = 20 req/sec
for pair in 100_pairs:
    fetch_ohlcv(pair, ...)  # BLOCAGE API garanti

# ❌ Pas de validation
df = fetch_data(...)
# df peut avoir 300 lignes au lieu de 500, personne le sait
```

### Implémentation Stratégies

**À FAIRE** ✅ :
```python
# src/strategies.py - Exemple complet avec 3 stratégies

class MeanReversion(Strategy):
    """Stratégie Mean Reversion basée sur RSI."""
    
    def __init__(self, rsi_period: int = 14, oversold: int = 30, overbought: int = 70):
        super().__init__("mean_reversion")
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['rsi'] = ta.momentum.rsi(df['close'], self.rsi_period)
        
        # Oversold (RSI < 30) = BUY
        # Overbought (RSI > 70) = SELL
        df['signal'] = np.where(
            df['rsi'] < self.oversold,
            1,  # BUY signal
            np.where(df['rsi'] > self.overbought, -1, 0)  # SELL signal
        )
        
        return df


class Momentum(Strategy):
    """Stratégie Momentum basée sur MACD."""
    
    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        super().__init__("momentum")
        self.fast = fast
        self.slow = slow
        self.signal_period = signal
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        macd = ta.trend.macd(df['close'], self.fast, self.slow, self.signal_period)
        df['macd'] = macd
        df['macd_signal'] = ta.trend.macd_signal(df['close'], self.fast, self.slow, self.signal_period)
        
        # MACD > signal = BUY
        df['signal'] = np.where(df['macd'] > df['macd_signal'], 1, -1)
        
        return df
```

### Backtest & Logging

**À FAIRE** ✅ :
```python
# src/backtest_engine.py
import vectorbt as vbt
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class VectorBTBacktester:
    """Engine de backtest vectorisé avec VectorBT."""
    
    def __init__(self, initial_capital: float = 2000, fees: float = 0.001):
        """
        Initialize VectorBT Backtest Engine.
        
        Args:
            initial_capital (float): Capital initial en USDT
            fees (float): Frais de trading (ex: 0.001 = 0.1%)
        """
        self.initial_capital = initial_capital
        self.fees = fees
    
    def backtest(
        self,
        df: pd.DataFrame,
        strategy_name: str,
        pair: str
    ) -> Dict:
        """
        Exécute un backtest avec VectorBT.
        
        Args:
            df (pd.DataFrame): DataFrame avec colonnes [close, signal]
            strategy_name (str): Nom de la stratégie
            pair (str): Paire tradée (ex: BTCUSDT)
        
        Returns:
            Dict: {total_return, sharpe_ratio, max_drawdown, win_rate, equity_curve, trades}
        """
        try:
            # Entrées = signal BUY, Sorties = signal SELL
            entries = df['signal'] == 1
            exits = df['signal'] == -1
            
            # Portfolio VectorBT
            portfolio = vbt.Portfolio.from_signals(
                close=df['close'],
                entries=entries,
                exits=exits,
                init_cash=self.initial_capital,
                fees=self.fees,
                freq='4h'  # 4 heures
            )
            
            total_return = float(portfolio.total_return()) * 100
            sharpe_ratio = float(portfolio.sharpe_ratio())
            max_drawdown = float(portfolio.max_drawdown()) * 100
            win_rate = float(portfolio.trades.win_rate.dropna().mean()) * 100 if len(portfolio.trades) > 0 else 0
            
            results = {
                'strategy': strategy_name,
                'pair': pair,
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'num_trades': len(portfolio.trades),
                'portfolio': portfolio,
                'equity_curve': portfolio.value()
            }
            
            logger.info(
                f"✅ Backtest {strategy_name} ({pair}): "
                f"Return={total_return:.2f}%, Sharpe={sharpe_ratio:.2f}, DD={max_drawdown:.2f}%"
            )
            
            return results
        
        except Exception as e:
            logger.error(f"❌ Erreur backtest {strategy_name}: {e}")
            raise


def export_backtest_results(results: Dict, output_dir: str) -> str:
    """Exporte les résultats du backtest en CSV."""
    df_results = pd.DataFrame([{
        'strategy': results['strategy'],
        'pair': results['pair'],
        'total_return': results['total_return'],
        'sharpe_ratio': results['sharpe_ratio'],
        'max_drawdown': results['max_drawdown'],
        'win_rate': results['win_rate'],
        'num_trades': results['num_trades']
    }])
    
    filename = f"{output_dir}/{results['pair']}_{results['strategy']}_backtest.csv"
    df_results.to_csv(filename, index=False)
    logger.info(f"✅ Backtest exporté : {filename}")
    return filename
```

---

## Phase 2 : OPTIMISATION

### Anti-overfitting : Walk-Forward Analysis

**À FAIRE** ✅ :
```python
# src/optimizer.py
def walk_forward_validation(
    data: pd.DataFrame,
    strategy_class: type,
    param_grid: Dict[str, List],
    train_size: float = 0.7,
    test_size: float = 0.3
) -> Dict:
    """
    Walk-forward analysis pour valider sans overfitting.
    
    Divise les données en fenêtres disjointes :
    - Train (70%) : optimise les paramètres
    - Test (30%) : valide les résultats (NO OPTIMIZATION HERE)
    """
    n = len(data)
    train_end = int(n * train_size)
    
    train_data = data.iloc[:train_end]
    test_data = data.iloc[train_end:]
    
    # Phase 1 : Grid search sur TRAIN uniquement
    best_params = None
    best_sharpe = -np.inf
    
    for params_combo in generate_param_combinations(param_grid):
        strategy = strategy_class(**params_combo)
        train_data_with_signals = strategy.generate_signals(train_data)
        backtest_results = backtest(train_data_with_signals)
        
        if backtest_results['sharpe_ratio'] > best_sharpe:
            best_sharpe = backtest_results['sharpe_ratio']
            best_params = params_combo
    
    logger.info(f"✅ Meilleurs params (TRAIN) : {best_params}, Sharpe={best_sharpe:.2f}")
    
    # Phase 2 : Valide sur TEST (sans optimiser)
    strategy = strategy_class(**best_params)
    test_data_with_signals = strategy.generate_signals(test_data)
    test_results = backtest(test_data_with_signals)
    
    logger.info(f"Validation (TEST) : Sharpe={test_results['sharpe_ratio']:.2f}")
    
    # Vérifie pas d'overfitting
    sharpe_diff = abs(best_sharpe - test_results['sharpe_ratio'])
    if sharpe_diff > 1.0:
        logger.warning(f"⚠️ POSSIBLE OVERFITTING : Sharpe train={best_sharpe:.2f} vs test={test_results['sharpe_ratio']:.2f}")
    
    return {
        'best_params': best_params,
        'train_results': {'sharpe': best_sharpe},
        'test_results': test_results,
        'overfitting_detected': sharpe_diff > 1.0
    }
```

**À ÉVITER** ❌ :
```python
# ❌ JAMAIS : Optimiser sur TOUS les données
best_params = None
for params in param_combinations:
    strategy = Strategy(**params)
    results = backtest(full_data, strategy)  # ❌ Overfitting 100% garanti
    if results['sharpe'] > best_sharpe:
        best_params = params

# ❌ JAMAIS : Trop de paramètres
param_grid = {
    'sma_short': range(5, 100),  # 95 combos
    'sma_long': range(100, 300),  # 200 combos
    'rsi_period': range(10, 30),  # 20 combos
    'position_size': [0.01, 0.02, 0.03],  # 3 combos
}
# Total : 95 * 200 * 20 * 3 = 1.14 MILLION combos = overfitting extrême
```

---

## Phase 3 : PAPER TRADING

### Configuration Testnet

**À FAIRE** ✅ :
```python
# config/backtest_config.json
{
  "pairs": ["BTCUSDT", "ETHUSDT", "BNBUSDT"],
  "timeframe": "4h",
  "initial_capital": 2000,
  "fees": 0.001,
  "max_position_size": 0.02,
  "stop_loss_percent": 5,
  "take_profit_percent": 10,
  "paper_trading": true,
  "testnet_enabled": true
}

# Démarrage bot papier
from src.live_bot import LiveBot

config = load_config("config/backtest_config.json")
bot = LiveBot(
    strategy=my_optimized_strategy,
    pairs=config['pairs'],
    paper_trading=True,  # ✅ TOUJOURS True pour paper
    testnet=config['testnet_enabled']
)
bot.run()
```

### Logging des trades papier

**À FAIRE** ✅ :
```python
# src/logger.py
class TradeLogger:
    """Logger les trades en CSV pour analyse."""
    
    def log_trade(
        self,
        pair: str,
        signal_type: str,
        entry_price: float,
        exit_price: Optional[float] = None,
        pnl_usdt: Optional[float] = None,
        status: str = "OPEN"
    ):
        """Log un trade en CSV."""
        timestamp = datetime.now()
        filename = f"data/live_trading/{pair}_paper_{timestamp.strftime('%Y%m%d')}.csv"
        
        trade_record = {
            'timestamp': timestamp,
            'pair': pair,
            'signal': signal_type,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl_usdt': pnl_usdt,
            'pnl_percent': (pnl_usdt / entry_price * 100) if pnl_usdt else None,
            'status': status
        }
        
        if os.path.exists(filename):
            df = pd.read_csv(filename)
            df = pd.concat([df, pd.DataFrame([trade_record])], ignore_index=True)
        else:
            df = pd.DataFrame([trade_record])
        
        df.to_csv(filename, index=False)
        logger.info(f"✅ Trade logged : {pair} {signal_type} @ {entry_price}")
```

---

## Testing avec Pytest

### Structure tests

```
tests/
├── __init__.py
├── test_strategies.py
├── test_backtest_engine.py
├── test_data_fetcher.py
├── test_logger.py
└── conftest.py  # Fixtures communes
```

### Exemple tests

**À FAIRE** ✅ :
```python
# tests/test_strategies.py
import pytest
import pandas as pd
import numpy as np
from src.strategies import TrendFollowing, MeanReversion, Momentum


@pytest.fixture
def sample_data():
    """Fixture : données OHLCV de test."""
    dates = pd.date_range('2024-01-01', periods=100, freq='4h')
    df = pd.DataFrame({
        'timestamp': dates,
        'open': np.random.uniform(40000, 45000, 100),
        'high': np.random.uniform(40000, 45000, 100),
        'low': np.random.uniform(40000, 45000, 100),
        'close': np.random.uniform(40000, 45000, 100),
        'volume': np.random.uniform(10, 1000, 100)
    })
    return df.sort_values('close').reset_index(drop=True)


class TestTrendFollowing:
    """Tests pour stratégie Trend Following."""
    
    def test_init_valid_params(self):
        """Test : initialisation avec params valides."""
        strategy = TrendFollowing(sma_short=20, sma_long=50)
        assert strategy.sma_short == 20
        assert strategy.sma_long == 50
    
    def test_init_invalid_params(self):
        """Test : levée erreur si sma_short >= sma_long."""
        with pytest.raises(ValueError):
            TrendFollowing(sma_short=50, sma_long=20)
    
    def test_generate_signals_output_shape(self, sample_data):
        """Test : output a même nombre de lignes que input."""
        strategy = TrendFollowing()
        result = strategy.generate_signals(sample_data)
        assert len(result) == len(sample_data)
    
    def test_generate_signals_contains_signal_column(self, sample_data):
        """Test : output contient colonne 'signal'."""
        strategy = TrendFollowing()
        result = strategy.generate_signals(sample_data)
        assert 'signal' in result.columns
    
    def test_generate_signals_valid_values(self, sample_data):
        """Test : signal ne contient que -1, 0, 1."""
        strategy = TrendFollowing()
        result = strategy.generate_signals(sample_data)
        assert set(result['signal'].unique()).issubset({-1, 0, 1})
    
    def test_missing_close_column_raises_error(self):
        """Test : erreur si colonne 'close' manquante."""
        strategy = TrendFollowing()
        bad_df = pd.DataFrame({'open': [1, 2, 3]})
        with pytest.raises(ValueError):
            strategy.generate_signals(bad_df)


class TestMeanReversion:
    """Tests pour stratégie Mean Reversion."""
    
    def test_oversold_generates_buy_signal(self, sample_data):
        """Test : RSI < oversold = BUY signal."""
        # Crée données avec RSI bas (survendu)
        sample_data['close'] = [i for i in range(100, 0, -1)]  # Descend
        
        strategy = MeanReversion(rsi_period=14, oversold=30, overbought=70)
        result = strategy.generate_signals(sample_data)
        
        # Dernières valeurs devraient être BUY (RSI bas)
        assert 1 in result['signal'].tail(10).values
    
    def test_overbought_generates_sell_signal(self, sample_data):
        """Test : RSI > overbought = SELL signal."""
        # Crée données avec RSI haut (suracheté)
        sample_data['close'] = [i for i in range(1, 101)]  # Monte
        
        strategy = MeanReversion(rsi_period=14, oversold=30, overbought=70)
        result = strategy.generate_signals(sample_data)
        
        # Dernières valeurs devraient être SELL (RSI haut)
        assert -1 in result['signal'].tail(10).values


# tests/test_backtest_engine.py
from src.backtest_engine import VectorBTBacktester


class TestVectorBTBacktester:
    """Tests pour VectorBT Backtest Engine."""
    
    def test_backtest_returns_valid_dict(self, sample_data):
        """Test : backtest retourne dict avec toutes les clés."""
        sample_data['signal'] = np.where(sample_data['close'] > sample_data['close'].mean(), 1, -1)
        
        backtest = VectorBTBacktester(initial_capital=2000)
        result = backtest.backtest(sample_data, "test_strategy", "BTCUSDT")
        
        required_keys = ['strategy', 'total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate']
        assert all(key in result for key in required_keys)
    
    def test_backtest_results_are_numeric(self, sample_data):
        """Test : tous les résultats sont numériques."""
        sample_data['signal'] = np.where(sample_data['close'] > sample_data['close'].mean(), 1, -1)
        
        backtest = VectorBTBacktester()
        result = backtest.backtest(sample_data, "test_strategy", "BTCUSDT")
        
        assert isinstance(result['total_return'], float)
        assert isinstance(result['sharpe_ratio'], float)
        assert isinstance(result['max_drawdown'], float)
        assert isinstance(result['win_rate'], float)
    
    def test_fees_affect_returns(self, sample_data):
        """Test : frais plus élevés = returns plus bas."""
        sample_data['signal'] = np.where(sample_data['close'] > sample_data['close'].mean(), 1, -1)
        
        backtest_low_fees = VectorBTBacktester(fees=0.0001)
        backtest_high_fees = VectorBTBacktester(fees=0.01)
        
        result_low = backtest_low_fees.backtest(sample_data, "test", "BTCUSDT")
        result_high = backtest_high_fees.backtest(sample_data, "test", "BTCUSDT")
        
        # Frais plus élevés = return plus bas
        assert result_low['total_return'] >= result_high['total_return']
```

**Exécution des tests** :
```bash
# Tous les tests
pytest tests/

# Tests spécifiques
pytest tests/test_strategies.py -v

# Avec couverture de code
pytest --cov=src tests/

# Tests en parallèle (rapide)
pytest -n auto tests/
```

---

## Logging & Monitoring

**À FAIRE** ✅ :
```python
# src/logger.py
import logging
from datetime import datetime
import os

def setup_logging(log_dir: str = "logs/") -> logging.Logger:
    """Setup logging pour l'application."""
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger("trading_bot")
    logger.setLevel(logging.DEBUG)
    
    # File handler
    fh = logging.FileHandler(f"{log_dir}/trading_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    fh.setLevel(logging.DEBUG)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


# Utilisation
logger = setup_logging()

# À chaque étape importante
logger.info("Application démarrée")
logger.debug("Config chargée : 3 paires, 4h timeframe")
logger.warning("Position non fermée depuis 24h : ETHUSDT")
logger.error("Erreur API Binance : Connection timeout")
```

---

## Git & Versioning

### .gitignore obligatoire

```
# Secrets & Env
.env
.env.local
config/.secrets.json

# Données sensibles
data/live_trading/*.csv
data/backtest_results/*.csv

# Logs
logs/
*.log

# Python
__pycache__/
*.py[cod]
*$py.class
.pytest_cache/
.coverage
htmlcov/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db
```

### Commits structure

**À FAIRE** ✅ :
```bash
# Commits clairs et atomiques
git commit -m "feat: ajouter stratégie Mean Reversion avec tests"
git commit -m "fix: corriger look-ahead bias dans backtest engine"
git commit -m "docs: mettre à jour README avec instructions setup"

# Évite les gros commits
git commit -m "trucs" ❌
```

### Branches

```bash
# Main : toujours stable
git checkout main

# Development : branche de travail
git checkout -b feature/new-strategy
git commit -m "feat: implémenter MACD strategy"
git push origin feature/new-strategy

# Pull request & review avant merge
```

---

## Checklist Intégrateur IA

### Avant de demander à une IA de coder

**Copie-colle cette section dans ta prompt pour l'IA** :

```
Bienvenue ! Tu aides au projet "Crypto Swing Trading Bot".

📋 CONSIGNES ESSENTIELLES :

1. **PEP 8 STRICTEMENT** :
   - Noms : snake_case (sauf classes = CamelCase)
   - Max 79 caractères par ligne
   - Docstrings pour chaque fonction/classe
   - Type hints obligatoires

2. **JAMAIS d'argent réel** :
   - Si paper_trading=False ET c'est du code live → ERREUR
   - Tous les tests doivent être sur Testnet/historique

3. **Anti-overfitting** :
   - Walk-forward validation obligatoire pour optimisation
   - Jamais optimiser sur 100% des données
   - Valide que Sharpe Ratio reste stable entre train/test

4. **Logging** :
   - Chaque trade = log INFO avec timestamp
   - Chaque erreur = log ERROR avec stack trace
   - CSV export pour chaque backtest

5. **Tests** :
   - pytest obligatoire pour tout code
   - Couverture > 80%
   - Test les cas normaux ET edge cases

6. **Sécurité** :
   - JAMAIS de clés API en dur
   - Utiliser .env + os.getenv()
   - Pas de données sensibles en commits

7. **Code structure** :
   - Classes pour strategies (héritage de Strategy)
   - Séparation data_fetcher / strategies / backtest_engine
   - Fonctions < 50 lignes

8. **Gestion d'erreur** :
   - try/except avec logging
   - Valide inputs avant traitement
   - Levée ValueError pour params invalides

Si tu dois coder :
- Fournis TOUT le code (pas de "..."), pas de pseudo-code
- Inclus imports, docstrings, type hints, logs
- Termine avec les tests pytest
- Mentionne PEP 8 compliance dans ta réponse
```

---

## Exemples à NE PAS faire

### ❌ Anti-Pattern #1 : Hardcoding

```python
# ❌ JAMAIS
API_KEY = "sk_live_abc123..."  
capital = 2000
fees = 0.001
```

### ❌ Anti-Pattern #2 : Pas de validation

```python
# ❌ JAMAIS
def backtest(data, strategy):
    results = vbt.backtest(data, ...)  # data peut être vide, wrong shape
    return results
```

### ❌ Anti-Pattern #3 : Look-ahead Bias

```python
# ❌ JAMAIS - Utilise données futures !
signal = np.where(df['close'].shift(-1) > df['close'], 1, -1)  # Triche !
```

### ❌ Anti-Pattern #4 : Pas de tests

```python
# ❌ JAMAIS
# Code en production sans aucun test
if __name__ == "__main__":
    bot.run()
```

---

## Ressources & Références

| Sujet | Lien |
|-------|------|
| **PEP 8** | https://pep8.org/ |
| **VectorBT** | https://vectorbt.pro/ |
| **Pytest** | https://docs.pytest.org/ |
| **Binance API** | https://binance-docs.github.io/apidocs/ |
| **TA (Technical Analysis)** | https://github.com/bukosabino/ta |
| **Python Typing** | https://docs.python.org/3/library/typing.html |
| **Walk-Forward Analysis** | https://en.wikipedia.org/wiki/Walk_forward_optimization |

---

**Dernière mise à jour : 2025-01-22**  
**Version : 1.0**