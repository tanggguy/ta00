# Architecture Complète : Projet Trading Algorithmique Crypto Swing

## 📋 Vue d'ensemble

**Objectif** : Système de trading swing crypto (4h/daily) en Python, avec backtest → optimisation → paper trading → live, avec dashboard + logs CSV.

**Stack** :
- **Backtest & Optimisation** : VectorBT (ultra-rapide, vectorisé)
- **Exécution Live/Paper** : Code maison simple + API Binance
- **Données** : Binance API (gratuit, 1000+ paires)
- **Logging** : CSV + SQLite (simple et puissant)
- **Dashboard** : Streamlit (simple à faire, très visuel, parfait pour un étudiant)
- **Infrastructure** : PC Windows + tâches planifiées, évolution vers RPi/cloud après

---

## 📁 Structure du projet

```
crypto-swing-bot/
│
├── data/
│   ├── raw/                    # Données brutes Binance
│   │   ├── BTCUSDT_1h.csv
│   │   ├── ETHUSDT_1h.csv
│   │   └── ...
│   ├── processed/              # Données transformées
│   └── backtest_results/       # Résultats backtest (JSON/Pickle)
│
├── src/
│   ├── __init__.py
│   ├── data_fetcher.py         # Récupération données Binance
│   ├── strategies.py           # Logique des stratégies (trend, mean reversion, etc.)
│   ├── backtest_engine.py      # Engine vectorisé avec VectorBT
│   ├── optimizer.py            # Optimisation des paramètres
│   ├── live_bot.py             # Exécution live/paper
│   ├── logger.py               # Logging CSV/SQLite
│   ├── risk_manager.py         # Gestion du risque (position sizing, stops)
│   └── utils.py                # Fonctions utilitaires
│
├── dashboard/
│   ├── app.py                  # Streamlit app principal reserve a la visualisation et creation de rapports
│   ├── pages/                  # Pages Streamlit
│   │   ├── 01_backtest_results.py
│   │   ├── 02_comparison.py
│   │   ├── 03_csv_explorer.py
│   │   └── 04_risk_analysis.py
│   │   ├── 05_backtest_results.py
│   │   ├── 06_live_monitoring.py
│   └── logs/                   # Fichiers log exportés
│
├── config/
│   ├── strategies.json         # Config stratégies (paramètres)
│   ├── risk.json               # Config risque (stop loss, position size)
│   ├── pairs.json              # Paires à trader
│   └── backtest_config.json    # Config backtest (dates, timeframe)
│
├── scripts/
│   ├── fetch_historical_data.py    # Script pour récupérer l'historique
│   ├── run_backtest.py             # Lancer backtest complet
│   ├── run_optimizer.py            # Optimiser paramètres
│   ├── run_live_bot.py             # Démarrer le bot live/paper
│   └── schedule_bot.py             # Scheduler cron-like pour Windows
│
├── tests/
│   ├── test_strategies.py
│   ├── test_backtest.py
│   └── test_logger.py
│
├── requirements.txt
├── README.md
└── .env.example                # Variables d'env (clés API, etc.)
```

---

## 🔄 Workflow : 3 phases

### Phase 1 : BACKTEST (Semaines 1-4)

**Objectif** : Valider que ta stratégie fonctionne sur l'historique.

**Étapes** :
1. Récupérer données historiques Binance (BTCUSDT, ETHUSDT, etc., 1-2 ans)
2. Implémenter stratégies simples :
   - **Trend Following** : SMA croisement (ex: SMA 20/50)
   - **Mean Reversion** : RSI survendu/suracheté
   - **Momentum** : MACD
3. Backtest vectorisé (VectorBT) : rapide, teste 1000 combos de params en secondes
4. Logs backtest → fichiers CSV :
   - `backtest_BTCUSDT_20240101_20250101.csv` avec colonnes : date, signal, prix, PnL, drawdown, Sharpe

**Output** :
- Meilleure stratégie + params optimaux identifiés
- Fichiers CSV des résultats
-Dashboard Streamlit page backtest
---

### Phase 2 : OPTIMISATION (Semaines 4-8)

**Objectif** : Trouver les meilleurs paramètres sans overfitting.

**Étapes** :
1. Grid search / Bayesian optimization sur les paramètres clés
   - Ex: SMA court [10-50], SMA long [50-200], position size [1-3%]
2. Validation croisée (walk-forward analysis) pour éviter l'overfitting
3. Exporter résultats d'optimisation en html
4. Comparer performance sur différentes paires
5. Définir règles de risque (stop loss, take profit, max position size)

**Output** :
- Config optimale en JSON (stratégies + paramètres)
- CSV détaillé d'optimisation (tous les tests, metrics)
- Rapports walk-forward montrant stabilité

---
2.1. Données Alternatives & Sentiment (Gratuit)
Objectif : Filtrer les signaux techniques par le contexte global du marché.

Source CryptoPanic (API) :

Implémenter un script src/data_sentiment.py pour récupérer le score de "votes" (bullish vs bearish) sur les dernières news.

Règle métier : Si le sentiment est > 70% bearish, interdiction d'ouvrir une position "BUY", même si le RSI est survendu.

Filtre de Volatilité Google Trends :

Utiliser la bibliothèque pytrends pour mesurer l'intérêt sur le mot "Bitcoin".

Une hausse soudaine de l'intérêt sert de "confirmation" pour une stratégie de momentum.

2.2. Machine Learning "Light" (Scikit-Learn)
Objectif : Créer un modèle de classification pour valider la qualité d'un signal.

Modèle : RandomForestClassifier ou XGBoost (via Scikit-Learn).

Features (Variables d'entrée) :

Technique : RSI, MACD, écart à la SMA.

Temps : Heure du signal (pour capturer les sessions US/Asie).

Sentiment : Score CryptoPanic du moment.

Target (Cible) : Le trade a-t-il atteint +1% avant de toucher un stop à -0.5% ? (Binaire : 1 ou 0).

Intégration : Le bot génère un signal (ex: Momentum), le modèle ML donne son "accord". Si la probabilité de succès est < 60%, le trade est ignoré.

2.3. Infrastructure & Optimisation Bayésienne
Objectif : Remplacer le Grid Search de src/optimizer.py par une recherche intelligente.

Outil : Optuna (Bibliothèque Python gratuite).

Plan de transfert :

Définir l'Objectif : Créer une fonction objective(trial) qui prend des paramètres suggérés par Optuna, lance un backtest rapide, et retourne le ratio de Sharpe.

Espace de recherche : Au lieu d'une liste fixe dans param_grid, définir des plages (ex: trial.suggest_int('sma_short', 10, 50)).

Élagage (Pruning) : Configurer Optuna pour arrêter immédiatement les tests de paramètres qui performent très mal dès les premières étapes du backtest (gain de temps énorme).

Stockage : Les résultats sont sauvegardés dans un fichier SQLite local (gratuit) pour pouvoir reprendre l'optimisation plus tard.
### Phase 3 : PAPER TRADING (Semaines 8-12)

**Objectif** : Valider que ça marche "en réel" (sans argent).

**Étapes** :
1. Connecter bot au **Binance Testnet** ou faire du suivi manuel
2. Exécuter les signaux générés par la stratégie optimisée
3. Logger chaque signal : timestamp, paire, direction, prix d'entrée, sortie, PnL réalisé
4. Comparer backtest vs paper trading réel (slippage, frais, exécution)
5. Dashboard Streamlit en temps réel affichant :
   - Courbe d'équité paper trading
   - Drawdown, Sharpe, Win rate
   - Derniers trades
   - CSV téléchargeable

**Output** :
- CSV de trades papier (date, paire, prix, PnL, etc.)
- Dashboard Streamlit actif
- Décision : OK pour live ou ajustements nécessaires

---

## 📊 Logging & Dashboard

### Structure CSV

**`backtest_results/BTCUSDT_backtest_20250101_20250120.csv`** :
```
timestamp,open,close,signal,position,entry_price,exit_price,pnl_percent,cumulative_pnl,drawdown,sharpe,win_count,loss_count
2024-01-01 00:00,42000,42100,BUY,1.0,42100,42500,0.95,0.95,0.0,1.5,1,0
2024-01-05 04:00,42500,42300,SELL,0.0,42500,42300,-0.47,0.48,0.0,1.3,1,1
...
```

**`live_trading/BTCUSDT_live_20250122.csv`** :
```
timestamp,paire,signal,prix_entree,prix_sortie,pnl_usdt,pnl_percent,status,notes
2025-01-22 14:30:00,BTCUSDT,BUY,43500.00,,,,OPEN,Waiting for exit signal
2025-01-22 18:45:00,BTCUSDT,SELL,43500.00,43800.00,300.00,0.69,CLOSED,Stop hit
...
```

**`optimization_results/optimization_sweep.csv`** :
```
strategy,sma_short,sma_long,position_size,total_return,sharpe,max_drawdown,win_rate,trades_count
trend_following,20,50,1.0,45.2,1.8,-12.3,58,234
trend_following,20,50,2.0,68.5,1.2,-18.5,60,234
trend_following,20,60,1.0,42.1,1.7,-11.2,55,198
...
```

### Dashboard Streamlit

**Pages principales** :

1. **Backtest Results**
   - Sélecteur : stratégie + paire + dates
   - Graphe : courbe d'équité + drawdown
   - Tableau : statistiques (Sharpe, Win Rate, Profit Factor)
   - Bouton : télécharger CSV correspondant
2. **Comparison**
   compare les dernier backtest

4. **CSV Explorer**
   - Upload/browse fichiers CSV (backtest ou live)
   - Affiche table interactive
   - Filtrage par date, paire, status
   - Télécharge données filtrées
5. **Optimization Results**
   - Sélecteur : stratégie + paire + dates
   - Graphe : courbe d'équité + drawdown
   - Tableau : statistiques (Sharpe, Win Rate, Profit Factor)
   - Bouton : télécharger html correspondant

6. **Risk Analysis**
   - Heatmap de correlation entre paires
   - Visualisation max drawdown par stratégie
   - Scenario analysis (what-if)
  
7. **Live Monitoring** (une fois en paper/live)
   - Positions ouvertes actuelles
   - Derniers trades fermés (PnL)
   - Courbe d'équité paper trading
   - Comparaison vs backtest (écart %)
---

## 💻 Implémentation : Stack technique détaillé

### Requirements.txt

```
pandas==2.0.3
numpy==1.24.3
vectorbt==0.25.0
binance==0.1.0               # SDK Binance
ccxt==2.0.0                  # Crypto exchanges
ta==0.10.2                   # Technical analysis
python-dotenv==1.0.0
streamlit==1.28.0
plotly==5.17.0
SQLAlchemy==2.0.21
requests==2.31.0
pyyaml==6.0
```

### Modules clés

#### 1. `data_fetcher.py` : Récupérer les données

```python
import ccxt
import pandas as pd
from datetime import datetime, timedelta

class BinanceDataFetcher:
    def __init__(self):
        self.exchange = ccxt.binance()
    
    def fetch_ohlcv(self, pair, timeframe, start_date, end_date):
        """
        Récupère OHLCV historique pour une paire
        pair: 'BTC/USDT'
        timeframe: '4h' ou '1d'
        """
        # Logique : appelle exchange.fetch_ohlcv en boucles (limite 500 par call)
        # Retourne DataFrame avec colonnes : timestamp, open, high, low, close, volume
        pass
    
    def fetch_live(self, pair, timeframe, limit=500):
        """Récupère les N dernières bougies"""
        pass
```

#### 2. `strategies.py` : Logique stratégies

```python
import ta  # Technical Analysis library
import numpy as np

class TrendFollowing:
    def __init__(self, sma_short=20, sma_long=50):
        self.sma_short = sma_short
        self.sma_long = sma_long
    
    def generate_signals(self, df):
        """
        df : DataFrame avec colonnes close
        Retourne colonne 'signal' : 1 (BUY), -1 (SELL), 0 (HOLD)
        """
        df['sma_short'] = ta.trend.sma_indicator(df['close'], self.sma_short)
        df['sma_long'] = ta.trend.sma_indicator(df['close'], self.sma_long)
        df['signal'] = np.where(df['sma_short'] > df['sma_long'], 1, -1)
        return df

class MeanReversion:
    def __init__(self, rsi_period=14, oversold=30, overbought=70):
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought
    
    def generate_signals(self, df):
        df['rsi'] = ta.momentum.rsi(df['close'], self.rsi_period)
        df['signal'] = np.where(df['rsi'] < self.oversold, 1,
                                np.where(df['rsi'] > self.overbought, -1, 0))
        return df
```

#### 3. `backtest_engine.py` : Backtest vectorisé

```python
import vectorbt as vbt
import pandas as pd

class VectorBTBacktester:
    def __init__(self, initial_capital=2000, fees=0.001):
        self.initial_capital = initial_capital
        self.fees = fees  # 0.1% frais Binance
    
    def backtest(self, df, signal_column):
        """
        df : DataFrame avec colonne 'signal'
        Retourne stats et courbe d'équité
        """
        # Utilise vbt.Portfolio pour simuler
        entries = df['signal'] == 1
        exits = df['signal'] == -1
        
        portfolio = vbt.Portfolio.from_signals(
            close=df['close'],
            entries=entries,
            exits=exits,
            init_cash=self.initial_capital,
            fees=self.fees
        )
        
        return {
            'portfolio': portfolio,
            'total_return': portfolio.total_return(),
            'sharpe_ratio': portfolio.sharpe_ratio(),
            'max_drawdown': portfolio.max_drawdown(),
            'win_rate': portfolio.trades.win_rate.dropna().mean(),
            'equity_curve': portfolio.value()
        }
```

#### 4. `logger.py` : Logging des trades

```python
import pandas as pd
from datetime import datetime
import os

class TradeLogger:
    def __init__(self, log_dir='live_trading'):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
    
    def log_trade(self, pair, signal, entry_price, exit_price=None, pnl=None):
        """Log un trade en CSV"""
        timestamp = datetime.now()
        filename = f"{self.log_dir}/{pair}_live_{timestamp.date()}.csv"
        
        trade = {
            'timestamp': timestamp,
            'pair': pair,
            'signal': signal,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl_usdt': pnl,
            'pnl_percent': (pnl / entry_price * 100) if pnl else None,
            'status': 'OPEN' if exit_price is None else 'CLOSED'
        }
        
        if os.path.exists(filename):
            df = pd.read_csv(filename)
            df = pd.concat([df, pd.DataFrame([trade])], ignore_index=True)
        else:
            df = pd.DataFrame([trade])
        
        df.to_csv(filename, index=False)
```

#### 5. `live_bot.py` : Exécution live/paper

```python
import ccxt
from datetime import datetime, timedelta
import time

class LiveBot:
    def __init__(self, strategy, pairs, paper_trading=True):
        self.exchange = ccxt.binance()
        self.strategy = strategy
        self.pairs = pairs
        self.paper_trading = paper_trading
        self.positions = {}  # Suivi des positions ouvertes
    
    def run(self):
        """Boucle principale du bot"""
        while True:
            for pair in self.pairs:
                # Récupère dernières bougies
                df = self.fetch_latest(pair)
                
                # Génère signal
                df = self.strategy.generate_signals(df)
                signal = df['signal'].iloc[-1]
                
                # Exécute (ou simule)
                if signal == 1 and pair not in self.positions:
                    self.buy(pair, df['close'].iloc[-1])
                elif signal == -1 and pair in self.positions:
                    self.sell(pair, df['close'].iloc[-1])
            
            # Attend 4 heures avant prochaine vérification (swing 4h)
            time.sleep(4 * 3600)
    
    def buy(self, pair, price):
        if self.paper_trading:
            print(f"[PAPER] BUY {pair} @ {price}")
        else:
            # Passe ordre réel via API
            self.exchange.create_limit_buy_order(pair, amount, price)
        self.positions[pair] = price
    
    def sell(self, pair, price):
        if self.paper_trading:
            pnl = (price - self.positions[pair]) / self.positions[pair] * 100
            print(f"[PAPER] SELL {pair} @ {price}, PnL: {pnl:.2f}%")
        else:
            # Passe ordre réel
            self.exchange.create_limit_sell_order(pair, amount, price)
        del self.positions[pair]
```

#### 6. Dashboard Streamlit (`dashboard/app.py`)

```python
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os

st.set_page_config(page_title="Trading Dashboard", layout="wide")

st.title("🚀 Crypto Swing Trading Dashboard")

# Sidebar navigation
page = st.sidebar.radio("Navigation", 
    ["📊 Backtest Results", "📈 Live Monitoring", "📂 CSV Explorer", "⚠️ Risk Analysis"])

if page == "📊 Backtest Results":
    st.header("Backtest Results")
    
    # Selecteur fichier
    csv_files = [f for f in os.listdir('data/backtest_results/') if f.endswith('.csv')]
    selected_file = st.selectbox("Sélectionner backtest", csv_files)
    
    # Charge et affiche
    df = pd.read_csv(f'data/backtest_results/{selected_file}')
    
    # Graphe équité
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['cumulative_pnl'], 
                             mode='lines', name='Equity'))
    st.plotly_chart(fig, use_container_width=True)
    
    # Stats
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Return", f"{df['cumulative_pnl'].iloc[-1]:.2f}%")
    col2.metric("Max Drawdown", f"{df['drawdown'].min():.2f}%")
    col3.metric("Sharpe Ratio", f"{df['sharpe'].mean():.2f}")
    col4.metric("Win Rate", f"{(df['win_count'].sum() / (df['win_count'].sum() + df['loss_count'].sum()) * 100):.1f}%")
    
    # Tableau détaillé
    st.dataframe(df, use_container_width=True)
    
    # Download
    st.download_button("Télécharger CSV", df.to_csv(index=False), "backtest.csv")

elif page == "📂 CSV Explorer":
    st.header("CSV Explorer")
    uploaded_file = st.file_uploader("Upload CSV", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df, use_container_width=True)
        st.download_button("Télécharger", df.to_csv(index=False), "export.csv")
```

---

## ⏱️ Exécution sur Windows (tâches planifiées)

### Script `scripts/schedule_bot.py`

```python
import schedule
import time
from src.live_bot import LiveBot

def job():
    print(f"[{datetime.now()}] Bot execution started")
    bot = LiveBot(strategy=config['strategy'], pairs=config['pairs'])
    bot.run_once()  # Exécute une itération
    print("[DONE] Bot execution completed")

# Planifie l'exécution toutes les 4 heures
schedule.every(4).hours.do(job)

while True:
    schedule.run_pending()
    time.sleep(60)
```

**Ou avec Windows Task Scheduler** (GUI Windows) :
- Crée une tâche planifiée qui exécute `python scripts/schedule_bot.py`
- Fréquence : toutes les 4 heures
- Démarre au boot du PC

---

## 📍 Chemin critique sur 3 mois

| Semaine | Phase                   | Tâches                                                                   |
| ------- | ----------------------- | ------------------------------------------------------------------------ |
| 1-2     | Setup + Data            | Cloner repo, installer dépendances, récupérer 2 ans d'historique BTCUSDT |
| 3-4     | Backtest v1             | Implémenter 3 stratégies simples, backtest séparé, logs CSV              |
| 5-6     | Optimisation            | Grid search sur les 3 stratégies, walk-forward analysis                  |
| 7-8     | Paper Trading           | Connecter bot à Testnet, exécuter signaux, logger trades                 |
| 9-10    | Dashboard + Ajustements | Streamlit dashboard, comparer backtest vs paper, ajustements             |
| 11-12   | Préparation Live        | Tests finaux, validation risque, préparation infra (RPi ou cloud)        |
| 13+     | LIVE                    | Lancer le bot avec capital réel                                          |

---

## ✅ Checklist avant de coder

- [ ] Installation Python + dépendances
- [ ] Clé API Binance (pour fetch + live)
- [ ] Structure projet créée
- [ ] Données historiques téléchargées
- [ ] Première stratégie implémentée
- [ ] Backtest lancé (même avec données partielles)
- [ ] Dashboard Streamlit basique en place
- [ ] Logs CSV générés
- [ ] Pipeline complet testée en "dry run"

---

## 🚨 Points d'attention

1. **Overfitting** : validation croisée obligatoire pour éviter les pièges
2. **Slippage & Frais** : dans le backtest, simule au moins 0.1 % de frais (frais Binance)
3. **Look-ahead bias** : assure-toi que les signaux ne "cheaten" pas avec des données futures
4. **Risque capital** : position sizing strict (ex: max 2 % du capital par trade)
5. **Monitoring paper** : 4+ semaines minimum avant de passer en live
6. **Infrastructure** : une fois en live, assure la stabilité (RPi ou micro-VM cloud)

---

## 📚 Ressources utiles

- **VectorBT docs** : https://vectorbt.pro/
- **Binance API** : https://binance-docs.github.io/apidocs/
- **TA library** : https://github.com/bukosabino/ta
- **Streamlit docs** : https://docs.streamlit.io/
- **CCXT** : https://docs.ccxt.com/