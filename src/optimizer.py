import itertools
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Type, Any, Tuple
from src.backtest_engine import BacktestEngine
from src.strategies import Strategy

logger = logging.getLogger(__name__)

def generate_param_combinations(param_grid: Dict[str, List]) -> List[Dict[str, Any]]:
    """
    Génère toutes les combinaisons de paramètres possibles pour une Grid Search.
    
    Args:
        param_grid: Dictionnaire {param_name: [liste_valeurs]}
        
    Returns:
        Liste de dictionnaires {param_name: valeur}
    """
    keys = param_grid.keys()
    values = param_grid.values()
    combinations = []
    
    for combination in itertools.product(*values):
        combinations.append(dict(zip(keys, combination)))
        
    return combinations

def walk_forward_validation(
    data: pd.DataFrame,
    strategy_class: Type[Strategy],
    param_grid: Dict[str, List],
    pair: str,
    train_size: float = 0.7,
    backtester_params: Dict = None
) -> Dict:
    """
    Walk-forward analysis pour valider sans overfitting.
    
    Divise les données en fenêtres :
    - Train (70%) : optimise les paramètres
    - Test (30%) : valide les résultats (NO OPTIMIZATION HERE)
    """
    if backtester_params is None:
        backtester_params = {}

    n = len(data)
    train_end = int(n * train_size)
    
    train_data = data.iloc[:train_end].copy()
    test_data = data.iloc[train_end:].copy()
    
    logger.info(f"Walk-Forward: Train rows={len(train_data)}, Test rows={len(test_data)}")
    
    # Phase 1 : Grid search sur TRAIN uniquement
    best_params = None
    best_sharpe = -np.inf
    
    combinations = generate_param_combinations(param_grid)
    logger.info(f"Testing {len(combinations)} combinations on Train set...")
    
    backtester = BacktestEngine(**backtester_params)
    
    # Pour stocker tous les résultats TRAIN pour analyse
    train_results_list = []

    for params in combinations:
        try:
            # Instantiate strategy with current params
            # Note: Strategy __init__ might require 'name' or other args not in grid.
            # Assuming params match __init__ args except 'name' which is usually fixed or optional
            strategy = strategy_class(**params)
            
            # Generate signals
            # Note: BacktestEngine.run generates signals internally if we pass the raw DF
            # But here walk_forward handles the data split.
            # BacktestEngine.run expects a dataframe with 'close'. 
            # It also calls strategy.generate_signals(df). 
            # So we just pass the raw train_data to backtester.run, BUT we need to make sure 
            # the strategy instance has the updated params.
            
            # Actually, BacktestEngine.run takes (df, strategy, pair).
            results = backtester.run(train_data, strategy, pair)
            
            sharpe = results['sharpe_ratio']
            train_results_list.append({**params, 'sharpe': sharpe, 'total_return': results['total_return']})

            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_params = params
        except Exception as e:
             logger.warning(f"Error testing params {params}: {e}")
             continue
    
    if best_params is None:
        logger.error("No valid combination found on Train set.")
        return {'error': 'Optimization failed'}

    logger.info(f"✅ Best params (TRAIN): {best_params}, Sharpe={best_sharpe:.2f}")
    
    # Phase 2 : Valide sur TEST (sans optimiser) avec les MEILLEURS params du Train
    strategy_test = strategy_class(**best_params)
    # BacktestEngine.run handles signal generation
    test_results = backtester.run(test_data, strategy_test, pair)
    
    test_sharpe = test_results['sharpe_ratio']
    logger.info(f"Validation (TEST): Sharpe={test_sharpe:.2f}")
    
    # Vérifie pas d'overfitting
    # Si Sharpe Train >> Sharpe Test => Overfitting probable
    sharpe_diff = best_sharpe - test_sharpe
    is_overfitting = sharpe_diff > 1.0  # Seuil arbitraire, à ajuster
    
    if is_overfitting:
        logger.warning(f"⚠️  POSSIBLE OVERFITTING : Train Sharpe={best_sharpe:.2f} vs Test Sharpe={test_sharpe:.2f}")
    
    return {
        'best_params': best_params,
        'train_stats': {'sharpe': best_sharpe},
        'test_stats': {
            'sharpe': test_sharpe,
            'total_return': test_results['total_return'],
            'max_drawdown': test_results['max_drawdown'],
            'win_rate': test_results['win_rate']
        },
        'overfitting_detected': 0.0 if np.isinf(sharpe_diff) or np.isnan(sharpe_diff) else float(sharpe_diff),
        'is_overfitting': bool(is_overfitting),
        'all_train_results': train_results_list
    }
