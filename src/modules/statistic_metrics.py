# src/modules/statistic_metrics.py

import numpy as np
from typing import Union

class StatisticsMetrics:
    """
    Calcula métricas estatísticas de desempenho
    entre séries observadas e previstas.
    """

    def __init__(self,
                 y_true: Union[np.ndarray, list],
                 y_pred: Union[np.ndarray, list]) -> None:
        """
        Inicializa uma instância da classe StatisticsMetrics

        Parameters
        ----------
        y_true : array-like
            Valores observados.
        y_pred : array-like
            Valores previstos.
        """
        self.y_true = np.asarray(y_true, dtype=float)
        self.y_pred = np.asarray(y_pred, dtype=float)

    @property
    def rmse(self) -> float:
        """Root Mean Squared Error."""
        return np.sqrt(np.nanmean((self.y_true - self.y_pred) ** 2))

    @property
    def mae(self) -> float:
        """Mean Absolute Error."""
        return np.nanmean(np.abs(self.y_true - self.y_pred))

    @property
    def nse(self):
        """Nash-Sutcliffe Efficiency."""
        denom = np.nanmean((self.y_true - np.nanmean(self.y_true))**2)
        if denom == 0:
            return np.nan
        return 1.0 - (np.nanmean((self.y_true - self.y_pred)**2) / denom)

    @property
    def kge(self):
        """Kling-Gupta Efficiency (original formulation)"""
        mask = ~np.isnan(self.y_true) & ~np.isnan(self.y_pred)
        if mask.sum() == 0:
            return np.nan

        y_true = self.y_true[mask]
        y_pred = self.y_pred[mask]

        r = np.corrcoef(y_true, y_pred)[0, 1]
        if np.isnan(r):
            return np.nan

        alpha = np.std(y_pred) / np.std(y_true) if np.std(y_true) != 0 else np.nan
        beta = np.mean(y_pred) / np.mean(y_true) if np.mean(y_true) != 0 else np.nan

        if np.isnan(alpha) or np.isnan(beta):
            return np.nan

        return float(1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))