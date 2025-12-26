# src/modules/evaluation.py

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from modules.statistic_metrics import StatisticsMetrics


class Evaluator:
    """
    Orquestra a avaliação de modelos de previsão,
    realizando o alinhamento entre observado e previsto
    e calculando métricas estatísticas.
    """

    def __init__(self,
                 df_true: pd.DataFrame,
                 cv_results: pd.DataFrame) -> None:
        """
        Inicializa uma instância da classe Evaluator

        Parameters
        ----------
        df_true : pd.DataFrame
            DataFrame com colunas ['unique_id', 'ds', 'y'].
        cv_results : pd.DataFrame
            Saída do método cross_validation do NeuralForecast.
        """
        self.df_true = df_true
        self.cv_results = cv_results

    def evaluate(self,
                 model_names: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Avalia os modelos especificados.

        Parameters
        ----------
        model_names : List[str]
            Lista com os nomes das colunas de previsão.

        Returns
        -------
        Dict[str, Dict[str, float]]
            Métricas agregadas por modelo.
        """
        results = {}

        for model in model_names:
            y_true, y_pred = self.extract_series(model)

            metrics = StatisticsMetrics(y_true, y_pred)

            results[model] = {
                'RMSE': metrics.rmse,
                'MAE': metrics.mae,
                'NSE': metrics.nse,
                'KGE': metrics.kge
            }

        return results

    def extract_series(self, 
                       model_name: str) -> Tuple[np.ndarray,np.ndarray]:
        """
        Alinha séries observadas e previstas por chave temporal.

        Parameters
        ----------
        model_name : str
            Nome da coluna de previsão.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Vetores (y_true, y_pred).
        """
        merged = self.cv_results[['unique_id', 'ds', model_name]].merge(
            self.df_true[['unique_id', 'ds', 'y']],
            on=['unique_id', 'ds'],
            how='left'
        )

        return merged['y'].values, merged[model_name].values

    def summary(self,
                metrics: Dict[str, Dict[str, float]]) -> None:
        """
        Imprime no console um resumo das métricas.

        Parameters
        ----------
        metrics : Dict[str, Dict[str, float]]
            Métricas por modelo.
        """
        for model, values in metrics.items():
            print(f'\nModel: {model}')
            for metric_name, metric_value in values.items():
                if metric_value is None or np.isnan(metric_value):
                    print(f'  {metric_name}: NaN')
                else:
                    print(f'  {metric_name}: {metric_value:.4f}')