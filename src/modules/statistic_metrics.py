# src/modules/statistic_metrics.py

import numpy as np          # Importa a biblioteca NumPy para operações numéricas vetorizadas
from typing import Union    # Importa Union para permitir múltiplos tipos nos type hints

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

        # Conversão de y_true e y_pred para um array NumPy de ponto flutuante
        # garantindo compatibilidade com operações matemáticas
        self.y_true = np.asarray(y_true, dtype=float)
        self.y_pred = np.asarray(y_pred, dtype=float)

    @property
    def rmse(self) -> float:
        """
        Calcula o RMSE (Root Mean Squared Error).

        Mede a raiz do erro quadrático médio entre observado e previsto,
        penalizando erros grandes de forma mais intensa.

        Returns
        -------
        float
            Valor do RMSE.
        """

        # Calcula o erro quadrático ponto a ponto
        # ignora valores NaN usando nanmean
        return np.sqrt(np.nanmean((self.y_true - self.y_pred) ** 2))

    @property
    def mae(self) -> float:
        """
        Calcula o MAE (Mean Absolute Error).

        Mede o erro médio absoluto entre observado e previsto,
        sendo menos sensível a outliers que o RMSE.

        Returns
        -------
        float
            Valor do MAE.
        """

        # Calcula o valor absoluto da diferença entre as séries
        # e obtém a média ignorando NaNs
        return np.nanmean(np.abs(self.y_true - self.y_pred))

    @property
    def nse(self) -> float:
        """
        Calcula o NSE (Nash-Sutcliffe Efficiency).

        Métrica amplamente usada em hidrologia que compara
        o desempenho do modelo com a média dos valores observados.

        Returns
        -------
        float
            Valor do NSE (ideal = 1, negativo indica mau desempenho).
        """

        # Calcula o denominador da fórmula do NSE:
        # a variância dos valores observados
        denom = np.nanmean((self.y_true - np.nanmean(self.y_true))**2)
        
        # Evita divisão por zero caso a série observada seja constante
        if denom == 0:
            return np.nan
        
        # Calcula o NSE conforme definição clássica
        return 1.0 - (np.nanmean((self.y_true - self.y_pred)**2) / denom)

    @property
    def kge(self) -> float:
        """
        Calcula o KGE (Kling-Gupta Efficiency).

        Métrica que combina:
        - Correlação (r)
        - Razão de variâncias (alpha)
        - Razão de médias (beta)

        Returns
        -------
        float
            Valor do KGE (ideal = 1).
        """
        
        # Cria uma máscara booleana para remover pares com NaN
        mask = ~np.isnan(self.y_true) & ~np.isnan(self.y_pred)
        
        # Se não houver dados válidos após a filtragem, retorna NaN
        if mask.sum() == 0:
            return np.nan

        # Aplica a máscara para alinhar corretamente as séries
        y_true = self.y_true[mask]
        y_pred = self.y_pred[mask]

        # Calcula o coeficiente de correlação de Pearson
        r = np.corrcoef(y_true, y_pred)[0, 1]
        
        # Se a correlação não puder ser calculada, retorna NaN
        if np.isnan(r):
            return np.nan

        # Calcula alpha: razão entre os desvios padrão
        alpha = np.std(y_pred) / np.std(y_true) if np.std(y_true) != 0 else np.nan

        # Calcula beta: razão entre as médias
        beta = np.mean(y_pred) / np.mean(y_true) if np.mean(y_true) != 0 else np.nan

        # Se qualquer componente for inválido, retorna NaN
        if np.isnan(alpha) or np.isnan(beta):
            return np.nan

        # Calcula o KGE conforme a formulação original
        return float(1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))