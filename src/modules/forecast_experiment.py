# src/modules/forecast_experiment.py

import pandas as pd
from neuralforecast import NeuralForecast
from modules.models import ModelFactory


class ForecastExperiment:
    """
    Encapsula a configuração e execução de experimentos
    de previsão com NeuralForecast.
    """

    def __init__(self, 
                 freq: str, 
                 input_size: int, 
                 horizon: int, 
                 max_steps: int) -> None:
        """
        Inicializa uma instância da classe ForecastExperiment
        Parameters
        ----------
        freq : str
            Frequência temporal (ex: 'H').
        input_size : int
            Janela de entrada.
        horizon : int
            Horizonte de previsão.
        max_steps : int
            Número máximo de iterações de treino.
        """
        self.freq = freq
        self.factory = ModelFactory(input_size, horizon, max_steps)
        self.models = self.factory.build()
        self.nf = NeuralForecast(models=self.models, freq=freq)

    def cross_validate(self, 
                       df: pd.DataFrame, 
                       n_windows: int, 
                       step_size: int) -> pd.DataFrame:
        """
        Executa validação cruzada em janela deslizante.

        Parameters
        ----------
        df : pd.DataFrame
            Série temporal no formato NeuralForecast.
        n_windows : int
            Número de janelas.
        step_size : int
            Passo entre janelas.

        Returns
        -------
        pd.DataFrame
            Resultados da validação cruzada.
        """
        return self.nf.cross_validation(df=df,
                                        n_windows=n_windows,
                                        step_size=step_size)