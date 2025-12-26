# src/modules/models.py

from typing import List
from neuralforecast.models import NBEATS, NHITS, LSTM

class ModelFactory:
    """
    Fábrica de modelos de previsão.

    Centraliza a criação e configuração dos modelos utilizados
    nos experimentos.
    """
    def __init__(self, 
                 input_size: int, 
                 horizon: int, 
                 max_steps: int) -> None:
        """
        Inicializa uma instância da classe ModelFactory

        Parameters
        ----------
        input_size : int
            Tamanho da janela de entrada.
        horizon : int
            Horizonte de previsão.
        max_steps : int
            Número máximo de iterações de treino.
        """
        self.input_size = input_size
        self.horizon = horizon
        self.max_steps = max_steps

    def build(self) -> List[object]:
        """
        Instancia os modelos configurados.

        Returns
        -------
        List[object]
            Lista de modelos NeuralForecast.
        """
        return [
            NBEATS(input_size=self.input_size, h=self.horizon, max_steps=self.max_steps),
            NHITS(input_size=self.input_size, h=self.horizon, max_steps=self.max_steps),
            LSTM(input_size=self.input_size, h=self.horizon, max_steps=self.max_steps),
        ]