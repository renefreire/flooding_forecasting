# src/modules/models.py

from typing import List                                 # Tipagem estática
from neuralforecast.models import NBEATS, NHITS, LSTM   # Importa modelos de previsão

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
        
        # Define parâmetros comuns aos modelos
        self.input_size = input_size
        self.horizon = horizon
        self.max_steps = max_steps

    def build(self) -> List[object]:
        """
        Cria e retorna a lista de modelos.

        Returns
        -------
        List[object]
            Lista de modelos NeuralForecast.
        """
        
        # Instancia cada modelo com parâmetros padronizados
        return [
            NBEATS(input_size=self.input_size, h=self.horizon, max_steps=self.max_steps),
            NHITS(input_size=self.input_size, h=self.horizon, max_steps=self.max_steps),
            LSTM(input_size=self.input_size, h=self.horizon, max_steps=self.max_steps),
        ]