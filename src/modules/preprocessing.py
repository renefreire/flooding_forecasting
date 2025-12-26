# src/modules/preprocessing.py

import pandas as pd

class Preprocessor:
    """
    Classe responsável pelo pré-processamento dos dados.

    Implementa a interface fit/transform para permitir
    extensões futuras (normalização, filtros, etc.).
    """

    def __init__(self) -> None:
        """
        Inicializa uma instância da classe Preprocessor
        """
        self.fitted = False

    def fit(self, 
            df: pd.DataFrame) -> "Preprocessor":
        """
        Ajusta o pré-processador aos dados.

        Parameters
        ----------
        df : pd.DataFrame
            Dados de entrada.

        Returns
        -------
        Preprocessor
            Instância ajustada.
        """
        self.fitted = True
        return self

    def transform(self, 
                  df: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica transformações aos dados.

        Parameters
        ----------
        df : pd.DataFrame
            Dados de entrada.

        Returns
        -------
        pd.DataFrame
            Dados transformados.
        """
        if not self.fitted:
            raise RuntimeError("Pre-processador não foi ajustado.")
        return df

    def fit_transform(self, 
                      df: pd.DataFrame) -> pd.DataFrame:
        """
        Ajusta e transforma os dados em uma única etapa.

        Parameters
        ----------
        df : pd.DataFrame
            Dados de entrada.

        Returns
        -------
        pd.DataFrame
            Dados transformados.
        """
        return self.fit(df).transform(df)
