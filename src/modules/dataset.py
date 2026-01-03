# src/modules/dataset.py

import pandas as pd                 # Manipulação de dados tabulares
import xarray as xr                 # Módulo para leitura de arquivos NetCDF
from typing import List, Optional   # Tipagem estática

class StationDataset:
    """
    Representa os dados de uma estação hidrológica específica.

    Converte um xarray.Dataset em um DataFrame no formato
    esperado por modelos de séries temporais.
    """

    def __init__(self, 
                 ds: xr.Dataset, 
                 station_id: str):
        """
        Inicializa uma instância da classe StationDataset

        Parameters
        ----------
        ds : xr.Dataset
            Dataset contendo as variáveis da estação.
        station_id : str
            Identificador único da estação.
        """
        
        # Armazena os atributos
        self.ds = ds
        self.station_id = station_id

    def to_dataframe(self,
                     time_col: str,
                     target_col: Optional[str] = None,
                     exog_cols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Converte o dataset em DataFrame no padrão:
        ['unique_id', 'ds', 'y', <exógenas>]

        Parameters
        ----------
        time_col : str
            Nome da variável temporal no Dataset.
        target_col : str
            Nome da variável alvo (ex: vazão).
        exog_cols : Optional[List[str]]
            Lista de variáveis exógenas opcionais.

        Returns
        -------
        pd.DataFrame
            DataFrame ordenado e sem valores nulos no alvo.
        """
       
        time_index = pd.to_datetime(self.ds[time_col].values)

        # Cria estrutura base exigida pelo NeuralForecast:
        # - unique_id
        # - ds (tempo)
        # - y (variável alvo)
        data = {
            "unique_id": self.station_id,
            "ds": time_index,
        }
        if target_col is not None:
            data["y"] = self.ds[target_col].to_series().values

        # Adiciona variáveis exógenas, se existirem
        if exog_cols:
            for col in exog_cols:
                if col in self.ds:
                    data[col] = self.ds[col].to_series().values

        # Constrói o DataFrame
        df = pd.DataFrame(data)

        # Remove registros sem valor observado apenas se 'y' existir
        if "y" in df.columns:
            df = df.dropna(subset=["y"])

        # Ordena temporalmente e reseta índice
        df = df.sort_values(["unique_id", "ds"]).reset_index(drop=True)

        # Retorna o DataFrame final
        return df