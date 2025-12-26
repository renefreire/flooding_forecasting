# src/modules/data_io.py

import os
import glob
import xarray as xr
from typing import List, Optional

class DataIO:
    """
    Responsável por operações de entrada e saída de dados em formato NetCDF (.nc).

    Permite listar arquivos, abrir um único dataset ou combinar múltiplos datasets
    localizados em um diretório.
    """
    
    def __init__(self, 
                 folder: str, 
                 engine: Optional[str] = None) -> None:
        """
        Inicializa uma instância da classe DataIO

        Parameters
        ----------
        folder : str
            Caminho para o diretório contendo arquivos .nc.
        engine : Optional[str]
            Engine do xarray para leitura dos arquivos (ex: 'netcdf4').
        """
        self.folder = os.path.abspath(folder)
        self.engine = engine

    def list_files(self) -> list[str]:
        """
        Lista todos os arquivos .nc no diretório.

        Returns
        -------
        List[str]
            Lista ordenada de caminhos completos dos arquivos .nc.

        Raises
        ------
        FileNotFoundError
            Se nenhum arquivo .nc for encontrado.
        """
        files = glob.glob(os.path.join(self.folder, "*.nc"))
        if not files:
            raise FileNotFoundError(f"Nenhum .nc encontrado em {self.folder}")
        return sorted(files)

    def open_dataset(self, 
                     path: str) -> xr.Dataset:
        """
        Abre um único arquivo NetCDF como xarray.Dataset.

        Parameters
        ----------
        path : str
            Caminho para o arquivo .nc.

        Returns
        -------
        xr.Dataset
            Dataset carregado em memória.
        """
        return xr.open_dataset(path, engine=self.engine)

    def open_all(self) -> xr.Dataset:
        """
        Abre e combina todos os arquivos .nc do diretório.

        Returns
        -------
        xr.Dataset
            Dataset combinado por coordenadas.
        """
        pattern = os.path.join(self.folder, "*.nc")
        return xr.open_mfdataset(pattern,
                                 combine="by_coords",
                                 parallel=True)