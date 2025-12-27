# src/modules/report.py

import sys
import os
from datetime import datetime
import numpy as np
from typing import Optional, Sequence, Any
import matplotlib.pyplot as plt
import pandas as pd

class ReportLogger:
    """
    Context manager responsável por redirecionar a saída padrão (stdout)
    para um arquivo texto, opcionalmente mantendo o eco no console.

    Permite que qualquer chamada a `print()` dentro do bloco `with`
    seja automaticamente salva em um arquivo de relatório.
    """

    def __init__(self, 
                 filepath: str, 
                 echo: bool = True) -> None:
        """
        Inicializa o logger do relatório.

        Parameters
        ----------
        filepath : str
            Caminho completo do arquivo .txt onde o relatório será salvo.
        echo : bool, optional
            Se True, imprime também no console além de salvar no arquivo.
        """
        
        # Armazena o caminho do arquivo de relatório
        self.filepath = filepath

        # Define se a saída também será exibida no console
        self.echo = echo
        
        # Guarda uma referência ao stdout original
        self._stdout = sys.stdout
        
        # Inicializa o handler do arquivo como None
        self._file = None

    def __enter__(self) -> "ReportLogger":
        """
        Entra no contexto do logger, abrindo o arquivo e
        redirecionando o stdout.

        Returns
        -------
        ReportLogger
            Instância ativa do logger.
        """
        
        # Abre o arquivo de relatório em modo escrita
        self._file = open(self.filepath, "w", encoding="utf-8")

        # Redireciona o stdout para esta instância
        sys.stdout = self

        # Retorna o próprio logger para uso no bloco with
        return self

    def __exit__(self, 
                 exc_type: Optional[type],
                 exc_val: Optional[BaseException],
                 exc_tb: Optional[Any]) -> None:
        """
        Sai do contexto do logger, restaurando o stdout original
        e fechando o arquivo de saída.

        Parameters
        ----------
        exc_type : type | None
            Tipo da exceção, se ocorreu.
        exc_val : BaseException | None
            Valor da exceção.
        exc_tb : Any | None
            Traceback da exceção.
        """
        
        # Restaura o stdout original do Python
        sys.stdout = self._stdout

        # Fecha o arquivo de relatório se estiver aberto
        if self._file is not None:
            self._file.close()

    def write(self, 
              message: str) -> None:
        """
        Escreve uma mensagem no arquivo de relatório
        e opcionalmente no console.

        Parameters
        ----------
        message : str
            Texto a ser escrito.
        """

        # Escreve no console se echo estiver habilitado
        if self.echo:
            self._stdout.write(message)
        
        # Escreve no arquivo de relatório
        if self._file is not None:
            self._file.write(message)

    def flush(self) -> None:
        """
        Garante que os buffers de escrita sejam descarregados.
        """
        
        # Força a escrita imediata no console se echo estiver habilitado
        if self.echo:
            self._stdout.flush()

        # Força a escrita imediata no arquivo
        self._file.flush()

def plot_time_series_observed_vs_predicted(
        df: pd.DataFrame,
        time_col: str,
        y_true_col: str,
        y_pred_col: str,
        output_path: str,
        title: Optional[str] = "Observed vs Predicted Streamflow"
    ) -> None:
    """
    Gera e salva um gráfico temporal da vazão observada e prevista,
    agregando os dados em média diária.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame contendo os dados observados e previstos.
    time_col : str
        Nome da coluna temporal (datetime).
    y_true_col : str
        Nome da coluna com valores observados.
    y_pred_col : str
        Nome da coluna com valores previstos.
    output_path : str
        Caminho completo onde o gráfico será salvo (.png).
    title : str, optional
        Título do gráfico.
    """

    # Cria uma cópia para evitar efeitos colaterais
    df = df.copy()

    # Garante que a coluna temporal esteja no formato datetime
    df[time_col] = pd.to_datetime(df[time_col])

    # Manter apenas colunas necessárias e numéricas
    df = df[[time_col, y_true_col, y_pred_col]]

    # Converte as séries para valores numéricos (strings viram NaN)
    df[y_true_col] = pd.to_numeric(df[y_true_col], errors="coerce")
    df[y_pred_col] = pd.to_numeric(df[y_pred_col], errors="coerce")

    # Define a coluna temporal como índice do DataFrame
    df = df.set_index(time_col)

    ## Cria a figura do gráfico
    plt.figure(figsize=(12, 5))

    # Plota a série observada
    plt.plot(df.index, df[y_true_col], label="Observed", linewidth=2)

    # Plota a série prevista
    plt.plot(df.index, df[y_pred_col], label="Predicted", linestyle="--")

    # Configurações visuais
    plt.xlabel("Date")
    plt.ylabel("Streamflow")
    plt.title(title)
    plt.legend()
    plt.grid(True)

    # Ajusta layout e salva a figura
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_scatter_observed_vs_predicted(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        output_path: str,
        title: Optional[str] = "Observed vs Predicted"
    ) -> None:
    """
    Gera e salva um gráfico de dispersão entre valores observados
    e valores previstos.

    Parameters
    ----------
    y_true : np.ndarray
        Série de valores observados.
    y_pred : np.ndarray
        Série de valores previstos.
    output_path : str
        Caminho completo onde a figura será salva (.png).
    title : str, optional
        Título do gráfico.
    """
    
    # Converte entradas para arrays numéricos
    y_true = np.asarray(pd.to_numeric(y_true, errors="coerce"), dtype=float)
    y_pred = np.asarray(pd.to_numeric(y_pred, errors="coerce"), dtype=float)

    # Remove pares com NaN
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    # Cria figura
    plt.figure(figsize=(6, 6))

    # Gráfico de dispersão
    plt.scatter(y_true, y_pred, alpha=0.5)

    # Linha 1:1 (previsão perfeita)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")

    # Configurações visuais
    plt.xlabel("Observed")
    plt.ylabel("Predicted")
    plt.title(title)
    plt.grid(True)
    
    # Ajusta layout e salva a figura 
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_error_histogram(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        output_path: str,
        bins: int = 30,
        title: Optional[str] = "Prediction Errors (y_true - y_pred)"
    ) -> None:
    """
    Gera e salva um histograma dos erros de previsão.

    Parameters
    ----------
    y_true : np.ndarray
        Valores observados.
    y_pred : np.ndarray
        Valores previstos.
    output_path : str
        Caminho completo onde a figura será salva (.png).
    bins : int, optional
        Número de classes do histograma.
    title : str, optional
        Título do gráfico.
    """
    
    # Conversão para float e limpeza de NaN
    y_true = np.asarray(pd.to_numeric(y_true, errors="coerce"), dtype=float)
    y_pred = np.asarray(pd.to_numeric(y_pred, errors="coerce"), dtype=float)

    # Cálculo do erro (observado - previsto)
    errors = y_true - y_pred
    errors = errors[~np.isnan(errors)]

    # Criação do histograma
    plt.figure(figsize=(7, 4))
    plt.hist(errors, bins=bins, edgecolor="black", alpha=0.7)
    
    # Linha vertical no erro zero
    plt.axvline(0.0, linestyle="--")

    # Configurações visuais
    plt.xlabel("Error")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.grid(True)

    # Ajusta layout e salva a figura
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def save_report(
        station: Any,
        nc_folder: str,
        files: Sequence[str],
        df: pd.DataFrame,
        experiment: Any,
        cv_results: pd.DataFrame,
        metrics: dict[str, dict[str, float]],
        evaluator: Any
    ) -> None:
    """
    Gera um relatório completo do experimento, incluindo:
    - Informações do dataset
    - Configuração dos modelos
    - Resultados da validação cruzada
    - Métricas de desempenho
    - Gráficos de avaliação salvos em disco

    Parameters
    ----------
    station : Any
        Objeto de estação (ex.: StationDataset).
    nc_folder : str
        Caminho da pasta de dados NetCDF.
    files : Sequence[str]
        Lista de arquivos NetCDF encontrados.
    df : pd.DataFrame
        DataFrame de entrada utilizado no experimento.
    experiment : Any
        Instância de ForecastExperiment.
    cv_results : pd.DataFrame
        Resultados da validação cruzada.
    metrics : dict[str, dict[str, float]]
        Métricas agregadas por modelo.
    evaluator : Any
        Objeto responsável pela avaliação (Evaluator).
    """
    
    # Diretório base do módulo
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # Diretório onde os resultados serão salvos
    results_dir = os.path.join(BASE_DIR, "..", "results")

    # Timestamp único para nomear arquivos
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Caminho do arquivo de relatório
    report_path = os.path.join(results_dir,
                               "reports",
                               f"forecast_report_{timestamp}.txt")

    # Redireciona stdout para o relatório
    with ReportLogger(report_path, echo=True):
        
        # Cabeçalho
        print("=" * 60)
        print(" RELATÓRIO DE EXPERIMENTO – FLOOD FORECASTING ")
        print("=" * 60)

        # Informações gerais
        print(f"Estação: {station.station_id}")
        print(f"Data/Hora: {timestamp}\n")
        print(f"Pasta dataset: {nc_folder}")
        print(f"Arquivos encontrados: {len(files)}")
        print(f"Arquivo usado: {files[0]}\n")

        # Visualização inicial dos dados
        print("Pré-visualização dos dados:")
        print(df.head(), "\n")
        print("Pré-processamento concluído.\n")
        
        # Lista de modelos
        print("Modelos utilizados:")
        for m in experiment.models:
            print(f" - {m.__class__.__name__}")
        print()

        # Resultados da validação cruzada
        print("Resultados da validação cruzada:")
        print(cv_results.head(), "\n")

        # Métricas
        print("Resumo das métricas:")
        evaluator.summary(metrics)

        # Mensagem de geração dos gráficos
        print("\nGeração de gráficos:")

        # Loop por modelo
        model_names = [m.__class__.__name__ for m in experiment.models]
        for model_name in model_names:

            # Merge entre observado e previsto
            merged = cv_results[['unique_id', 'ds', model_name]].merge(
                df[['unique_id', 'ds', 'y']],
                on=['unique_id', 'ds'],
                how='left'
            )

            # Extração das séries
            y_true = merged['y'].values
            y_pred = merged[model_name].values

            # Caminhos das figuras
            timeseries_path = os.path.join(
                results_dir,
                "timeseries",
                f"time_{model_name}_{timestamp}.png"
            )
            scatter_path = os.path.join(
                results_dir,
                "scatter",
                f"scatter_{model_name}_{timestamp}.png"
            )
            hist_path = os.path.join(
                results_dir,
                "histogram",
                f"hist_{model_name}_{timestamp}.png"
            )

            # Geração dos gráficos
            plot_time_series_observed_vs_predicted(
                df=merged,
                time_col="ds",
                y_true_col="y",
                y_pred_col=model_name,
                output_path=timeseries_path,
                title=f"Daily Streamflow – {model_name}"
            )           
            plot_scatter_observed_vs_predicted(
                y_true=y_true,
                y_pred=y_pred,
                output_path=scatter_path,
                title=f"Observed vs Predicted – {model_name}"
            )
            plot_error_histogram(
                y_true=y_true,
                y_pred=y_pred,
                output_path=hist_path,
                title=f"Prediction Errors – {model_name}"
            )

            # Log no relatório
            print(f" - {model_name}:")
            print(f"   Série Temporal: {timeseries_path}")
            print(f"   Dispersão: {scatter_path}")
            print(f"   Histograma: {hist_path}")

        # Mensagem de finalização da execução
        print("\nExecução finalizada com sucesso.")